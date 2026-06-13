import cv2
import numpy as np
from ultralytics import YOLO


def _smooth_keypoints_ema(keypoints, alpha=0.35):
    """Apply exponential moving average to reduce temporal jitter.

    Args:
        keypoints: (T, 17, 3) array [x, y, confidence]
        alpha: blend factor (0=full smooth, 1=no smooth). Lower = more stable.

    Returns:
        smoothed (T, 17, 3) array
    """
    if keypoints is None or len(keypoints) < 2:
        return keypoints
    out = keypoints.copy().astype(np.float32)
    for t in range(1, len(out)):
        # Blend x,y; keep confidence as max of current and previous
        out[t, :, :2] = alpha * keypoints[t, :, :2] + (1 - alpha) * out[t - 1, :, :2]
        out[t, :, 2] = np.maximum(keypoints[t, :, 2], out[t - 1, :, 2])
    return out


def _bbox_from_keypoints(kpts):
    """Compute bounding box from keypoints (only visible ones).

    Args:
        kpts: (17, 3) array [x, y, confidence]

    Returns:
        (x1, y1, x2, y2) bounding box or None if no visible keypoints
    """
    visible = kpts[:, 2] > 0.1
    if not visible.any():
        return None
    xs = kpts[visible, 0]
    ys = kpts[visible, 1]
    return (xs.min(), ys.min(), xs.max(), ys.max())


def _bbox_area(bbox):
    """Compute area of a bounding box."""
    if bbox is None:
        return 0
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_iou(a, b):
    """Compute IoU between two bounding boxes."""
    if a is None or b is None:
        return 0.0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_center(bbox):
    """Return center (cx, cy) of a bounding box."""
    if bbox is None:
        return None
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _point_in_bbox(px, py, bbox, margin=30):
    """Check if point is inside bbox with margin."""
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    return (x1 - margin) <= px <= (x2 + margin) and (y1 - margin) <= py <= (y2 + margin)


def _crop_to_roi(frame, bbox, padding_ratio=0.2):
    """Crop frame to bbox with padding. Always returns a valid crop (clamped to image).

    Args:
        frame: (H, W, 3) BGR image
        bbox: (x1, y1, x2, y2)
        padding_ratio: fraction of bbox width/height to add on each side (e.g. 0.2 = 20%)

    Returns:
        cropped: (h, w, 3) image
        offset_x, offset_y: top-left of crop in full-frame coordinates (so keypoints can be converted)
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_w = max(1, bw * padding_ratio)
    pad_h = max(1, bh * padding_ratio)
    cx1 = max(0, int(x1 - pad_w))
    cy1 = max(0, int(y1 - pad_h))
    cx2 = min(w, int(x2 + pad_w))
    cy2 = min(h, int(y2 + pad_h))
    if cx2 <= cx1 or cy2 <= cy1:
        return frame, 0, 0
    cropped = frame[cy1:cy2, cx1:cx2].copy()
    return cropped, cx1, cy1


def _keypoints_crop_to_full(kpts, offset_x, offset_y):
    """Convert keypoints from crop coordinates to full-frame coordinates."""
    out = kpts.copy()
    out[..., 0] += offset_x
    out[..., 1] += offset_y
    return out


class PoseExtractor:
    """Extract human pose keypoints from video using YOLOv8 Pose model.

    Features:
        - ROI cropping: pose is always run on a cropped region (person bbox + padding) when roi_padding_ratio > 0
        - ByteTrack for stable person tracking across frames (video mode)
        - Temporal EMA smoothing to reduce keypoint jitter
        - Selects the largest person by bounding box area (video mode)
        - Supports click-to-select a specific person (webcam mode)
        - IoU-based frame-to-frame tracking (webcam fallback)
        - Falls back to largest person if tracking is lost

    Uses COCO 17-keypoint format:
        0:Nose, 1:L-Eye, 2:R-Eye, 3:L-Ear, 4:R-Ear,
        5:L-Shoulder, 6:R-Shoulder, 7:L-Elbow, 8:R-Elbow,
        9:L-Wrist, 10:R-Wrist, 11:L-Hip, 12:R-Hip,
        13:L-Knee, 14:R-Knee, 15:L-Ankle, 16:R-Ankle
    """

    KEYPOINT_NAMES = [
        'Nose', 'L-Eye', 'R-Eye', 'L-Ear', 'R-Ear',
        'L-Shoulder', 'R-Shoulder', 'L-Elbow', 'R-Elbow',
        'L-Wrist', 'R-Wrist', 'L-Hip', 'R-Hip',
        'L-Knee', 'R-Knee', 'L-Ankle', 'R-Ankle'
    ]

    IOU_THRESHOLD = 0.3  # minimum IoU to consider same person

    def __init__(self, model_size='s', confidence_threshold=0.5, smoothing_alpha=0.35, roi_padding_ratio=0.2):
        """Initialize PoseExtractor.

        Args:
            model_size: YOLOv8 size ('n','s','m','l','x'). Larger = more stable.
            confidence_threshold: minimum hip confidence to accept a frame.
            smoothing_alpha: EMA blend factor (0.2-0.5). Lower = smoother, 0 = disable.
            roi_padding_ratio: padding around bbox when cropping ROI before pose (e.g. 0.2 = 20%). 0 = disable ROI crop.
        """
        self.model = YOLO(f'yolov8{model_size}-pose.pt')
        self.confidence_threshold = confidence_threshold
        self.smoothing_alpha = smoothing_alpha
        self.roi_padding_ratio = roi_padding_ratio
        # Tracking state (IoU-based for webcam)
        self._tracked_bbox = None
        # ByteTrack state (for video)
        self._tracked_id = None
        # Live smoothing (for extract_from_frame)
        self._prev_keypoints = None

    def reset_tracking(self):
        """Reset the tracked person (call when switching videos/modes)."""
        self._tracked_bbox = None
        self._tracked_id = None
        self._prev_keypoints = None

    def select_person_at(self, frame, click_x, click_y):
        """Select the person at the given (x, y) click position.

        Runs detection on the frame and picks the person whose bbox
        contains the click point. Sets internal tracking state.

        Args:
            frame: BGR image
            click_x, click_y: click coordinates in pixel space

        Returns:
            keypoints (17, 3) of selected person, or None if no person at click
        """
        all_kpts, all_bboxes = self._detect_all(frame)
        if len(all_kpts) == 0:
            return None

        # Find person whose bbox contains the click
        for kpts, bbox in zip(all_kpts, all_bboxes):
            if _point_in_bbox(click_x, click_y, bbox):
                self._tracked_bbox = bbox
                return kpts

        # No exact hit — pick closest person by center distance
        min_dist = float('inf')
        best_kpts = None
        for kpts, bbox in zip(all_kpts, all_bboxes):
            center = _bbox_center(bbox)
            if center is None:
                continue
            dist = (center[0] - click_x) ** 2 + (center[1] - click_y) ** 2
            if dist < min_dist:
                min_dist = dist
                best_kpts = kpts
                self._tracked_bbox = bbox

        return best_kpts

    def extract_from_frame(self, frame):
        """Extract keypoints from a single frame with tracking.

        Always runs pose on a cropped ROI (person bbox + padding) when roi_padding_ratio > 0.
        If a person is being tracked, uses their bbox to crop; otherwise picks largest from full-frame detection.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            keypoints: (17, 3) array [x, y, confidence] in full-frame coordinates, or None
            roi_ox, roi_oy: crop offset for overlay (0, 0 when no detection)
        """
        all_kpts, all_bboxes = self._detect_all(frame)
        if len(all_kpts) == 0:
            return None, 0, 0

        # Choose which person: tracked or largest
        best_idx = -1
        if self._tracked_bbox is not None:
            best_iou = 0
            for i, bbox in enumerate(all_bboxes):
                iou = _bbox_iou(self._tracked_bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou < self.IOU_THRESHOLD:
                best_idx = -1
                self._prev_keypoints = None

        if best_idx < 0:
            best_idx = max(range(len(all_bboxes)), key=lambda i: _bbox_area(all_bboxes[i]))

        bbox = all_bboxes[best_idx]

        # Run pose on cropped ROI (or full frame if roi_padding_ratio is 0)
        kpts, new_bbox, ox, oy = self._run_pose_on_roi(frame, bbox)
        if kpts is None:
            return None, 0, 0
        self._tracked_bbox = new_bbox if new_bbox is not None else bbox

        if self.smoothing_alpha > 0 and self._prev_keypoints is not None:
            kpts = (
                self.smoothing_alpha * kpts
                + (1 - self.smoothing_alpha) * self._prev_keypoints
            ).astype(np.float32)
        self._prev_keypoints = kpts.copy()
        return kpts, ox, oy

    def extract_all_detections(self, frame):
        """Return all detected persons' keypoints and bboxes (for UI overlay).

        Args:
            frame: BGR image

        Returns:
            all_kpts: list of (17, 3) arrays
            all_bboxes: list of (x1, y1, x2, y2) tuples
            tracked_idx: index of the currently tracked person, or -1
        """
        all_kpts, all_bboxes = self._detect_all(frame)
        tracked_idx = -1

        if self._tracked_bbox is not None and len(all_bboxes) > 0:
            best_iou = 0
            for i, bbox in enumerate(all_bboxes):
                iou = _bbox_iou(self._tracked_bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    tracked_idx = i

            if best_iou < self.IOU_THRESHOLD:
                tracked_idx = -1

        return all_kpts, all_bboxes, tracked_idx

    def _detect_all(self, frame):
        """Run YOLOv8 and return all detected persons (full frame, for UI / person selection).

        Returns:
            all_kpts: list of (17, 3) numpy arrays
            all_bboxes: list of (x1, y1, x2, y2) tuples
        """
        results = self.model(frame, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            return [], []

        kpts_data = results[0].keypoints.data.cpu().numpy()
        if len(kpts_data) == 0:
            return [], []

        all_kpts = []
        all_bboxes = []
        for kpts in kpts_data:
            bbox = _bbox_from_keypoints(kpts)
            if bbox is not None:
                all_kpts.append(kpts)
                all_bboxes.append(bbox)

        return all_kpts, all_bboxes

    def _run_pose_on_roi(self, frame, bbox):
        """Run pose on a cropped ROI; return keypoints in crop coordinates (for model consistency).

        Returns (kpts, new_bbox_full, offset_x, offset_y). Keypoints are in crop space so the
        model sees a consistent, centered person. new_bbox_full is in full-frame for next crop.
        When roi_padding_ratio is 0, returns (kpts_full, bbox, 0, 0).
        """
        if self.roi_padding_ratio <= 0:
            results = self.model(frame, verbose=False)
            if len(results) == 0 or results[0].keypoints is None:
                return None, None, 0, 0
            kpts_data = results[0].keypoints.data.cpu().numpy()
            boxes = results[0].boxes
            if boxes is None or len(kpts_data) == 0:
                return None, None, 0, 0
            bboxes = [(b[0], b[1], b[2], b[3]) for b in boxes.xyxy.cpu().numpy()]
            best_iou = 0
            best_idx = -1
            for i, b in enumerate(bboxes):
                iou = _bbox_iou(bbox, b)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx < 0:
                return None, None, 0, 0
            return kpts_data[best_idx], bboxes[best_idx], 0, 0

        cropped, ox, oy = _crop_to_roi(frame, bbox, self.roi_padding_ratio)
        results = self.model(cropped, verbose=False)
        if len(results) == 0 or results[0].keypoints is None:
            return None, None, 0, 0
        kpts_data = results[0].keypoints.data.cpu().numpy()
        if len(kpts_data) == 0:
            return None, None, 0, 0
        kpts_crop = kpts_data[0]
        # Full-frame bbox for next frame's crop (from crop keypoints + offset)
        kpts_full = _keypoints_crop_to_full(kpts_crop, ox, oy)
        new_bbox = _bbox_from_keypoints(kpts_full)
        return kpts_crop, new_bbox, ox, oy

    def _extract_frame_with_track(self, frame):
        """Extract keypoints using ByteTrack; always run pose on cropped ROI when available.

        Returns (kpts, bbox, offset_x, offset_y). Keypoints are in crop space for model consistency.
        """
        if self._tracked_bbox is not None and self.roi_padding_ratio > 0:
            kpts, new_bbox, ox, oy = self._run_pose_on_roi(frame, self._tracked_bbox)
            if kpts is not None and new_bbox is not None:
                self._tracked_bbox = new_bbox
                return kpts, new_bbox, ox, oy

        try:
            results = self.model.track(
                frame, persist=True, verbose=False,
                tracker="bytetrack.yaml",
            )
        except Exception:
            results = self.model.track(frame, persist=True, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            return None, None, 0, 0

        kpts_data = results[0].keypoints.data.cpu().numpy()
        boxes = results[0].boxes
        if len(kpts_data) == 0:
            return None, None, 0, 0

        ids = boxes.id.cpu().numpy() if boxes.id is not None else np.full(len(kpts_data), -1)
        bboxes = [(b[0], b[1], b[2], b[3]) for b in boxes.xyxy.cpu().numpy()]

        chosen_idx = -1
        if self._tracked_id is not None:
            for i, tid in enumerate(ids):
                if tid == self._tracked_id:
                    chosen_idx = i
                    break
            if chosen_idx < 0:
                self._tracked_id = None

        if chosen_idx < 0:
            chosen_idx = max(range(len(bboxes)), key=lambda i: _bbox_area(bboxes[i]))

        bbox = bboxes[chosen_idx]
        tid = ids[chosen_idx]
        if tid is not None and int(tid) >= 0:
            self._tracked_id = int(tid)
        self._tracked_bbox = bbox

        if self.roi_padding_ratio > 0:
            kpts, new_bbox, ox, oy = self._run_pose_on_roi(frame, bbox)
            if kpts is not None and new_bbox is not None:
                self._tracked_bbox = new_bbox
                return kpts, new_bbox, ox, oy

        return kpts_data[chosen_idx], bbox, 0, 0

    def extract_from_video(self, video_path):
        """Extract keypoint sequence from a video, tracking the largest person.

        Uses ByteTrack for stable tracking and EMA smoothing for jitter reduction.

        Args:
            video_path: path to video file

        Returns:
            keypoints: (T, 17, 3) array
            fps: video frame rate
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        keypoint_sequence = []
        self.reset_tracking()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            kpts, _, _, _ = self._extract_frame_with_track(frame)
            if kpts is not None:
                hip_conf = (kpts[11, 2] + kpts[12, 2]) / 2
                if hip_conf >= self.confidence_threshold:
                    keypoint_sequence.append(kpts)

        cap.release()
        self.reset_tracking()

        if len(keypoint_sequence) == 0:
            raise ValueError(f"No valid poses detected in video: {video_path}")

        arr = np.array(keypoint_sequence, dtype=np.float32)
        if self.smoothing_alpha > 0:
            arr = _smooth_keypoints_ema(arr, alpha=self.smoothing_alpha)
        return arr, fps

    def extract_from_video_with_frames(self, video_path):
        """Extract keypoints along with original frames, tracking the largest person.

        Keypoints are in crop (ROI) coordinates so the model sees consistent input.
        Returns crop_offsets so callers can convert to full-frame for drawing.

        Returns:
            keypoints: (T, 17, 3) array in crop coordinates
            frames: list of BGR images (full frame)
            fps: video frame rate
            crop_offsets: (T, 2) array (offset_x, offset_y) per frame for drawing
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        keypoint_sequence = []
        frame_list = []
        offset_list = []
        self.reset_tracking()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            kpts, _, ox, oy = self._extract_frame_with_track(frame)
            if kpts is not None:
                hip_conf = (kpts[11, 2] + kpts[12, 2]) / 2
                if hip_conf >= self.confidence_threshold:
                    keypoint_sequence.append(kpts)
                    frame_list.append(frame)
                    offset_list.append((ox, oy))

        cap.release()
        self.reset_tracking()

        if len(keypoint_sequence) == 0:
            raise ValueError(f"No valid poses detected in video: {video_path}")

        arr = np.array(keypoint_sequence, dtype=np.float32)
        if self.smoothing_alpha > 0:
            arr = _smooth_keypoints_ema(arr, alpha=self.smoothing_alpha)
        crop_offsets = np.array(offset_list, dtype=np.float32)
        return arr, frame_list, fps, crop_offsets
