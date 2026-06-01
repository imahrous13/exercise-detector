import os
import subprocess
import tempfile

import cv2
import numpy as np

# COCO skeleton connections for drawing
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]

# Colors for different body parts (BGR)
COLORS = {
    'head': (255, 200, 100),     # light blue
    'arms': (100, 255, 100),     # green
    'torso': (100, 100, 255),    # red
    'legs': (255, 100, 255),     # purple
}

LIMB_COLORS = [
    COLORS['head'], COLORS['head'], COLORS['head'], COLORS['head'],  # head
    COLORS['torso'],                                                    # shoulders
    COLORS['arms'], COLORS['arms'], COLORS['arms'], COLORS['arms'],  # arms
    COLORS['torso'], COLORS['torso'], COLORS['torso'],                # torso
    COLORS['legs'], COLORS['legs'], COLORS['legs'], COLORS['legs'],  # legs
]


def draw_person_boxes(frame, all_bboxes, tracked_idx=-1):
    """Draw bounding boxes around all detected people.

    The tracked person gets a green box; others get gray boxes with
    'Click to select' label.

    Args:
        frame: BGR image
        all_bboxes: list of (x1, y1, x2, y2) tuples
        tracked_idx: index of currently tracked person (-1 if none)

    Returns:
        frame with boxes drawn
    """
    frame = frame.copy()
    for i, bbox in enumerate(all_bboxes):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if i == tracked_idx:
            color = (0, 255, 0)  # green for tracked
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "TRACKING", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            color = (128, 128, 128)  # gray for others
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, "Click to select", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return frame


def draw_skeleton(frame, keypoints, confidence_threshold=0.3):
    """Draw skeleton overlay on a frame.

    Args:
        frame: BGR image (H, W, 3)
        keypoints: (17, 3) array with [x, y, confidence]
        confidence_threshold: minimum confidence to draw a joint

    Returns:
        frame with skeleton drawn on it
    """
    frame = frame.copy()

    # Draw connections (bones)
    for i, (src, dst) in enumerate(SKELETON_CONNECTIONS):
        if (keypoints[src, 2] > confidence_threshold and
                keypoints[dst, 2] > confidence_threshold):
            pt1 = (int(keypoints[src, 0]), int(keypoints[src, 1]))
            pt2 = (int(keypoints[dst, 0]), int(keypoints[dst, 1]))
            color = LIMB_COLORS[i]
            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw joints
    for j in range(17):
        if keypoints[j, 2] > confidence_threshold:
            pt = (int(keypoints[j, 0]), int(keypoints[j, 1]))
            cv2.circle(frame, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 4, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def draw_feedback(frame, exercise_name, is_correct, feedback_messages,
                  rep_count=None, correct_reps=None, incorrect_reps=None,
                  form_score=None):
    """Draw exercise classification, form feedback, and rep counter on frame.

    Args:
        frame: BGR image (H, W, 3)
        exercise_name: detected exercise string
        is_correct: bool indicating form correctness
        feedback_messages: list of feedback strings
        rep_count: optional total repetition count (legacy, still supported)
        correct_reps: number of correct repetitions (shown in green)
        incorrect_reps: number of incorrect repetitions (shown in red)
        form_score: optional 0-100 form score

    Returns:
        frame with text overlay
    """
    frame = frame.copy()
    h, w = frame.shape[:2]

    # Count extra lines needed for rep counter row
    has_reps = correct_reps is not None or incorrect_reps is not None or rep_count is not None
    extra_lines = 1 if has_reps else 0

    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10),
                  (w - 10, 40 + 25 * (len(feedback_messages) + 2 + extra_lines)),
                  (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    y_offset = 35

    # Exercise name
    exercise_text = f"Exercise: {exercise_name.replace('_', ' ').title()}"
    cv2.putText(frame, exercise_text, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 28

    # Form status with score
    if is_correct:
        status_text = "Form: CORRECT"
        status_color = (0, 255, 0)  # green
    else:
        status_text = "Form: NEEDS CORRECTION"
        status_color = (0, 0, 255)  # red

    if form_score is not None:
        status_text += f" ({form_score}/100)"

    cv2.putText(frame, status_text, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    y_offset += 25

    # Rep counter
    if correct_reps is not None and incorrect_reps is not None:
        # Draw correct count in green
        correct_text = f"Reps  OK: {correct_reps}"
        cv2.putText(frame, correct_text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Draw incorrect count in red next to it
        incorrect_text = f"  BAD: {incorrect_reps}"
        # Measure width of correct_text to position incorrect_text
        text_size = cv2.getTextSize(correct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(frame, incorrect_text, (20 + text_size[0], y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25
    elif rep_count is not None:
        # Legacy single rep count
        cv2.putText(frame, f"Reps: {rep_count}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25

    # Feedback messages
    for msg in feedback_messages:
        cv2.putText(frame, f"  - {msg}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 180, 255), 1)
        y_offset += 22

    return frame


def _get_ffmpeg_path():
    """Get path to ffmpeg binary (bundled via imageio-ffmpeg or system)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"  # hope it's on PATH


def _remux_to_h264(src_path, dst_path, fps):
    """Re-encode an mp4v video to H.264 so browsers can play it."""
    ffmpeg = _get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-i", src_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        dst_path,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True, timeout=300)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def create_output_video(frames, keypoints_list, results_list, output_path, fps=30.0):
    """Create output video with skeleton overlay, feedback, and rep counter.

    Writes with OpenCV (mp4v codec), then re-encodes to H.264 so the video
    is playable in browsers and Streamlit's st.video().

    Args:
        frames: list of BGR images
        keypoints_list: list of (17, 3) keypoint arrays (raw, not normalized)
        results_list: list of dicts with 'exercise', 'is_correct', 'feedback',
                      and optionally 'correct_reps', 'incorrect_reps'
        output_path: path to save output video
        fps: output frame rate
    """
    if len(frames) == 0:
        return

    h, w = frames[0].shape[:2]

    # Step 1: Write with OpenCV (mp4v — not browser-compatible)
    tmp_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    for i, frame in enumerate(frames):
        # Draw skeleton
        if i < len(keypoints_list):
            frame = draw_skeleton(frame, keypoints_list[i])

        # Draw feedback + rep counter
        if i < len(results_list) and results_list[i] is not None:
            result = results_list[i]
            frame = draw_feedback(
                frame,
                exercise_name=result.get('exercise', 'unknown'),
                is_correct=result.get('is_correct', True),
                feedback_messages=result.get('feedback', []),
                correct_reps=result.get('correct_reps'),
                incorrect_reps=result.get('incorrect_reps'),
                form_score=result.get('form_score'),
            )

        writer.write(frame)

    writer.release()

    # Step 2: Re-encode to H.264 for browser playback
    output_str = str(output_path)
    if _remux_to_h264(tmp_path, output_str, fps):
        # H.264 conversion succeeded
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    else:
        # Fallback: just move the mp4v file (may not play in browser)
        import shutil
        shutil.move(tmp_path, output_str)
