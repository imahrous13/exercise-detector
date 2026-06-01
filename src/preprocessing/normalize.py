import numpy as np


# COCO keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# 12 joint angle definitions: (point_a, vertex_b, point_c)
# Angle is measured at vertex_b
ANGLE_DEFINITIONS = [
    (L_HIP, L_SHOULDER, L_ELBOW),       # 0: L-Shoulder angle
    (R_HIP, R_SHOULDER, R_ELBOW),       # 1: R-Shoulder angle
    (L_SHOULDER, L_ELBOW, L_WRIST),     # 2: L-Elbow angle
    (R_SHOULDER, R_ELBOW, R_WRIST),     # 3: R-Elbow angle
    (L_SHOULDER, L_HIP, L_KNEE),        # 4: L-Hip angle
    (R_SHOULDER, R_HIP, R_KNEE),        # 5: R-Hip angle
    (L_HIP, L_KNEE, L_ANKLE),           # 6: L-Knee angle
    (R_HIP, R_KNEE, R_ANKLE),           # 7: R-Knee angle
    (L_KNEE, L_HIP, L_SHOULDER),        # 8: L-Trunk angle
    (R_KNEE, R_HIP, R_SHOULDER),        # 9: R-Trunk angle
    (L_HIP, L_ANKLE, L_KNEE),           # 10: L-Ankle angle
    (R_HIP, R_ANKLE, R_KNEE),           # 11: R-Ankle angle
]


def _compute_single_angle(a, b, c):
    """Compute angle at vertex b formed by points a-b-c.

    Args:
        a, b, c: 2D points as (x, y) arrays

    Returns:
        Angle in degrees [0, 180]
    """
    ba = a - b
    bc = c - b
    dot_product = np.dot(ba, bc)
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cosine = np.clip(dot_product / norm_product, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def recenter_skeleton(keypoints):
    """Recenter skeleton so hip center is at origin.

    Args:
        keypoints: (T, 17, 3) array with [x, y, confidence]

    Returns:
        (T, 17, 3) recentered keypoints (confidence unchanged)
    """
    result = keypoints.copy()
    # Hip center = midpoint of left and right hip
    hip_center_x = (keypoints[:, L_HIP, 0] + keypoints[:, R_HIP, 0]) / 2  # (T,)
    hip_center_y = (keypoints[:, L_HIP, 1] + keypoints[:, R_HIP, 1]) / 2  # (T,)

    # Subtract hip center from all joints' x and y
    result[:, :, 0] -= hip_center_x[:, np.newaxis]
    result[:, :, 1] -= hip_center_y[:, np.newaxis]
    # Confidence channel stays the same

    return result


def normalize_scale(keypoints):
    """Scale-normalize skeleton by torso length (shoulder-to-hip distance).

    Args:
        keypoints: (T, 17, 3) array (should be recentered first)

    Returns:
        (T, 17, 3) scale-normalized keypoints
    """
    result = keypoints.copy()

    # Mid-shoulder and mid-hip positions
    mid_shoulder_x = (keypoints[:, L_SHOULDER, 0] + keypoints[:, R_SHOULDER, 0]) / 2
    mid_shoulder_y = (keypoints[:, L_SHOULDER, 1] + keypoints[:, R_SHOULDER, 1]) / 2
    mid_hip_x = (keypoints[:, L_HIP, 0] + keypoints[:, R_HIP, 0]) / 2
    mid_hip_y = (keypoints[:, L_HIP, 1] + keypoints[:, R_HIP, 1]) / 2

    # Torso length per frame
    torso_length = np.sqrt(
        (mid_shoulder_x - mid_hip_x) ** 2 +
        (mid_shoulder_y - mid_hip_y) ** 2
    )  # (T,)

    # Handle zero-length frames: use previous frame's value or fallback to 1
    for i in range(len(torso_length)):
        if torso_length[i] < 1e-6:
            torso_length[i] = torso_length[i - 1] if i > 0 else 1.0

    # Divide x, y by torso length
    result[:, :, 0] /= torso_length[:, np.newaxis]
    result[:, :, 1] /= torso_length[:, np.newaxis]

    return result


def compute_angles(keypoints):
    """Compute 12 joint angles for each frame.

    Args:
        keypoints: (T, 17, 3) array with [x, y, confidence]

    Returns:
        (T, 12) array of angles in degrees [0, 180]
    """
    T = keypoints.shape[0]
    angles = np.zeros((T, 12), dtype=np.float32)

    for t in range(T):
        for idx, (a_idx, b_idx, c_idx) in enumerate(ANGLE_DEFINITIONS):
            a = keypoints[t, a_idx, :2]  # x, y only
            b = keypoints[t, b_idx, :2]
            c = keypoints[t, c_idx, :2]
            angles[t, idx] = _compute_single_angle(a, b, c)

    return angles


def preprocess_skeleton(keypoints):
    """Full preprocessing pipeline: recenter -> scale normalize -> compute angles.

    Args:
        keypoints: (T, 17, 3) raw keypoints from YOLOv8

    Returns:
        normalized_keypoints: (T, 17, 3) preprocessed skeleton
        angles: (T, 12) joint angles in degrees
    """
    kpts = recenter_skeleton(keypoints)
    kpts = normalize_scale(kpts)
    angles = compute_angles(keypoints)  # Use original coordinates for angles
    return kpts, angles
