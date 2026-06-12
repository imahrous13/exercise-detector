"""Rule-based rep SEGMENTATION core logic (boundaries only — no form judging).

This module turns already-extracted pose skeletons + joint angles into a smoothed
per-exercise "rep driver signal" and detects where each repetition starts and
ends. It deliberately does NOT classify form quality (no form_label / no
mistake_type) — that is intentionally left to a separate/manual step.

Used by ``scripts/auto_segment_reps.py``.

Design principles
-----------------
* Conservative: when evidence is weak, the caller routes the video to review
  rather than emitting guessed reps.
* Adaptive: detection thresholds come from per-video robust percentiles of the
  driver signal, not only fixed magic numbers.
* Self-contained: only NumPy is required; SciPy is used when available for
  smoothing / peak finding, otherwise pure-NumPy fallbacks are used.

Data assumptions (validated at load time)
-----------------------------------------
* Skeleton ``.npy`` shape ``(T, 17, C)`` with C in {3, 6}; channels are
  ``x, y, confidence[, vx, vy, bone_length]`` (COCO-17 layout).
  Coordinates are hip-recentered and torso-scale-normalized (see
  ``src/preprocessing/normalize.py``), so vertical signals are scale invariant.
* Angle ``.npy`` shape ``(T, 12)`` in degrees, layout defined in
  ``src/preprocessing/normalize.py`` (ANGLE_DEFINITIONS).

The 12 angle indices (degrees, measured at the middle joint):
    0: L-Shoulder   1: R-Shoulder
    2: L-Elbow      3: R-Elbow
    4: L-Hip        5: R-Hip
    6: L-Knee       7: R-Knee
    8: L-Trunk      9: R-Trunk
   10: L-Ankle     11: R-Ankle
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional SciPy acceleration (smoothing / peak finding). Pure-NumPy fallbacks
# are provided so the module works even without SciPy installed.
try:  # pragma: no cover - import guard
    from scipy.ndimage import median_filter as _scipy_median_filter
    from scipy.signal import find_peaks as _scipy_find_peaks
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Joint / angle index constants (COCO-17 + the 12-angle layout above)
# ---------------------------------------------------------------------------
KP = {
    "l_shoulder": 5, "r_shoulder": 6,
    "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10,
    "l_hip": 11, "r_hip": 12,
}

ANGLE_IDX = {
    "shoulder": {"L": 0, "R": 1},
    "elbow": {"L": 2, "R": 3},
    "hip": {"L": 4, "R": 5},
    "trunk": {"L": 8, "R": 9},
}


# ---------------------------------------------------------------------------
# Per-exercise biomechanical specification
# ---------------------------------------------------------------------------
# primary       : which joint angle drives rep detection
# rest_is_high  : True  -> at rest the driver angle is HIGH and the active phase
#                          is a downward dip (e.g. biceps curl: arm starts
#                          extended ~160 deg, curls down to ~50 deg, returns).
#                 False -> at rest the driver angle is LOW and the active phase
#                          is an upward peak (e.g. triceps pushdown / shoulder
#                          press: starts flexed/low, extends/presses up, returns).
# good_rom_deg  : expected full range of motion (deg) for a clean rep; used only
#                 for FORM judgement, not for detection (detection is adaptive).
# incomplete_frac: rep ROM below this fraction of good_rom_deg => incomplete_range.
# min_rom_deg   : if the whole video's driver ROM is below this, the video has too
#                 little movement to label reliably -> review_needed.
EXERCISE_REP_SPEC: Dict[str, dict] = {
    # NOTE: shoulder_press is driven by the ELBOW angle, not the shoulder angle.
    # In a single 2D view the hip-shoulder-elbow angle barely changes for many
    # camera positions, whereas elbow extension (bent ~90 deg -> locked overhead
    # ~175 deg) is large and reliably detected. You cannot complete a press
    # without extending the elbows overhead, so elbow extension is a sound proxy.
    "biceps":         {"primary": "elbow",    "rest_is_high": True,  "good_rom_deg": 70.0,  "incomplete_frac": 0.55, "min_rom_deg": 35.0, "bilateral": False},
    "bench_press":    {"primary": "elbow",    "rest_is_high": True,  "good_rom_deg": 45.0,  "incomplete_frac": 0.55, "min_rom_deg": 25.0, "bilateral": True},
    "shoulder_press": {"primary": "elbow",    "rest_is_high": False, "good_rom_deg": 70.0,  "incomplete_frac": 0.55, "min_rom_deg": 30.0, "bilateral": True},
    "triceps":        {"primary": "elbow",    "rest_is_high": False, "good_rom_deg": 60.0,  "incomplete_frac": 0.55, "min_rom_deg": 30.0, "bilateral": False},
}


@dataclass
class DetectConfig:
    """Tunable detection parameters (overridable from the CLI / config)."""

    fps: float = 30.0
    min_rep_seconds: float = 0.5      # below this a rep is impossible -> rejected
    max_rep_seconds: float = 8.0      # above this a rep is impossible -> rejected
    good_min_seconds: float = 0.8     # below -> tempo confidence penalty
    good_max_seconds: float = 4.0     # above -> tempo confidence penalty
    smoothing_window: int = 7         # median/EMA smoothing window (frames, odd)
    max_gap_fill: int = 5             # interpolate confidence gaps up to this many frames
    side_min_confidence: float = 0.30 # a side is usable only above this mean confidence
    cooldown_seconds: float = 0.30    # min gap between consecutive rep peaks
    # --- rep_start / rep_end onset refinement (leaves/returns to rest zone) ---
    start_margin_frac: float = 0.12   # signal must move this fraction of local ROM
                                      # away from rest to count as "movement begun"
    start_confirm_frames: int = 4     # movement must continue in-direction this many frames
    onset_vel_eps: float = 0.20       # deg/frame: while velocity exceeds this we keep
                                      # walking back to the true onset of the rise


# ---------------------------------------------------------------------------
# Stage 1 — loading
# ---------------------------------------------------------------------------
def load_skeleton_and_angles(
    filename: str,
    skeleton_dir: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Load skeleton ``(T,17,C)`` and angles ``(T,12)`` for one video stem.

    Returns ``(skeleton, angles, error)``. On success ``error`` is "" and arrays
    are non-None. On failure arrays are None and ``error`` explains why.
    """
    skel_path = os.path.join(skeleton_dir, f"{filename}.npy")
    ang_path = os.path.join(skeleton_dir, f"{filename}_angles.npy")

    if not os.path.isfile(skel_path):
        return None, None, "missing skeleton .npy"

    try:
        skel = np.load(skel_path).astype(np.float32)
    except Exception as e:  # pragma: no cover
        return None, None, f"failed to load skeleton: {e}"

    if skel.ndim != 3 or skel.shape[1] != 17 or skel.shape[2] < 3:
        return None, None, f"unexpected skeleton shape {skel.shape}"

    angles: Optional[np.ndarray] = None
    if os.path.isfile(ang_path):
        try:
            angles = np.load(ang_path).astype(np.float32)
        except Exception:
            angles = None

    # Recompute angles from keypoints if the angle file is missing/mismatched.
    if angles is None or angles.shape[0] != skel.shape[0] or angles.shape[1] < 12:
        try:
            from src.preprocessing.normalize import compute_angles
            angles = compute_angles(skel[:, :, :3]).astype(np.float32)
        except Exception as e:  # pragma: no cover
            return None, None, f"no angles and failed to compute: {e}"

    return skel, angles, ""


# ---------------------------------------------------------------------------
# Stage 2 — smoothing / cleaning
# ---------------------------------------------------------------------------
def _median_filter_1d(x: np.ndarray, window: int) -> np.ndarray:
    """1-D median filter with reflect padding (SciPy if available)."""
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    if _HAS_SCIPY:
        return _scipy_median_filter(x, size=window, mode="nearest")
    half = window // 2
    padded = np.pad(x, half, mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(padded[i:i + window])
    return out


def _ema(x: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Causal-then-symmetric exponential moving average (forward+backward)."""
    fwd = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc = alpha * x[i] + (1 - alpha) * acc
        fwd[i] = acc
    bwd = np.empty_like(x)
    acc = fwd[-1]
    for i in range(len(x) - 1, -1, -1):
        acc = alpha * fwd[i] + (1 - alpha) * acc
        bwd[i] = acc
    return bwd


def interpolate_gaps(signal: np.ndarray, valid: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate short invalid runs; leave long gaps as-is (NaN-safe).

    Args:
        signal: 1-D values.
        valid: boolean mask, True where the value is trustworthy.
        max_gap: only fill consecutive invalid runs up to this length.
    """
    out = signal.astype(np.float64).copy()
    n = len(out)
    if valid.all() or not valid.any():
        return out.astype(np.float32)

    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        # invalid run is [i, j)
        run_len = j - i
        left = i - 1
        right = j
        if 0 <= left and right < n and run_len <= max_gap:
            lv, rv = out[left], out[right]
            for k in range(i, j):
                t = (k - left) / (right - left)
                out[k] = lv + t * (rv - lv)
        i = j
    return out.astype(np.float32)


def smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
    """Median filter (kills spikes) then symmetric EMA (kills jitter)."""
    if len(signal) < 3:
        return signal.astype(np.float32)
    med = _median_filter_1d(signal.astype(np.float64), window)
    return _ema(med, alpha=0.4).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 1b — per-video pose quality
# ---------------------------------------------------------------------------
def assess_pose_quality(skel: np.ndarray, cfg: DetectConfig) -> Tuple[float, float, str]:
    """Return ``(avg_confidence, jitter, reason)`` for the upper-body joints.

    ``reason`` is non-empty when the video should go to review for quality.
    """
    conf = skel[:, :, 2]
    upper = [KP["l_shoulder"], KP["r_shoulder"], KP["l_elbow"], KP["r_elbow"],
             KP["l_wrist"], KP["r_wrist"]]
    avg_conf = float(np.mean(conf[:, upper]))

    # Jitter: mean absolute frame-to-frame motion of the wrists (torso units).
    wr = skel[:, [KP["l_wrist"], KP["r_wrist"]], :2]
    if wr.shape[0] > 1:
        jitter = float(np.mean(np.abs(np.diff(wr, axis=0))))
    else:
        jitter = 0.0

    reason = ""
    if avg_conf < cfg.side_min_confidence:
        reason = f"low avg pose confidence ({avg_conf:.2f})"
    elif jitter > 0.5:
        reason = f"excessive pose jitter ({jitter:.2f})"
    return avg_conf, jitter, reason


# ---------------------------------------------------------------------------
# Stage 3 — side selection
# ---------------------------------------------------------------------------
def _robust_rom(x: np.ndarray) -> float:
    """ROM as p90 - p10 (robust to outliers)."""
    if len(x) == 0:
        return 0.0
    return float(np.percentile(x, 90) - np.percentile(x, 10))


def _side_confidence(skel: np.ndarray, side: str) -> float:
    js = [KP[f"{side}_shoulder"], KP[f"{side}_elbow"], KP[f"{side}_wrist"]]
    return float(np.mean(skel[:, js, 2]))


def select_side(
    skel: np.ndarray,
    angles: np.ndarray,
    joint: str,
    cfg: DetectConfig,
) -> Tuple[str, np.ndarray, dict]:
    """Pick the more reliable body side for the driver angle.

    Returns ``(side, driver_signal, info)`` where side is "L", "R", or "LR"
    (combined). ``info`` holds per-side confidence/ROM for the report.
    """
    li, ri = ANGLE_IDX[joint]["L"], ANGLE_IDX[joint]["R"]
    l_ang = smooth_signal(angles[:, li], cfg.smoothing_window)
    r_ang = smooth_signal(angles[:, ri], cfg.smoothing_window)

    l_conf = _side_confidence(skel, "l")
    r_conf = _side_confidence(skel, "r")
    l_rom = _robust_rom(l_ang)
    r_rom = _robust_rom(r_ang)

    info = {
        "l_conf": round(l_conf, 3), "r_conf": round(r_conf, 3),
        "l_rom": round(l_rom, 1), "r_rom": round(r_rom, 1),
    }

    l_ok = l_conf >= cfg.side_min_confidence
    r_ok = r_conf >= cfg.side_min_confidence

    # Both sides reliable and similar ROM -> combine for a cleaner signal.
    if l_ok and r_ok:
        denom = max(l_rom, r_rom, 1e-6)
        if abs(l_rom - r_rom) / denom <= 0.30:
            return "LR", (l_ang + r_ang) / 2.0, info
        return ("L", l_ang, info) if l_rom >= r_rom else ("R", r_ang, info)
    if l_ok:
        return "L", l_ang, info
    if r_ok:
        return "R", r_ang, info
    # Neither side confident — fall back to the higher-confidence side so the
    # caller can still inspect, but quality gating will route it to review.
    return ("L", l_ang, info) if l_conf >= r_conf else ("R", r_ang, info)


# ---------------------------------------------------------------------------
# Stage 5/6 — adaptive thresholds + rep detection state machine
# ---------------------------------------------------------------------------
def _find_peaks_np(x: np.ndarray, min_distance: int, height: float) -> List[int]:
    """Pure-NumPy local-maxima finder with a height floor and min spacing."""
    peaks: List[int] = []
    n = len(x)
    for i in range(1, n - 1):
        if x[i] >= height and x[i] >= x[i - 1] and x[i] > x[i + 1]:
            if peaks and (i - peaks[-1]) < min_distance:
                # keep the taller of the two close peaks
                if x[i] > x[peaks[-1]]:
                    peaks[-1] = i
                continue
            peaks.append(i)
    return peaks


def _refine_start(sig: np.ndarray, v_left: int, pk: int, cfg: DetectConfig) -> int:
    """Frame where movement clearly LEAVES the rest zone (not the valley/peak).

    ``sig`` is the transformed driver where the active phase rises to a peak, so
    rest is the local minimum ``v_left`` before the peak. We find the first frame
    whose rise above rest exceeds an adaptive margin AND keeps rising for several
    frames (jitter rejection), then walk back along the still-rising slope to the
    true onset of the movement.
    """
    if pk <= v_left:
        return v_left
    base = float(sig[v_left])
    local_rom = float(sig[pk]) - base
    if local_rom <= 1e-6:
        return v_left
    margin = max(cfg.start_margin_frac * local_rom, 1.0)  # >= 1 deg
    confirm = max(1, cfg.start_confirm_frames)

    for t in range(v_left, pk):
        if sig[t] - base >= margin:
            # confirm the movement continues upward (not a one-frame spike)
            k = min(pk, t + confirm)
            if sig[k] - sig[t] <= 0:
                continue
            onset = t
            # back off to the true onset: while the signal is still rising,
            # step back toward the rest valley (but never past it)
            while onset > v_left and (sig[onset] - sig[onset - 1]) > cfg.onset_vel_eps:
                onset -= 1
            return onset
    return v_left


def _refine_end(sig: np.ndarray, pk: int, v_right: int, cfg: DetectConfig) -> int:
    """Frame where movement RETURNS to the rest zone after the peak.

    Mirror of ``_refine_start``: scan forward from the peak for the first frame
    that has come back within the rest margin, then settle forward while the
    signal is still descending so rep_end sits at the resting return point.
    """
    if v_right <= pk:
        return v_right
    base = float(sig[v_right])
    local_rom = float(sig[pk]) - base
    if local_rom <= 1e-6:
        return v_right
    margin = max(cfg.start_margin_frac * local_rom, 1.0)

    for t in range(pk, v_right + 1):
        if sig[t] - base <= margin:
            settle = t
            while settle < v_right and (sig[settle - 1] - sig[settle]) > cfg.onset_vel_eps:
                settle += 1
            return settle
    return v_right


def detect_reps(
    driver: np.ndarray,
    rest_is_high: bool,
    cfg: DetectConfig,
) -> Tuple[List[Tuple[int, int, int]], dict]:
    """Detect FULL-cycle reps: rest -> active extreme -> rest.

    The driver is transformed so the active phase is always an UPWARD peak. Peaks
    (the mid-rep extreme) are found with an adaptive height floor; for each peak
    the surrounding rest valleys are located, and the boundaries are then refined
    so that:

        rep_start = first frame the body LEAVES the resting/start position
                    (onset of movement — NOT the valley and NOT the peak)
        rep_end   = first frame the body RETURNS to the resting/start position

    Returns ``(reps, info)`` where each rep is ``(rep_start, peak, rep_end)`` in
    frame indices, and ``info`` holds the adaptive thresholds for the report.
    """
    n = len(driver)
    info: dict = {}
    if n < 5:
        return [], info

    # Transform so "active" is a peak above the rest baseline.
    sig = (driver.max() - driver) if rest_is_high else driver.copy()

    p10, p50, p90 = (float(np.percentile(sig, q)) for q in (10, 50, 90))
    rom = p90 - p10
    info.update({"p10": round(p10, 1), "p50": round(p50, 1),
                 "p90": round(p90, 1), "rom": round(rom, 1)})
    if rom < 1e-3:
        return [], info

    height = p10 + 0.55 * rom          # a real active extreme must exceed this
    low_zone = p10 + 0.30 * rom        # rest zone (hysteresis) — localizes the
                                       # immediate trough around each peak
    min_dist = max(1, int(cfg.cooldown_seconds * cfg.fps))

    if _HAS_SCIPY:
        idx, _ = _scipy_find_peaks(sig, height=height, distance=min_dist)
        peaks = [int(i) for i in idx]
    else:
        peaks = _find_peaks_np(sig, min_dist, height)

    reps: List[Tuple[int, int, int]] = []
    prev_end = 0
    for i, pk in enumerate(peaks):
        # Search bounds: between the neighbouring peaks (and never inside the
        # previous rep — quality check: rep_start not inside the previous rep).
        left_bound = max(prev_end, peaks[i - 1] if i > 0 else 0)
        right_bound = peaks[i + 1] if i < len(peaks) - 1 else n - 1
        if pk <= left_bound or pk >= right_bound:
            continue

        # IMMEDIATE rest trough before the peak: drop into the rest zone via
        # hysteresis, then descend to the nearest local minimum. Using the
        # immediate trough (not the global minimum) prevents a rep from
        # swallowing earlier shallow/partial movements before it.
        s = pk
        while s > left_bound and sig[s] > low_zone:
            s -= 1
        while s > left_bound and sig[s - 1] < sig[s]:
            s -= 1
        v_left = s

        # IMMEDIATE rest trough after the peak (mirror of the above).
        e = pk
        while e < right_bound and sig[e] > low_zone:
            e += 1
        while e < right_bound and sig[e + 1] < sig[e]:
            e += 1
        v_right = e

        rep_start = _refine_start(sig, v_left, pk, cfg)   # leaves rest position
        rep_end = _refine_end(sig, pk, v_right, cfg)      # returns to rest position
        if rep_end <= rep_start:
            continue
        reps.append((rep_start, pk, rep_end))
        prev_end = rep_end
    return reps, info
