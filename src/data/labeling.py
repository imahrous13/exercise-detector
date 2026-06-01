"""Hybrid labeling: manual annotations first, rule-based fallback for unlabeled videos."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.annotations import (
    RepAnnotation,
    VideoAnnotations,
    label_window_from_reps,
    load_rep_annotations,
)
from src.feedback.form_rules import (
    NO_REP_EXERCISES,
    _PASS_THRESHOLDS,
    score_form,
    segment_reps,
)


def load_annotations_index_safe(
    path: Optional[str],
    exercise_names: Optional[List[str]] = None,
) -> Dict[str, VideoAnnotations]:
    """Load manual annotations; return empty dict if file missing or empty."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        return load_rep_annotations(path, exercise_names)
    except (ValueError, pd.errors.EmptyDataError, FileNotFoundError):
        return {}


def _resolve_manual_key(filename: str, index: Dict[str, VideoAnnotations]) -> Optional[str]:
    if filename in index and index[filename].reps:
        return filename
    for key, va in index.items():
        if not va.reps:
            continue
        if key == filename or filename.endswith(f"_{key}") or key.endswith(filename):
            return key
    return None


def reps_from_rules(
    exercise_name: str,
    angles: np.ndarray,
) -> List[RepAnnotation]:
    """Build rep segments with form labels from segment_reps + score_form."""
    if exercise_name in NO_REP_EXERCISES or angles is None:
        return []

    segments = segment_reps(exercise_name, angles)
    reps: List[RepAnnotation] = []
    for seg in segments:
        rep_start, rep_end = int(seg[0]), int(seg[1])
        is_correct = int(seg[2]) if len(seg) > 2 else 1
        reps.append(
            RepAnnotation(
                rep_start=rep_start,
                rep_end=rep_end,
                form_label=is_correct,
                mistake_type="rules_auto",
                exercise=exercise_name,
            )
        )
    return reps


def get_video_reps(
    filename: str,
    exercise_name: str,
    angles: Optional[np.ndarray],
    manual_index: Dict[str, VideoAnnotations],
    *,
    allow_rules_fallback: bool = True,
    cache: Optional[Dict[str, Tuple[List[RepAnnotation], str]]] = None,
) -> Tuple[List[RepAnnotation], str]:
    """Return rep list and source tag: 'manual' or 'rules' or 'none'.

    Args:
        filename: skeleton filename (no .npy)
        exercise_name: exercise class name
        angles: (T, 12) joint angles for rule fallback
        manual_index: loaded human annotations
        allow_rules_fallback: if False, only manual (empty when missing)
        cache: optional per-dataset cache dict
    """
    if cache is not None and filename in cache:
        return cache[filename]

    manual_key = _resolve_manual_key(filename, manual_index)
    if manual_key is not None:
        result = (list(manual_index[manual_key].reps), "manual")
        if cache is not None:
            cache[filename] = result
        return result

    if allow_rules_fallback and angles is not None:
        rules_reps = reps_from_rules(exercise_name, angles)
        result = (rules_reps, "rules" if rules_reps else "none")
        if cache is not None:
            cache[filename] = result
        return result

    result = ([], "none")
    if cache is not None:
        cache[filename] = result
    return result


def label_window(
    win_start: int,
    win_end: int,
    reps: List[RepAnnotation],
    *,
    exercise_name: Optional[str] = None,
    win_angles: Optional[np.ndarray] = None,
    use_completion_frame_for_rules: bool = False,
    label_source: str = "manual",
) -> Tuple[int, int, bool, Optional[str]]:
    """Label one sliding window from rep segments (manual or rule-derived).

    Default: overlap-based (same for manual and converted rule segments).
    Optional legacy rule rep head: rep_end in [win_start, win_end) when
    use_completion_frame_for_rules and label_source == 'rules'.
    """
    if use_completion_frame_for_rules and label_source == "rules" and reps:
        for rep in reps:
            if win_start <= rep.rep_end < win_end:
                return 1, rep.form_label, True, rep.mistake_type
        # No completion in window — still allow overlap form for partial reps
        rep_l, form_l, form_valid, mistake = label_window_from_reps(win_start, win_end, reps)
        if rep_l:
            return rep_l, form_l, form_valid, mistake
        if win_angles is not None and exercise_name:
            score, _ = score_form(exercise_name, win_angles)
            threshold = _PASS_THRESHOLDS.get(exercise_name, 60)
            return 0, 1 if score >= threshold else 0, False, "rules_auto"
        return 0, 0, False, None

    return label_window_from_reps(win_start, win_end, reps)


def labeling_config(config: Optional[dict]) -> dict:
    """Normalize data.labeling config with defaults."""
    labeling = (config or {}).get("data", {}).get("labeling", {})
    return {
        "source": labeling.get("source", "hybrid"),
        "annotations_file": labeling.get("annotations_file", "data/annotations/reps.csv"),
        "fallback_to_rules": labeling.get("fallback_to_rules", True),
        "rules_rep_completion_in_window": labeling.get(
            "rules_rep_completion_in_window", True,
        ),
    }
