"""Shared utilities for all dataset adapters.

Defines the unified annotation schema, exercise name normalisation table,
and the task-support matrix that documents which tasks each dataset can train.
"""

from __future__ import annotations

import re
import pandas as pd

# ---------------------------------------------------------------------------
# Unified schema
# ---------------------------------------------------------------------------

UNIFIED_COLUMNS = [
    "source_dataset",   # "captured" | "repcount" | "countix_fitness" |
                        # "mmfit" | "realtime" | "roboflow"
    "video",            # skeleton stem used by prepare_data.py (no .npy)
    "exercise",         # normalised exercise name (see CANONICAL_EXERCISES)
    "rep_start",        # int frame or ""  (empty = no per-rep timing)
    "rep_end",          # int frame or ""
    "form_label",       # 1=correct, 0=incorrect, "" = unknown
    "mistake_type",     # specific mistake string or ""
    "confidence",       # float 0–1  (annotation reliability estimate)
    "split",            # "train" | "val" | "test"
]

# Which tasks a row supports (derived at runtime from column completeness)
TASKS = ["exercise_classification", "rep_counting", "form_correction"]

# ---------------------------------------------------------------------------
# Canonical exercise names (superset of the 4-class project config)
# ---------------------------------------------------------------------------

CANONICAL_EXERCISES: list[str] = [
    # Project's primary 4
    "bench_press", "biceps", "shoulder_press", "triceps",
    # Common public-dataset exercises
    "squat", "push_up", "pull_up", "deadlift", "lunge",
    "sit_up", "jumping_jack", "plank", "leg_raise",
    "lat_pulldown", "hip_thrust", "romanian_deadlift",
    "barbell_row", "dumbbell_row", "cable_fly",
    "lateral_raise", "front_raise", "face_pull",
    "hammer_curl", "tricep_dip", "tricep_pushdown",
    "leg_press", "leg_extension", "leg_curl",
    "calf_raise", "box_jump", "burpee", "mountain_climber",
    "battle_rope", "kettlebell_swing", "clean_and_jerk",
]

# ---------------------------------------------------------------------------
# Alias map: many spelling variants → canonical name
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {
    # bench press variants
    "bench pressing":           "bench_press",
    "bench_pressing":           "bench_press",
    "benchpress":               "bench_press",
    "bench press":              "bench_press",
    # biceps / curl
    "bicep_curl":               "biceps",
    "barbell_biceps_curl":      "biceps",
    "barbell biceps curl":      "biceps",
    "bicep curl":               "biceps",
    "biceps curl":              "biceps",
    "dumbbell curl":            "biceps",
    "curl":                     "biceps",
    "BC":                       "biceps",     # MM-Fit code
    # shoulder press / overhead press
    "shoulder press":           "shoulder_press",
    "overhead press":           "shoulder_press",
    "OHP":                      "shoulder_press",  # MM-Fit
    "ohp":                      "shoulder_press",
    "military press":           "shoulder_press",
    "seated dumbbell press":    "shoulder_press",
    # triceps
    "tricep pushdown":          "triceps",
    "triceps pushdown":         "triceps",
    "tricep_pushdown":          "triceps",
    "cable pushdown":           "triceps",
    "overhead_tricep_extension":"triceps",
    "TE":                       "triceps",    # MM-Fit code
    # squat
    "squats":                   "squat",
    "back squat":               "squat",
    "barbell squat":            "squat",
    "goblet squat":             "squat",
    "SQ":                       "squat",      # MM-Fit
    # push-up
    "push_up":                  "push_up",
    "push up":                  "push_up",
    "pushup":                   "push_up",
    "push ups":                 "push_up",
    "PU":                       "push_up",    # MM-Fit
    # pull-up
    "pull_up":                  "pull_up",
    "pull up":                  "pull_up",
    "pullup":                   "pull_up",
    "pull ups":                 "pull_up",
    "chin up":                  "pull_up",
    # deadlift
    "deadlift":                 "deadlift",
    "conventional deadlift":    "deadlift",
    "DL":                       "deadlift",
    # lunge
    "lunges":                   "lunge",
    "lunge":                    "lunge",
    "LU":                       "lunge",      # MM-Fit
    # sit-up / crunch
    "sit up":                   "sit_up",
    "sit ups":                  "sit_up",
    "situp":                    "sit_up",
    "situps":                   "sit_up",
    "crunch":                   "sit_up",
    "SU":                       "sit_up",     # MM-Fit
    # jumping jacks
    "jumping jack":             "jumping_jack",
    "jumping jacks":            "jumping_jack",
    "JJ":                       "jumping_jack",   # MM-Fit
    # plank
    "plank":                    "plank",
    "PL":                       "plank",
    # lat pulldown
    "lat pulldown":             "lat_pulldown",
    "lat_pulldown":             "lat_pulldown",
    "LP":                       "lat_pulldown",
    # lateral raise
    "lateral raise":            "lateral_raise",
    "lateral raises":           "lateral_raise",
    "side raise":               "lateral_raise",
    "front raise":              "front_raise",
    # hammer curl
    "hammer curl":              "hammer_curl",
    "hammer_curl":              "hammer_curl",
    # hip thrust
    "hip thrust":               "hip_thrust",
    "hip thrusts":              "hip_thrust",
    # leg raises
    "leg raise":                "leg_raise",
    "leg raises":               "leg_raise",
    # row variants
    "barbell row":              "barbell_row",
    "bent over row":            "barbell_row",
    "t-bar row":                "barbell_row",
    "dumbbell row":             "dumbbell_row",
    # cable fly
    "cable fly":                "cable_fly",
    "chest fly":                "cable_fly",
    "pec fly":                  "cable_fly",
    # other
    "romanian deadlift":        "romanian_deadlift",
    "rdl":                      "romanian_deadlift",
}


def normalize_exercise(name: str) -> str:
    """Map any exercise name spelling to its canonical form.

    Returns the canonical name if found, or the original lowercased/underscored
    name if not found (so unknown exercises are preserved, not dropped).
    """
    if not name or (isinstance(name, float)):
        return ""
    s = str(name).strip()
    # Direct alias lookup (case-sensitive first for abbreviations like BC, SQ)
    if s in _ALIASES:
        return _ALIASES[s]
    # Lowercase version
    lower = s.lower().replace("-", "_").replace(" ", "_")
    if lower in _ALIASES:
        return _ALIASES[lower]
    # Check canonical list directly
    if lower in CANONICAL_EXERCISES:
        return lower
    # Space-separated alias
    spaced = s.lower()
    if spaced in _ALIASES:
        return _ALIASES[spaced]
    # Return normalised but unknown
    return lower


# ---------------------------------------------------------------------------
# Task support matrix
# ---------------------------------------------------------------------------

# For each dataset source, which tasks its rows can train.
# Derived programmatically in validate_unified.py but documented here.

DATASET_TASK_SUPPORT: dict[str, dict[str, str]] = {
    "captured": {
        "exercise_classification": "YES  (all videos have exercise label)",
        "rep_counting":            "YES  (rep_start/rep_end from segmentation)",
        "form_correction":         "YES  (score_form rule-based labels)",
    },
    "repcount": {
        "exercise_classification": "YES  (per-video exercise label)",
        "rep_counting":            "YES  (per-video count + cycle timestamps)",
        "form_correction":         "NO   (no form annotations)",
    },
    "countix_fitness": {
        "exercise_classification": "YES  (Kinetics class label)",
        "rep_counting":            "YES  (temporal repetition count)",
        "form_correction":         "NO   (no form annotations)",
    },
    "mmfit": {
        "exercise_classification": "YES  (per-session activity labels)",
        "rep_counting":            "PARTIAL (rep count per set, variable per-rep timing)",
        "form_correction":         "NO   (no form labels in public release)",
    },
    "realtime": {
        "exercise_classification": "YES  (video-level labels)",
        "rep_counting":            "NO   (no per-rep timing annotations)",
        "form_correction":         "NO   (no form labels)",
    },
    "roboflow": {
        "exercise_classification": "YES  (image-level class label)",
        "rep_counting":            "NO   (image-level only, no temporal info)",
        "form_correction":         "YES  (phase/form labels per frame)",
    },
}


def row_tasks(row: pd.Series) -> list[str]:
    """Return list of tasks a single unified row supports."""
    tasks = []
    # Exercise classification: always possible if exercise field is set
    if row.get("exercise", "") not in ("", None):
        tasks.append("exercise_classification")
    # Rep counting: only if both temporal boundaries are set
    try:
        rs = str(row.get("rep_start", "")).strip()
        re_ = str(row.get("rep_end", "")).strip()
        if rs != "" and re_ != "" and int(rs) < int(re_):
            tasks.append("rep_counting")
    except (ValueError, TypeError):
        pass
    # Form correction: only if form_label is 0 or 1 (not empty)
    try:
        fl = str(row.get("form_label", "")).strip()
        if fl in ("0", "1"):
            tasks.append("form_correction")
    except (ValueError, TypeError):
        pass
    return tasks


def empty_unified_df() -> pd.DataFrame:
    """Return an empty DataFrame with the unified schema."""
    return pd.DataFrame(columns=UNIFIED_COLUMNS)
