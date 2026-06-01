"""Manual rep annotation loading and overlap-based window labeling.

Ground-truth rep boundaries and form labels come from human-annotated CSV/JSON.
Rule-based systems (segment_reps, score_form) are not used for training labels.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from src.feedback.form_rules import get_exercise_names, EXERCISE_NAMES


@dataclass
class RepAnnotation:
    """One human-annotated repetition in a video."""

    rep_start: int
    rep_end: int
    form_label: int  # 1 = correct, 0 = incorrect
    mistake_type: str = "none"
    exercise: Optional[str] = None


@dataclass
class VideoAnnotations:
    """All rep annotations for a single video."""

    video_key: str
    exercise: Optional[str] = None
    reps: List[RepAnnotation] = field(default_factory=list)


def _parse_form_label(value) -> int:
    """Parse form_label from CSV/JSON (correct/incorrect or 0/1)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0
    s = str(value).strip().lower()
    if s in ("1", "correct", "true", "ok", "good"):
        return 1
    if s in ("0", "incorrect", "false", "bad", "wrong"):
        return 0
    raise ValueError(f"Unrecognized form_label: {value!r}")


def _normalize_mistake(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "none"
    s = str(value).strip().lower()
    return s if s else "none"


def _frame_overlap(win_start: int, win_end: int, rep_start: int, rep_end: int) -> int:
    """Number of frames in [win_start, win_end) overlapping [rep_start, rep_end] inclusive."""
    a0, a1 = win_start, win_end - 1
    b0, b1 = rep_start, rep_end
    overlap_start = max(a0, b0)
    overlap_end = min(a1, b1)
    return max(0, overlap_end - overlap_start + 1)


def label_window_from_reps(
    win_start: int,
    win_end: int,
    reps: List[RepAnnotation],
) -> Tuple[int, int, bool, Optional[str]]:
    """Assign rep/form labels to a sliding window from manual rep segments.

    Args:
        win_start: window start frame (inclusive)
        win_end: window end frame (exclusive), e.g. start+30
        reps: list of RepAnnotation for this video

    Returns:
        rep_label: 1 if window overlaps any annotated rep, else 0
        form_label: 1 correct / 0 incorrect from best-overlap rep (only meaningful if rep_label=1)
        form_valid: False when no rep overlaps (form head should mask this sample)
        mistake_type: from best-overlap rep, or None if no overlap
    """
    if not reps:
        return 0, 0, False, None

    best_overlap = 0
    best_rep: Optional[RepAnnotation] = None

    for rep in reps:
        n = _frame_overlap(win_start, win_end, rep.rep_start, rep.rep_end)
        if n > best_overlap:
            best_overlap = n
            best_rep = rep

    if best_rep is None or best_overlap <= 0:
        return 0, 0, False, None

    return 1, best_rep.form_label, True, best_rep.mistake_type


def resolve_skeleton_filename(
    video_ref: str,
    exercise: Optional[str],
    known_filenames: Optional[List[str]] = None,
) -> str:
    """Map annotation ``video`` field to skeleton ``filename`` (no .npy).

    Supports:
      - exact match to unique_name (e.g. squat_01 or squat_squat_01)
      - bare video stem + exercise prefix (squat_01.mp4 + squat -> squat_squat_01)
      - basename match against known_filenames
    """
    ref = str(video_ref).strip()
    stem = Path(ref).stem
    candidates = []

    if not ref.endswith(".npy"):
        candidates.append(stem)
        if exercise:
            ex = exercise.strip().replace(" ", "_")
            candidates.append(f"{ex}_{stem}")
    candidates.append(ref.replace(".mp4", "").replace(".avi", "").replace(".mov", ""))

    if known_filenames:
        for c in candidates:
            if c in known_filenames:
                return c
        for fn in known_filenames:
            for c in candidates:
                if fn == c or fn.endswith(f"_{c}") or fn.endswith(c):
                    return fn

    if exercise:
        ex = exercise.strip().replace(" ", "_")
        return f"{ex}_{stem}"
    return stem


def _exercise_name_to_index(name: str, exercise_names: List[str]) -> Optional[int]:
    if not name:
        return None
    key = name.strip().replace(" ", "_").lower()
    for i, ex in enumerate(exercise_names):
        if ex.lower() == key or ex.replace("_", " ").lower() == key.replace("_", " "):
            return i
    return None


def load_rep_annotations_csv(
    path: str,
    exercise_names: Optional[List[str]] = None,
) -> Dict[str, VideoAnnotations]:
    """Load rep annotations from CSV.

    Required columns: video, rep_start, rep_end, form_label
    Optional: exercise, mistake_type
    """
    names = exercise_names or EXERCISE_NAMES
    df = pd.read_csv(path)
    if df.empty:
        return {}

    required = {"video", "rep_start", "rep_end", "form_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Annotation CSV missing columns: {sorted(missing)}")

    index: Dict[str, VideoAnnotations] = {}

    for _, row in df.iterrows():
        if pd.isna(row.get("rep_start")) or str(row.get("rep_start", "")).strip() == "":
            continue
        if pd.isna(row.get("rep_end")) or str(row.get("rep_end", "")).strip() == "":
            continue

        video_ref = row["video"]
        ex_name = row.get("exercise")
        if pd.notna(ex_name):
            ex_name = str(ex_name).strip()
        else:
            ex_name = None

        video_key = resolve_skeleton_filename(str(video_ref), ex_name)
        rep = RepAnnotation(
            rep_start=int(row["rep_start"]),
            rep_end=int(row["rep_end"]),
            form_label=_parse_form_label(row["form_label"]),
            mistake_type=_normalize_mistake(row.get("mistake_type")),
            exercise=ex_name,
        )
        if rep.rep_end < rep.rep_start:
            raise ValueError(
                f"rep_end < rep_start for {video_ref}: {rep.rep_start}-{rep.rep_end}"
            )

        if video_key not in index:
            ex_idx = _exercise_name_to_index(ex_name, names) if ex_name else None
            index[video_key] = VideoAnnotations(
                video_key=video_key,
                exercise=names[ex_idx] if ex_idx is not None else ex_name,
                reps=[],
            )
        index[video_key].reps.append(rep)

    for va in index.values():
        va.reps.sort(key=lambda r: r.rep_start)

    return index


def load_rep_annotations_json(
    path: str,
    exercise_names: Optional[List[str]] = None,
) -> Dict[str, VideoAnnotations]:
    """Load rep annotations from JSON.

    Format:
    {
      "annotations": [
        {
          "video": "squat_01.mp4",
          "exercise": "squat",
          "reps": [
            {"rep_start": 45, "rep_end": 92, "form_label": "correct", "mistake_type": "none"}
          ]
        }
      ]
    }
    """
    names = exercise_names or EXERCISE_NAMES
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("annotations", data if isinstance(data, list) else [])
    index: Dict[str, VideoAnnotations] = {}

    for entry in entries:
        video_ref = entry["video"]
        ex_name = entry.get("exercise")
        video_key = resolve_skeleton_filename(str(video_ref), ex_name)

        if video_key not in index:
            index[video_key] = VideoAnnotations(
                video_key=video_key,
                exercise=ex_name,
                reps=[],
            )

        for r in entry.get("reps", []):
            rep = RepAnnotation(
                rep_start=int(r["rep_start"]),
                rep_end=int(r["rep_end"]),
                form_label=_parse_form_label(r["form_label"]),
                mistake_type=_normalize_mistake(r.get("mistake_type")),
                exercise=ex_name,
            )
            if rep.rep_end < rep.rep_start:
                raise ValueError(
                    f"rep_end < rep_start for {video_ref}: {rep.rep_start}-{rep.rep_end}"
                )
            index[video_key].reps.append(rep)

    for va in index.values():
        va.reps.sort(key=lambda r: r.rep_start)

    return index


def load_rep_annotations(
    path: str,
    exercise_names: Optional[List[str]] = None,
    *,
    allow_missing: bool = False,
) -> Dict[str, VideoAnnotations]:
    """Load annotations from CSV or JSON by file extension."""
    path = str(path)
    if not os.path.isfile(path):
        if allow_missing:
            return {}
        raise FileNotFoundError(f"Annotations file not found: {path}")

    ext = Path(path).suffix.lower()
    if ext == ".json":
        return load_rep_annotations_json(path, exercise_names)
    if ext in (".csv", ".tsv"):
        return load_rep_annotations_csv(path, exercise_names)
    raise ValueError(f"Unsupported annotation format: {ext} (use .csv or .json)")


def get_annotations_path(config: dict, project_root: Optional[str] = None) -> Optional[str]:
    """Resolve annotations file path from config.data.labeling."""
    data_cfg = config.get("data", {}) if config else {}
    labeling = data_cfg.get("labeling", {})
    source = labeling.get("source", "hybrid")
    if source not in ("manual", "hybrid", "rules"):
        return None
    path = labeling.get("annotations_file")
    if not path:
        return None

    candidates = [path]
    if project_root:
        candidates.append(os.path.join(project_root, path))
    if not os.path.isabs(path):
        candidates.append(os.path.join(os.getcwd(), path))

    for p in candidates:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return None
