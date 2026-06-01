"""
Stop hook — runs after every agent session ends.
Updates the STATUS section in .cursor/cursor.md with current real counts.
Reads stdin (Cursor hook JSON), writes nothing to stdout (passive hook).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent  # test2/

def count_files(path, pattern="*.npy", exclude="_angles.npy"):
    p = ROOT / path
    if not p.exists():
        return 0
    return sum(
        1 for f in p.glob(pattern)
        if not f.name.endswith("_angles.npy") and not f.name.endswith("_raw.npy")
    )

def splits_exist():
    splits_dir = ROOT / "data" / "splits"
    return all((splits_dir / f"{s}.csv").exists() for s in ("train", "val", "test"))

def checkpoint_exists():
    return (ROOT / "checkpoints" / "best.pt").exists()

def annotation_rows():
    csv = ROOT / "data" / "annotations" / "reps.csv"
    if not csv.exists():
        return 0
    with open(csv) as f:
        lines = [l for l in f.readlines() if l.strip() and not l.startswith("video")]
    return len(lines)

def build_status_block():
    skeleton_count = count_files("data/processed/skeletons")
    splits_done = splits_exist()
    ckpt = checkpoint_exists()
    ann_rows = annotation_rows()

    rows = [
        ("Code complete",                    "DONE"),
        ("4 exercises configured",           "DONE"),
        ("Videos organized",                 "DONE — 125 videos in 4 folders"),
        (f"Pose extraction (skeletons)",     f"{'DONE' if splits_done else 'RUNNING/INCOMPLETE'} — {skeleton_count} .npy files"),
        ("Train/val/test splits",            "DONE" if splits_done else "NOT DONE — run prepare_data.py"),
        (f"Manual labeling (reps.csv)",      f"{'DONE — ' + str(ann_rows) + ' rep rows' if ann_rows > 0 else 'NOT STARTED — fill data/annotations/reps.csv'}"),
        ("Training (needs GPU)",             "DONE — best.pt exists" if ckpt else "NOT STARTED — run train.py on GPU machine"),
        ("checkpoints/best.pt",              "EXISTS" if ckpt else "DOES NOT EXIST YET"),
    ]

    lines = ["## Current Project Status\n", "\n", "| Task | Status |\n", "|---|---|\n"]
    for task, status in rows:
        lines.append(f"| {task} | {status} |\n")
    return lines

def update_cursor_md():
    md_path = ROOT / ".cursor" / "cursor.md"
    if not md_path.exists():
        return

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_block = "".join(build_status_block())

    # Replace everything between the status section header and the next ---
    import re
    pattern = r"(## Current Project Status\n)(.+?)(\n---)"
    replacement = new_block.rstrip() + r"\3"
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    if new_content != content:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(new_content)

try:
    _ = sys.stdin.read()  # consume hook JSON (required by Cursor hook protocol)
    update_cursor_md()
except Exception:
    pass  # fail open — never block the agent
