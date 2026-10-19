"""
Generate the README visual assets for imahrous13/exercise-detector (Taqwin).

Every geometric value that represents project behaviour is derived from real
repository data:

  * the skeleton poses in the hero come from
    data/processed/skeletons/biceps_biceps_003.npy  (real YOLOv8-Pose output)
  * the motion-signal trace comes from
    data/processed/skeletons/biceps_biceps_003_angles.npy
  * the annotation statistics come from data/annotations/unified_reps.csv
  * the bone topology comes from src/models/graph_conv.py (COCO_EDGES)
  * the limb colours come from src/utils/visualization.py (LIMB_COLORS)

Run:
    python assets/readme/source/generate_readme_assets.py

Output (written to assets/readme/):
    hero.svg, proof.svg, pipeline.svg
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
ROOT = ASSETS.parent.parent
SK_DIR = ROOT / "data" / "processed" / "skeletons"
ANN = ROOT / "data" / "annotations" / "unified_reps.csv"

# ---------------------------------------------------------------------------
# Frozen art direction
# ---------------------------------------------------------------------------
BG = "#06090D"
PANEL = "#0C1218"
RULE = "#1E2A36"
FG = "#EAF1F8"
FG_DIM = "#9FB0C2"
MUTED = "#68798B"
PRIMARY = "#5CC8FF"
ACCENT = "#FF7A6B"
OK = "#62D97E"
WARN = "#FFD166"
LEGS = "#C77DFF"

# Limb colours are toned-down versions of LIMB_COLORS in
# src/utils/visualization.py (which stores them as BGR tuples).
LIMB = {
    "head": "#5CC8FF",
    "arms": "#62D97E",
    "torso": "#FF7A6B",
    "legs": "#C77DFF",
}
JOINT = "#FFD166"

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# COCO 17-joint skeleton edges, verbatim from src/models/graph_conv.py
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Edge -> limb region, matching LIMB_COLORS order in src/utils/visualization.py
EDGE_LIMB = ["head"] * 4 + ["torso"] + ["arms"] * 4 + ["torso"] * 3 + ["legs"] * 4


# ---------------------------------------------------------------------------
# Tiny SVG helpers
# ---------------------------------------------------------------------------
def esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def txt(x, y, s, size=18, fill=FG, weight="400", family=SANS,
        anchor="start", ls=None, opacity=None):
    extra = ""
    if ls is not None:
        extra += f' letter-spacing="{ls}"'
    if opacity is not None:
        extra += f' opacity="{opacity}"'
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra}>{esc(s)}</text>'
    )


def box(x, y, w, h, rx=14, fill=PANEL, stroke=RULE, sw=1.2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def line(x1, y1, x2, y2, stroke=RULE, sw=1.2, dash=None, cap="round", opacity=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' opacity="{opacity}"' if opacity is not None else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="{cap}"{d}{o}/>'
    )


def polyline(points, stroke=PRIMARY, sw=2.0, fill="none", opacity=None, dash=None):
    pts = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    o = f' opacity="{opacity}"' if opacity is not None else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"{d}{o}/>'
    )


def head(x, y, fill=MUTED, size=7, direction="down"):
    """Solid triangular arrow head at (x, y)."""
    if direction == "down":
        pts = f"M {x:.1f} {y:.1f} L {x - size * 0.62:.1f} {y - size:.1f} L {x + size * 0.62:.1f} {y - size:.1f} Z"
    else:  # up
        pts = f"M {x:.1f} {y:.1f} L {x - size * 0.62:.1f} {y + size:.1f} L {x + size * 0.62:.1f} {y + size:.1f} Z"
    return f'<path d="{pts}" fill="{fill}"/>'


def arrow_h(x1, x2, y, stroke=MUTED, sw=1.6, size=7):
    return line(x1, y, x2 - size * 0.72, y, stroke=stroke, sw=sw, cap="butt") + \
        f'<path d="M {x2:.1f} {y:.1f} L {x2 - size:.1f} {y - size * 0.62:.1f} ' \
        f'L {x2 - size:.1f} {y + size * 0.62:.1f} Z" fill="{stroke}"/>'


def arrow_v(x, y1, y2, stroke=MUTED, sw=1.6, size=7, direction="down"):
    if direction == "down":
        return line(x, y1, x, y2 - size * 0.72, stroke=stroke, sw=sw, cap="butt") + \
            head(x, y2, stroke, size)
    return line(x, y1, x, y2 + size * 0.72, stroke=stroke, sw=sw, cap="butt") + \
        head(x, y2, stroke, size, "up")


def svg_open(width, height, title, desc, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="svgTitle svgDesc">\n'
        f'  <title id="svgTitle">{esc(title)}</title>\n'
        f'  <desc id="svgDesc">{esc(desc)}</desc>\n'
        + body
        + "\n</svg>\n"
    )


# ---------------------------------------------------------------------------
# Real project data
# ---------------------------------------------------------------------------
def load_pose_sequence(stem: str):
    skel = np.load(SK_DIR / f"{stem}.npy").astype(np.float64)
    angles = np.load(SK_DIR / f"{stem}_angles.npy").astype(np.float64)
    return skel, angles


def one_repetition_frames(angles, elbow_col=2, lo=10, hi=60):
    """Return 3 chronological frames: extended -> mid -> contracted of one rep."""
    elbow = angles[:, elbow_col]
    peak = lo + int(np.argmax(elbow[lo:hi]))
    trough = peak + int(np.argmin(elbow[peak:hi]))
    mid = (peak + trough) // 2
    return [peak, mid, trough], elbow


def draw_skeleton(points_2d, cx, cy, scale, bone_w=2.6, joint_r=3.4, opacity=1.0):
    """Draw one pose. points_2d is (17, 2) in normalised (hip-centred) space."""
    xs = cx + points_2d[:, 0] * scale
    ys = cy + points_2d[:, 1] * scale
    out = []
    for (a, b), limb in zip(COCO_EDGES, EDGE_LIMB):
        out.append(
            f'<line x1="{xs[a]:.1f}" y1="{ys[a]:.1f}" x2="{xs[b]:.1f}" y2="{ys[b]:.1f}" '
            f'stroke="{LIMB[limb]}" stroke-width="{bone_w}" stroke-linecap="round" '
            f'opacity="{opacity * 0.85:.2f}"/>'
        )
    for i in range(17):
        out.append(
            f'<circle cx="{xs[i]:.1f}" cy="{ys[i]:.1f}" r="{joint_r}" fill="{JOINT}" '
            f'opacity="{opacity:.2f}"/>'
        )
    return "".join(out), xs, ys


def centered(pose):
    """Centre a pose on its own bounding box."""
    return np.stack([pose[:, 0] - pose[:, 0].mean(), pose[:, 1] - pose[:, 1].mean()], axis=1)


# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
def build_hero():
    stem = "biceps_biceps_003"
    skel, angles = load_pose_sequence(stem)
    picks, elbow = one_repetition_frames(angles)

    W, H = 1200, 460
    b = [f'<rect width="{W}" height="{H}" rx="22" fill="{BG}"/>']

    # --- left: identity + value -------------------------------------------
    b.append(txt(56, 74, "SKELETON-BASED EXERCISE RECOGNITION", 17, PRIMARY,
                 "600", MONO, ls=2.4))
    b.append(txt(54, 146, "Taqwin", 68, FG, "700", SANS, ls="-1.5"))
    b.append(txt(56, 188, "ST-GCN Exercise Detector", 30, FG_DIM, "500", SANS))
    b.append(txt(56, 238,
                 "YOLOv8-Pose extracts a 17-joint skeleton from video. A spatial-",
                 20, MUTED))
    b.append(txt(56, 266,
                 "temporal graph CNN then classifies the exercise, scores the form,",
                 20, MUTED))
    b.append(txt(56, 294,
                 "and counts every repetition.",
                 20, MUTED))
    b.append(line(56, 330, 604, 330, RULE, 1.2))
    b.append(txt(56, 360,
                 "PYTORCH  ·  ULTRALYTICS YOLOv8-POSE  ·  OPENCV  ·  STREAMLIT",
                 16, MUTED, "500", MONO))
    b.append(txt(56, 388, "imahrous13/exercise-detector", 16, MUTED, "400", MONO))
    b.append(txt(56, 420, "train  ·  evaluate  ·  streamlit run app.py", 16, MUTED,
                 "400", MONO))

    # --- right: real skeleton output --------------------------------------
    px, py, pw, ph = 684, 44, 460, 372
    b.append(box(px, py, pw, ph, 18, PANEL, RULE, 1.2))
    b.append(txt(px + 24, py + 34, "REAL SKELETON OUTPUT", 16, MUTED, "500", MONO))
    b.append(txt(px + pw - 24, py + 34, "ONE REPETITION", 16, MUTED, "500", MONO,
                 anchor="end"))

    # One common transform for the three frames, so the motion is truthful.
    frames = [skel[fi, :, :2] for fi in picks]
    allx = np.concatenate([f[:, 0] for f in frames])
    ally = np.concatenate([f[:, 1] for f in frames])
    span_y = ally.max() - ally.min()
    span_x = allx.max() - allx.min()
    box_w, box_h = 260, 240
    scale = min(box_w / span_x, box_h / span_y)
    cx = px + pw / 2
    cy = py + 190

    def place(pose):
        return cx + pose[:, 0] * scale, cy + pose[:, 1] * scale

    # ghosted earlier phases, current phase on top
    for k, fi in enumerate(picks):
        xs, ys = place(frames[k])
        op = 0.45 if k < len(picks) - 1 else 1.0
        s, _, _ = draw_skeleton(frames[k], 0, 0, scale, bone_w=2.4, joint_r=3.2,
                                 opacity=op)
        s = s.replace(' x1="0.0"', "").replace(' x1="0.0"', "")
        # rebuild with the real translate
        parts = []
        for (a, b_), limb in zip(COCO_EDGES, EDGE_LIMB):
            parts.append(
                f'<line x1="{xs[a]:.1f}" y1="{ys[a]:.1f}" x2="{xs[b_]:.1f}" '
                f'y2="{ys[b_]:.1f}" stroke="{LIMB[limb]}" stroke-width="2.4" '
                f'stroke-linecap="round" opacity="{op * 0.85:.2f}"/>')
        for i in range(17):
            parts.append(f'<circle cx="{xs[i]:.1f}" cy="{ys[i]:.1f}" r="3.2" '
                         f'fill="{JOINT}" opacity="{op:.2f}"/>')
        b.append("".join(parts))

    # wrist trajectory of the working arm across the three phases
    traj = []
    for f in frames:
        xs, ys = place(f)
        traj.append((xs[9], ys[9]))
    b.append(polyline(traj, WARN, 2.0, opacity=0.85, dash="5 4"))
    for (tx, ty) in traj:
        b.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="4.5" fill="none" '
                 f'stroke="{WARN}" stroke-width="1.6"/>')

    b.append(txt(cx, py + 330, f"{stem}  ·  frame {picks[-1]}", 16, FG_DIM, "500",
                 MONO, anchor="middle"))
    b.append(txt(cx, py + 354, "17 JOINTS  ·  16 BONES  ·  30-FRAME WINDOW", 16,
                 MUTED, "500", MONO, anchor="middle"))

    return svg_open(
        W, H,
        "Taqwin - ST-GCN exercise detector",
        "Project identity: Taqwin, an ST-GCN exercise detector. YOLOv8-Pose extracts "
        "a 17-joint skeleton; a spatial-temporal graph CNN classifies the exercise, "
        "scores form quality and counts repetitions. The panel shows a real skeleton "
        "motion study - three frames of one biceps curl from the repository's own "
        "data - with the tracked wrist trajectory.",
        "\n  ".join("  " + s if s.strip() else s for s in b),
    )


# ---------------------------------------------------------------------------
# PROOF BOARD
# ---------------------------------------------------------------------------
def annotation_stats():
    with ANN.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    per_class, form, split = {}, {"1": 0, "0": 0}, {}
    videos = set()
    for r in rows:
        per_class[r["exercise"]] = per_class.get(r["exercise"], 0) + 1
        form[r["form_label"]] = form.get(r["form_label"], 0) + 1
        split[r["split"]] = split.get(r["split"], 0) + 1
        videos.add(r["video"])
    return len(rows), per_class, form, split, len(videos)


# canonical anatomical layout of the 17 COCO joints, plus label offsets
GRAPH_NODES = {
    0: (92, 100), 1: (74, 82), 2: (110, 82), 3: (58, 96), 4: (126, 96),
    5: (64, 136), 6: (120, 136), 7: (44, 180), 8: (140, 180),
    9: (32, 222), 10: (152, 222), 11: (78, 216), 12: (106, 216),
    13: (74, 266), 14: (110, 266), 15: (70, 312), 16: (114, 312),
}
GRAPH_LABELS = {
    0: (0, 22, "middle"), 1: (-10, -7, "end"), 2: (10, -7, "start"),
    3: (-10, 5, "end"), 4: (10, 5, "start"),
    5: (-13, 4, "end"), 6: (13, 4, "start"),
    7: (-13, 4, "end"), 8: (13, 4, "start"),
    9: (-13, 4, "end"), 10: (13, 4, "start"),
    11: (-13, 4, "end"), 12: (13, 4, "start"),
    13: (-13, 4, "end"), 14: (13, 4, "start"),
    15: (0, 22, "middle"), 16: (0, 22, "middle"),
}


def build_proof():
    stem = "biceps_biceps_003"
    skel, angles = load_pose_sequence(stem)
    elbow = angles[:, 2]
    n_rows, per_class, form, split, n_videos = annotation_stats()

    W, H = 1200, 700
    b = [f'<rect width="{W}" height="{H}" rx="22" fill="{BG}"/>']
    b.append(txt(56, 60, "PROOF — REAL OUTPUTS FROM THIS REPOSITORY", 17, PRIMARY,
                 "600", MONO, ls=2.4))

    # ---------------- panel A: the skeleton graph --------------------------
    ax, ay, aw, ah = 56, 86, 520, 350
    b.append(box(ax, ay, aw, ah, 18, PANEL, RULE, 1.2))
    b.append(txt(ax + 24, ay + 36, "1 · THE SKELETON GRAPH", 20, FG, "600", MONO))
    b.append(txt(ax + 24, ay + 60, "17 nodes · 16 edges · fixed adjacency", 16, MUTED,
                 "400", MONO))

    gscale = 0.82
    gx, gy = ax + 44, ay + 58
    for (i, j), limb in zip(COCO_EDGES, EDGE_LIMB):
        x1, y1 = GRAPH_NODES[i]
        x2, y2 = GRAPH_NODES[j]
        b.append(
            f'<line x1="{gx + x1 * gscale:.1f}" y1="{gy + y1 * gscale:.1f}" '
            f'x2="{gx + x2 * gscale:.1f}" y2="{gy + y2 * gscale:.1f}" '
            f'stroke="{LIMB[limb]}" stroke-width="2.4" stroke-linecap="round" opacity="0.8"/>'
        )
    for i, (px_, py_) in GRAPH_NODES.items():
        dx, dy, anchor = GRAPH_LABELS[i]
        b.append(f'<circle cx="{gx + px_ * gscale:.1f}" cy="{gy + py_ * gscale:.1f}" '
                 f'r="4" fill="{JOINT}"/>')
        b.append(txt(gx + px_ * gscale + dx, gy + py_ * gscale + dy, str(i), 16, MUTED,
                     "400", MONO, anchor=anchor))

    lx = ax + 240
    b.append(txt(lx, ay + 112, "A ∈ {0,1}   17 × 17", 18, FG_DIM, "500", MONO))
    b.append(txt(lx, ay + 136, "symmetric, self-loops", 16, MUTED, "400", MONO))
    b.append(txt(lx, ay + 172, "Â = D⁻¹ᐟ² (A + I) D⁻¹ᐟ²", 18, PRIMARY, "600", MONO))
    b.append(txt(lx, ay + 196, "fixed buffer · never learned", 16, MUTED, "400", MONO))
    b.append(line(lx, ay + 216, ax + aw - 24, ay + 216, RULE, 1.0))
    for k, (name, col) in enumerate(LIMB.items()):
        yy = ay + 244 + k * 26
        b.append(f'<circle cx="{lx + 5}" cy="{yy - 5}" r="4.5" fill="{col}"/>')
        b.append(txt(lx + 20, yy, name, 18, FG_DIM, "500", MONO))

    # ---------------- panel B: the motion signal --------------------------
    bx, by, bw, bh = 604, 86, 540, 350
    b.append(box(bx, by, bw, bh, 18, PANEL, RULE, 1.2))
    b.append(txt(bx + 24, by + 36, "2 · THE MOTION SIGNAL", 20, FG, "600", MONO))
    b.append(txt(bx + 24, by + 60, f"left-elbow angle (deg) vs frame · {stem}", 16,
                 MUTED, "400", MONO))

    px0, px1 = bx + 62, bx + bw - 30
    py0, py1 = by + 108, by + 250
    lo, hi = 0.0, 180.0
    n = len(elbow)

    def gx_(i):
        return px0 + (px1 - px0) * (i / (n - 1))

    def gy_(v):
        return py1 - (py1 - py0) * ((v - lo) / (hi - lo))

    for gv in (0, 45, 90, 135, 180):
        b.append(line(px0, gy_(gv), px1, gy_(gv), RULE, 1.0))
        b.append(txt(px0 - 10, gy_(gv) + 5, f"{gv}°", 16, MUTED, "400", MONO,
                     anchor="end"))
    b.append(line(px0, py0, px0, py1, RULE, 1.2))
    b.append(line(px0, py1, px1, py1, RULE, 1.2))

    win_s, win_e = 24, 54
    b.append(f'<rect x="{gx_(win_s):.1f}" y="{py0:.1f}" width="{gx_(win_e) - gx_(win_s):.1f}" '
             f'height="{py1 - py0:.1f}" fill="{PRIMARY}" opacity="0.13"/>')
    b.append(line(gx_(win_s), py0, gx_(win_s), py1, PRIMARY, 1.2, dash="4 3", opacity=0.8))
    b.append(line(gx_(win_e), py0, gx_(win_e), py1, PRIMARY, 1.2, dash="4 3", opacity=0.8))
    b.append(txt((gx_(win_s) + gx_(win_e)) / 2, py0 - 12, "30-frame window", 16, PRIMARY,
                 "500", MONO, anchor="middle"))

    b.append(polyline([(gx_(i), gy_(elbow[i])) for i in range(n)], PRIMARY, 2.0,
                       opacity=0.95))

    for i in (0, 100, 200, 300, 400):
        if i < n:
            b.append(line(gx_(i), py1, gx_(i), py1 + 6, RULE, 1.0))
            b.append(txt(gx_(i), py1 + 24, str(i), 16, MUTED, "400", MONO, anchor="middle"))

    b.append(txt(bx + 24, by + 300, "window 30  ·  stride 15", 18, FG_DIM, "500", MONO))
    b.append(txt(bx + 24, by + 324, "12 angles per frame feed the angle branch",
                 16, MUTED, "400", MONO))

    # ---------------- panel C: the annotations ----------------------------
    cx0, cy0, cw, ch = 56, 458, 1088, 194
    b.append(box(cx0, cy0, cw, ch, 18, PANEL, RULE, 1.2))
    b.append(txt(cx0 + 24, cy0 + 36, "3 · THE ANNOTATIONS", 20, FG, "600", MONO))
    b.append(txt(cx0 + 24, cy0 + 60,
                 f"{n_rows} labelled repetitions · {n_videos} of 125 videos · "
                 f"data/annotations/unified_reps.csv",
                 16, MUTED, "400", MONO))

    order = ["biceps", "triceps", "shoulder_press", "bench_press"]
    bx_bar = cx0 + 196
    bar_max = 320
    for k, name in enumerate(order):
        yy = cy0 + 92 + k * 26
        v = per_class.get(name, 0)
        b.append(txt(cx0 + 24, yy + 4, name, 18, FG_DIM, "500", MONO))
        b.append(f'<rect x="{bx_bar}" y="{yy - 10}" width="{bar_max}" height="14" rx="7" '
                 f'fill="{PRIMARY}" opacity="0.10"/>')
        w = bar_max * v / 300
        b.append(f'<rect x="{bx_bar}" y="{yy - 10}" width="{w:.1f}" height="14" rx="7" '
                 f'fill="{PRIMARY}" opacity="0.8"/>')
        b.append(txt(bx_bar + bar_max + 16, yy + 3, str(v), 18, FG, "500", MONO))

    sx0 = cx0 + 660
    tiles = [
        ("FORM CORRECT", str(form.get("1", 0)), OK),
        ("FORM INCORRECT", str(form.get("0", 0)), ACCENT),
        ("SPLIT TRAIN", str(split.get("train", 0)), FG_DIM),
        ("SPLIT VAL / TEST", f"{split.get('val', 0)} / {split.get('test', 0)}", FG_DIM),
    ]
    for k, (label, value, col) in enumerate(tiles):
        tx = sx0 + (k % 2) * 210
        ty = cy0 + 84 + (k // 2) * 56
        b.append(txt(tx, ty, label, 16, MUTED, "500", MONO))
        b.append(txt(tx, ty + 26, value, 26, col, "700", MONO))

    return svg_open(
        W, H,
        "Proof board: skeleton graph, motion signal and annotation statistics",
        "Three panels built from repository data. Left: the 17-node, 16-edge COCO "
        "skeleton graph with the symmetric normalisation used by the spatial graph "
        "convolution. Middle: the real left-elbow angle trace of biceps_biceps_003 "
        "with one 30-frame window highlighted. Right: counts of labelled "
        "repetitions per class, form labels and split sizes from "
        "data/annotations/unified_reps.csv.",
        "\n  ".join("  " + s if s.strip() else s for s in b),
    )


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------
def build_pipeline():
    W, H = 1200, 740
    b = [f'<rect width="{W}" height="{H}" rx="22" fill="{BG}"/>']
    b.append(txt(56, 46, "AS IMPLEMENTED — src/models/classifier.py + src/feedback/form_rules.py",
                 17, PRIMARY, "600", MONO, ls=2.4))

    def stage(x, y, w, h, title, sub=None, tcol=FG, scol=MUTED, ts=20, ss=16,
              fill=PANEL, stroke=RULE, sw=1.2, dash=None):
        out = [box(x, y, w, h, 14, fill, stroke, sw, dash)]
        cx = x + w / 2
        if sub:
            out.append(txt(cx, y + h / 2 - 2, title, ts, tcol, "600", MONO,
                           anchor="middle"))
            out.append(txt(cx, y + h / 2 + 22, sub, ss, scol, "400", MONO,
                           anchor="middle"))
        else:
            out.append(txt(cx, y + h / 2 + 6, title, ts, tcol, "600", MONO,
                           anchor="middle"))
        return "".join(out)

    # ---- row 0 : perception ---------------------------------------------
    b.append(stage(56, 72, 210, 72, "Video frame", "mp4 · webcam"))
    b.append(arrow_h(276, 300, 108))
    b.append(stage(300, 72, 580, 72, "YOLOv8-Pose · ROI crop + EMA · normalise",
                   "17 keypoints · bbox + 20 % · EMA α 0.35 · hip-centred"))
    b.append(box(920, 72, 224, 84, 14, "none", RULE, 1.2, dash="5 4"))
    b.append(txt(936, 102, "YOLO extracts joints.", 16, FG_DIM, "500", MONO))
    b.append(txt(936, 126, "It never sees a", 16, FG_DIM, "500", MONO))
    b.append(txt(936, 148, "class label.", 16, WARN, "600", MONO))

    # ---- fork into the two branches --------------------------------------
    b.append(arrow_v(590, 144, 176, PRIMARY, 1.6, 7))
    b.append(line(365, 176, 835, 176, PRIMARY, 1.6))
    b.append(arrow_v(365, 176, 196, PRIMARY, 1.6, 7))
    b.append(arrow_v(835, 176, 196, PRIMARY, 1.6, 7))

    b.append(stage(180, 196, 370, 72, "Skeleton  (T, 17, 6)",
                   "x · y · confidence · vx · vy · bone"))
    b.append(stage(650, 196, 370, 72, "12 joint angles  (T, 12)",
                   "degrees · 0–180 · see normalize.py"))
    b.append(arrow_v(365, 268, 292))
    b.append(arrow_v(835, 268, 292))
    b.append(stage(180, 292, 370, 72, "ST-GCN  ×4 blocks", "64 · 128 · 256 · 512",
                   tcol=PRIMARY))
    b.append(stage(650, 292, 370, 72, "Angle MLP  12 → 64 → 64", "2 linear layers",
                   tcol=PRIMARY))

    # ---- merge into the fusion row ---------------------------------------
    b.append(arrow_v(365, 364, 388, PRIMARY, 1.6, 7))
    b.append(arrow_v(835, 364, 388, PRIMARY, 1.6, 7))
    b.append(line(365, 388, 835, 388, PRIMARY, 1.6))
    b.append(arrow_v(375, 388, 408, PRIMARY, 1.6, 7))

    b.append(stage(280, 408, 190, 72, "Concat", "B × 30 × 576"))
    b.append(arrow_h(470, 500, 444))
    b.append(stage(500, 408, 190, 72, "BiLSTM", "2 × 256 · bidir"))
    b.append(arrow_h(690, 720, 444))
    b.append(stage(720, 408, 190, 72, "Mean pool", "B × 512"))

    # ---- the three heads (only the exercise head reaches the UI) ---------
    b.append(arrow_v(815, 480, 496, PRIMARY, 1.6, 7))
    b.append(line(275, 496, 715, 496, PRIMARY, 1.6))
    heads = [("Exercise head", "B × 4", PRIMARY, False),
             ("Form head", "B × 2", MUTED, True),
             ("Rep head", "B × 2", MUTED, True)]
    for k, (t, s, col, dim) in enumerate(heads):
        hx = 180 + k * 220
        b.append(arrow_v(hx + 95, 496, 516, PRIMARY, 1.6, 7))
        b.append(stage(hx, 516, 190, 72, t, s, tcol=col, scol=MUTED,
                       fill=PANEL if not dim else "none",
                       stroke=RULE if not dim else RULE,
                       dash="5 4" if dim else None))

    # ---- rule engine lane (runs at inference) ----------------------------
    b.append(box(180, 630, 400, 86, 14, "none", RULE, 1.2, dash="5 4"))
    b.append(txt(200, 662, "form_head · rep_head", 18, MUTED, "600", MONO))
    b.append(txt(200, 688, "trained, but not used at inference", 16, MUTED, "400", MONO))

    b.append(box(610, 630, 534, 86, 14, "none", WARN, 1.2, dash="5 4"))
    b.append(txt(630, 662, "RULE ENGINE · src/feedback/form_rules.py", 18, WARN,
                 "600", MONO))
    b.append(txt(630, 688, "→ form score 0–100 + fault messages   → rep count",
                 16, FG_DIM, "400", MONO))

    # dashed feed: 12 joint angles -> rule engine
    b.append(f'<path d="M 1020 232 L 1090 232 L 1090 622" fill="none" '
             f'stroke="{WARN}" stroke-width="1.6" stroke-dasharray="5 4" '
             f'stroke-linejoin="round"/>')
    b.append(head(1090, 630, WARN, 7))

    return svg_open(
        W, H,
        "Taqwin pipeline: pose to exercise class, rule-based form and rep output",
        "Video frames become 17-keypoint poses via YOLOv8-Pose with ROI cropping and "
        "EMA smoothing, then hip-centred and torso-scaled. The skeleton tensor and "
        "the 12 joint angles run through two parallel branches - four ST-GCN blocks "
        "and a small angle MLP - are concatenated into a 576-wide sequence and modelled "
        "by a bidirectional LSTM. The model has three heads; only the exercise head "
        "reaches the user. Form score, fault messages and repetition counting come from "
        "the deterministic rule engine, which runs at inference as well as at label "
        "generation time.",
        "\n  ".join("  " + s if s.strip() else s for s in b),
    )


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, builder in (("hero", build_hero), ("proof", build_proof),
                          ("pipeline", build_pipeline)):
        path = ASSETS / f"{name}.svg"
        path.write_text(builder(), encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
