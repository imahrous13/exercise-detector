"""Run inference on a video file - detect exercise and assess form.

Usage:
    python scripts/inference.py --video path/to/video.mp4 --checkpoint checkpoints/best.pt
    python scripts/inference.py --video path/to/video.mp4 --checkpoint checkpoints/best.pt --output output.mp4
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pose_extraction.extractor import PoseExtractor
from src.preprocessing.normalize import preprocess_skeleton, compute_angles
from src.data.dataset import compute_extra_features
from src.models.classifier import build_model
from src.feedback.form_rules import get_exercise_names, EXERCISE_NAMES
from src.utils.visualization import draw_skeleton, draw_feedback, create_output_video


def run_inference(video_path, model, extractor, device, window_size=30, stride=15, exercise_names=None, use_rep_head=True):
    """Run full inference pipeline on a video.

    Args:
        video_path: path to input video
        model: trained ExerciseSTGCN model
        extractor: PoseExtractor instance
        device: torch device
        window_size: frames per window
        stride: stride between windows
        exercise_names: optional list of class names (e.g. from config); if None, uses built-in
        use_rep_head: if True and model has trained rep head, count reps from rep head; else 0

    Returns:
        results: list of per-window result dicts
        window_assignments: per-frame result dicts (with rep counts)
        raw_keypoints: (T, 17, 3) in crop/ROI space (for model)
        frames: list of BGR frames
        fps: video frame rate
        crop_offsets: (T, 2) offset per frame to convert keypoints to full-frame for drawing
    """
    names = exercise_names if exercise_names is not None else EXERCISE_NAMES
    # Extract keypoints with frames (keypoints in crop/ROI space for model consistency)
    raw_keypoints, frames, fps, crop_offsets = extractor.extract_from_video_with_frames(video_path)
    T = raw_keypoints.shape[0]
    print(f"Extracted {T} frames from video ({fps:.1f} fps)")

    # Preprocess
    normalized_kpts, angles = preprocess_skeleton(raw_keypoints)

    # Compute extra features (velocity + bone lengths): (T, 17, 3) -> (T, 17, 6)
    skeleton_features = compute_extra_features(normalized_kpts)

    # Generate windows and run model
    model.eval()
    results = []
    window_assignments = [None] * T  # Maps each frame to its result
    correct_reps, incorrect_reps = 0, 0
    last_counted_frame = -100
    rep_cooldown_frames = stride  # avoid double-counting overlapping windows

    with torch.no_grad():
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            window_feat = skeleton_features[start:end]   # (30, 17, 6)
            window_angles = angles[start:end]             # (30, 12)

            # Model prediction (exercise, form, rep heads)
            x = torch.FloatTensor(window_feat).unsqueeze(0).to(device)      # (1, 30, 17, 6)
            a = torch.FloatTensor(window_angles).unsqueeze(0).to(device)     # (1, 30, 12)
            exercise_logits, form_logits, rep_logits = model(x, a)

            exercise_pred = exercise_logits.argmax(1).item()
            form_pred = form_logits.argmax(1).item()
            exercise_conf = torch.softmax(exercise_logits, 1).max().item()
            form_probs = torch.softmax(form_logits, 1).cpu().numpy()[0]
            form_score = int(round(100 * form_probs[1]))  # prob of correct class

            # Learned rep head: count rep when prob(rep) > 0.5 and cooldown passed; correct/incorrect from form
            if use_rep_head:
                rep_probs = torch.softmax(rep_logits, 1).cpu().numpy()[0]
                rep_prob = float(rep_probs[1])
                end_frame = end - 1
                if rep_prob > 0.5 and end_frame >= last_counted_frame + rep_cooldown_frames:
                    if form_pred == 1:
                        correct_reps += 1
                    else:
                        incorrect_reps += 1
                    last_counted_frame = end_frame

            # Form from model only (rules used only for labeling training data)
            result = {
                'exercise': names[exercise_pred] if exercise_pred < len(names) else 'unknown',
                'exercise_idx': exercise_pred,
                'exercise_confidence': exercise_conf,
                'is_correct': form_pred == 1,
                'form_confidence': float(form_probs[form_pred]),
                'form_score': form_score,
                'feedback': [],
                'start_frame': start,
                'end_frame': end,
                'correct_reps': correct_reps,
                'incorrect_reps': incorrect_reps,
            }
            results.append(result)

            # Assign result to frames in this window
            for f in range(start, end):
                window_assignments[f] = result

    # Handle remaining frames at the end
    if T >= window_size:
        last_result = results[-1] if results else None
        for f in range(len(window_assignments)):
            if window_assignments[f] is None:
                window_assignments[f] = last_result

    # Stamp per-frame result (copy rep counts from base so each frame has correct_reps/incorrect_reps)
    for f in range(T):
        base = window_assignments[f]
        if base is not None:
            frame_result = dict(base)
        else:
            frame_result = {'exercise': 'unknown', 'is_correct': True,
                            'feedback': [], 'correct_reps': 0, 'incorrect_reps': 0}
        frame_result['correct_reps'] = base.get('correct_reps', 0) if base else 0
        frame_result['incorrect_reps'] = base.get('incorrect_reps', 0) if base else 0
        window_assignments[f] = frame_result

    return results, window_assignments, raw_keypoints, frames, fps, crop_offsets


def main():
    parser = argparse.ArgumentParser(description='Exercise detection inference')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--output', type=str, default=None, help='Output video path (optional)')
    parser.add_argument('--model_size', type=str, default=None,
                        help='YOLOv8 pose size (n/s/m/l/x). Default from config.')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint['model_state_dict']
    use_rep_head = any(k.startswith('rep_head') for k in state)
    model.load_state_dict(state, strict=use_rep_head)  # strict=False allows old checkpoints without rep_head
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {args.checkpoint}" + (" (with rep head)" if use_rep_head else " (no rep head, reps=0)"))

    # Create pose extractor (use config defaults for stability)
    pose_cfg = config.get("pose", {})
    model_size = args.model_size or pose_cfg.get("model_size", "s")
    smoothing = pose_cfg.get("smoothing_alpha", 0.35)
    extractor = PoseExtractor(
        model_size=model_size,
        confidence_threshold=pose_cfg.get("confidence_threshold", 0.5),
        smoothing_alpha=smoothing,
        roi_padding_ratio=pose_cfg.get("roi_padding_ratio", 0.2),
    )

    # Run inference (exercise names from config so pipeline works with any data)
    results, window_assignments, raw_keypoints, frames, fps, crop_offsets = run_inference(
        video_path=args.video,
        model=model,
        extractor=extractor,
        device=device,
        window_size=config['data']['window_size'],
        stride=config['data']['stride'],
        exercise_names=get_exercise_names(config),
        use_rep_head=use_rep_head,
    )

    # Print results summary
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)

    if not results:
        print("No windows could be processed (video too short?)")
        return

    # Aggregate predictions across windows (majority vote)
    from collections import Counter
    exercise_votes = Counter(r['exercise'] for r in results)
    most_common_exercise = exercise_votes.most_common(1)[0][0]
    correct_count = sum(1 for r in results if r['is_correct'])
    total_windows = len(results)

    # Average form score
    form_scores = [r.get('form_score', 100) for r in results]
    avg_form_score = sum(form_scores) / len(form_scores) if form_scores else 100

    print(f"Detected exercise: {most_common_exercise.replace('_', ' ').title()}")
    print(f"Average form score: {avg_form_score:.0f}/100")
    print(f"Form correctness: {correct_count}/{total_windows} windows correct "
          f"({100 * correct_count / total_windows:.0f}%)")

    # Rep count summary (from last result which carries final counts)
    last = results[-1]
    c_reps = last.get('correct_reps', 0)
    i_reps = last.get('incorrect_reps', 0)
    print(f"Reps detected: {c_reps + i_reps} total  "
          f"(Correct: {c_reps}, Incorrect: {i_reps})")

    # Collect unique feedback across all windows
    all_feedback = set()
    for r in results:
        all_feedback.update(r['feedback'])

    if all_feedback:
        print("\nForm Feedback:")
        for msg in sorted(all_feedback):
            print(f"  - {msg}")
    else:
        print("\nForm looks good! No corrections needed.")

    # Per-window details
    print(f"\nPer-window breakdown ({total_windows} windows):")
    for i, r in enumerate(results):
        status = "OK" if r['is_correct'] else "FIX"
        score = r.get('form_score', '?')
        print(f"  Window {i+1} [frames {r['start_frame']}-{r['end_frame']}]: "
              f"{r['exercise']} ({r['exercise_confidence']:.2f}) [{status}] score={score}")

    # Save output video if requested (convert crop-space keypoints to full-frame for drawing)
    if args.output:
        print(f"\nSaving output video to: {args.output}")
        keypoints_for_drawing = []
        for i in range(len(frames)):
            k = raw_keypoints[i].copy()
            k[..., 0] += crop_offsets[i, 0]
            k[..., 1] += crop_offsets[i, 1]
            keypoints_for_drawing.append(k)
        create_output_video(
            frames=frames,
            keypoints_list=keypoints_for_drawing,
            results_list=window_assignments,
            output_path=args.output,
            fps=fps,
        )
        print("Output video saved.")


if __name__ == '__main__':
    main()
