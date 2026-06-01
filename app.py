"""Streamlit app for Exercise Detection & Form Feedback.

Two modes:
  1. Upload Video  - upload a video file, run full inference, view results
  2. Live Webcam   - real-time exercise detection via webcam feed

Usage:
    streamlit run app.py
"""

import os
import sys
import tempfile
from collections import deque

import cv2
import numpy as np
import streamlit as st
import torch
import yaml

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from src.models.classifier import build_model
from src.pose_extraction.extractor import PoseExtractor
from src.preprocessing.normalize import preprocess_skeleton, compute_angles
from src.data.dataset import compute_extra_features
from src.feedback.form_rules import get_exercise_names, EXERCISE_NAMES
from src.utils.visualization import draw_skeleton, draw_feedback, draw_person_boxes, create_output_video

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Exercise Detector",
    page_icon="\U0001F3CB",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resource loaders (run once, shared across reruns)
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best.pt")


@st.cache_resource
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state = checkpoint["model_state_dict"]
    use_rep_head = any(k.startswith("rep_head") for k in state)
    model.load_state_dict(state, strict=use_rep_head)
    model = model.to(device)
    model.eval()
    return model, device, use_rep_head


@st.cache_resource
def load_extractor():
    config = load_config()
    pose = config.get("pose", {})
    return PoseExtractor(
        model_size=pose.get("model_size", "s"),
        confidence_threshold=pose.get("confidence_threshold", 0.5),
        smoothing_alpha=pose.get("smoothing_alpha", 0.35),
        roi_padding_ratio=pose.get("roi_padding_ratio", 0.2),
    )


# ---------------------------------------------------------------------------
# Helper: run inference on an uploaded video (reuses scripts/inference logic)
# ---------------------------------------------------------------------------
def run_video_inference(video_path, model, extractor, device, config, use_rep_head=True):
    """Run full inference pipeline on a video file.

    Returns results, window_assignments, raw_keypoints, frames, fps, crop_offsets.
    Keypoints are in crop/ROI space; use crop_offsets to convert to full-frame for drawing.
    """
    from scripts.inference import run_inference

    return run_inference(
        video_path=video_path,
        model=model,
        extractor=extractor,
        device=device,
        window_size=config["data"]["window_size"],
        stride=config["data"]["stride"],
        exercise_names=get_exercise_names(config),
        use_rep_head=use_rep_head,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("\U0001F3CB Exercise Detector")
mode = st.sidebar.radio("Mode", ["Upload Video", "Live Webcam"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Model info**")
st.sidebar.text(f"Checkpoint: best.pt")
device_name = "CUDA" if torch.cuda.is_available() else "CPU"
st.sidebar.text(f"Device: {device_name}")
st.sidebar.text(f"Exercises: {len(get_exercise_names(load_config()))}")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with **ST-GCN + BiLSTM** for exercise classification "
    "and angle-based form feedback."
)

# ---------------------------------------------------------------------------
# Mode 1: Upload Video
# ---------------------------------------------------------------------------
if mode == "Upload Video":
    st.title("\U0001F4F9 Upload Video for Analysis")
    st.write("Upload an exercise video to detect the exercise type and get form feedback.")

    uploaded = st.file_uploader(
        "Choose a video file", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded is not None:
        # Save uploaded file to a temp path
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        # Show the original video
        st.video(tmp_path)

        if st.button("Analyze Video", type="primary"):
            model, device, use_rep_head = load_model()
            extractor = load_extractor()
            extractor.reset_tracking()  # fresh tracking for each video
            config = load_config()

            with st.spinner("Extracting poses and running model inference..."):
                try:
                    results, window_assignments, raw_keypoints, frames, fps, crop_offsets = (
                        run_video_inference(tmp_path, model, extractor, device, config, use_rep_head)
                    )
                except ValueError as e:
                    st.error(f"Error processing video: {e}")
                    st.stop()

            if not results:
                st.warning("Video too short to process (need at least 30 frames with a visible person).")
                st.stop()

            # ---- Aggregate results ----
            from collections import Counter

            exercise_votes = Counter(r["exercise"] for r in results)
            most_common_exercise = exercise_votes.most_common(1)[0][0]
            correct_count = sum(1 for r in results if r["is_correct"])
            total_windows = len(results)
            form_pct = 100 * correct_count / total_windows

            # Average form score from scoring system
            form_scores = [r.get("form_score", 100) for r in results]
            avg_form_score = sum(form_scores) / len(form_scores) if form_scores else 100

            # ---- Rep counts (from last window_assignment which has final counts) ----
            final_frame_result = window_assignments[-1] if window_assignments else {}
            c_reps = final_frame_result.get("correct_reps", 0)
            i_reps = final_frame_result.get("incorrect_reps", 0)

            # ---- Display metrics ----
            st.markdown("---")
            st.subheader("Results")

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Exercise Detected", most_common_exercise.replace("_", " ").title())
            col2.metric("Form Score", f"{avg_form_score:.0f}/100")
            col3.metric("Windows Correct", f"{form_pct:.0f}%")
            col4.metric("Total Reps", c_reps + i_reps)
            col5.metric("Correct Reps", c_reps)
            col6.metric("Incorrect Reps", i_reps)

            # ---- Form feedback ----
            all_feedback = set()
            for r in results:
                all_feedback.update(r["feedback"])

            if all_feedback:
                st.subheader("Form Feedback")
                for msg in sorted(all_feedback):
                    st.warning(f"\u26A0\uFE0F {msg}")
            else:
                st.success("\u2705 Form looks good! No corrections needed.")

            # ---- Per-window breakdown ----
            with st.expander(f"Per-window breakdown ({total_windows} windows)"):
                for i, r in enumerate(results):
                    status = "\u2705" if r["is_correct"] else "\u274C"
                    st.text(
                        f"Window {i+1} [frames {r['start_frame']}-{r['end_frame']}]: "
                        f"{r['exercise']} ({r['exercise_confidence']:.2f}) {status}"
                    )

            # ---- Generate annotated output video (convert crop keypoints to full-frame for drawing) ----
            st.subheader("Annotated Video")
            with st.spinner("Generating annotated video..."):
                out_path = tempfile.mktemp(suffix=".mp4")
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
                    output_path=out_path,
                    fps=fps,
                )

            st.video(out_path)

            # Download button
            with open(out_path, "rb") as f:
                st.download_button(
                    label="Download Annotated Video",
                    data=f.read(),
                    file_name=f"analyzed_{uploaded.name}",
                    mime="video/mp4",
                )

            # Clean up temp files
            try:
                os.unlink(tmp_path)
                os.unlink(out_path)
            except OSError:
                pass

# ---------------------------------------------------------------------------
# Mode 2: Live Webcam
# ---------------------------------------------------------------------------
elif mode == "Live Webcam":
    st.title("\U0001F4F7 Live Webcam Exercise Detection")
    st.write("Start your webcam for real-time exercise detection and form feedback.")
    st.info("The system auto-tracks the **largest person**. Use the person selector "
            "to switch who is being tracked.")

    # Session state for webcam control
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False
    if "num_people" not in st.session_state:
        st.session_state.num_people = 0
    if "selected_person" not in st.session_state:
        st.session_state.selected_person = 0  # 0 = auto (largest)

    def start_webcam():
        st.session_state.webcam_running = True
        st.session_state.selected_person = 0

    def stop_webcam():
        st.session_state.webcam_running = False

    col_start, col_stop = st.columns(2)
    col_start.button("Start Webcam", on_click=start_webcam, type="primary",
                     disabled=st.session_state.webcam_running)
    col_stop.button("Stop Webcam", on_click=stop_webcam,
                    disabled=not st.session_state.webcam_running)

    # Person selector (0 = auto/largest, 1+ = specific person)
    person_choice = st.sidebar.number_input(
        "Track person #", min_value=0,
        max_value=max(st.session_state.num_people, 1),
        value=st.session_state.selected_person,
        help="0 = auto (largest person). Set 1, 2, ... to pick a specific person.",
    )
    st.session_state.selected_person = person_choice

    # Placeholders for live feed and sidebar metrics
    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()
    feedback_placeholder = st.empty()

    if st.session_state.webcam_running:
        model, device, use_rep_head = load_model()
        extractor = load_extractor()
        extractor.reset_tracking()
        config = load_config()
        config_ex_names = get_exercise_names(config)
        window_size = config["data"]["window_size"]
        stride = config["data"]["stride"]
        # Rep counting from learned rep head (cooldown to avoid double-count)
        webcam_correct_reps, webcam_incorrect_reps = 0, 0
        webcam_last_counted_frame = -100
        webcam_frame_counter = 0
        rep_cooldown = stride

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam. Please check your camera connection.")
            st.session_state.webcam_running = False
            st.stop()

        # Rolling buffer for keypoints
        kpt_buffer = deque(maxlen=window_size)
        frame_count = 0
        current_result = None

        try:
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Webcam frame read failed.")
                    break

                # Get all detections for bounding box display
                all_kpts, all_bboxes, tracked_idx = extractor.extract_all_detections(frame)
                st.session_state.num_people = len(all_kpts)

                # If user selected a specific person, override tracking; else extract from ROI (crop-space keypoints)
                sel = st.session_state.selected_person
                if sel > 0 and sel <= len(all_kpts):
                    idx = sel - 1  # 1-indexed to 0-indexed
                    kpts, roi_ox, roi_oy = all_kpts[idx], 0, 0
                    extractor._tracked_bbox = all_bboxes[idx]
                    tracked_idx = idx
                else:
                    kpts, roi_ox, roi_oy = extractor.extract_from_frame(frame)
                    if kpts is not None and extractor._tracked_bbox is not None:
                        for i, bbox in enumerate(all_bboxes):
                            from src.pose_extraction.extractor import _bbox_iou
                            if _bbox_iou(extractor._tracked_bbox, bbox) > 0.5:
                                tracked_idx = i
                                break

                # Draw bounding boxes around all people
                frame = draw_person_boxes(frame, all_bboxes, tracked_idx)

                if kpts is not None:
                    kpt_buffer.append(kpts)  # crop-space for model
                    frame_count += 1
                    webcam_frame_counter += 1

                    # Run inference every <stride> frames once buffer is full
                    if len(kpt_buffer) == window_size and frame_count % stride == 0:
                        window = np.array(kpt_buffer)  # (30, 17, 3)
                        norm_kpts, angles = preprocess_skeleton(window)

                        # Compute extra features: (30, 17, 3) -> (30, 17, 6)
                        skel_features = compute_extra_features(norm_kpts)

                        with torch.no_grad():
                            x = torch.FloatTensor(skel_features).unsqueeze(0).to(device)
                            a = torch.FloatTensor(angles).unsqueeze(0).to(device)
                            ex_logits, form_logits, rep_logits = model(x, a)
                            ex_pred = ex_logits.argmax(1).item()
                            form_pred = form_logits.argmax(1).item()
                            ex_conf = torch.softmax(ex_logits, 1).max().item()
                            form_probs = torch.softmax(form_logits, 1).cpu().numpy()[0]
                            form_score = int(round(100 * form_probs[1]))

                        # Learned rep head: count with cooldown; correct/incorrect from form
                        if use_rep_head:
                            rep_probs = torch.softmax(rep_logits, 1).cpu().numpy()[0]
                            rep_prob = float(rep_probs[1])
                            if rep_prob > 0.5 and webcam_frame_counter >= webcam_last_counted_frame + rep_cooldown:
                                if form_pred == 1:
                                    webcam_correct_reps += 1
                                else:
                                    webcam_incorrect_reps += 1
                                webcam_last_counted_frame = webcam_frame_counter

                        exercise_name = config_ex_names[ex_pred] if ex_pred < len(config_ex_names) else "unknown"
                        # Form from model only (rules used only for labeling training data)
                        current_result = {
                            "exercise": exercise_name,
                            "is_correct": form_pred == 1,
                            "feedback": [],
                            "confidence": ex_conf,
                            "correct_reps": webcam_correct_reps,
                            "incorrect_reps": webcam_incorrect_reps,
                            "form_score": form_score,
                        }

                    # Draw skeleton overlay (convert crop keypoints to full-frame for drawing)
                    kpts_draw = kpts.copy()
                    kpts_draw[..., 0] += roi_ox
                    kpts_draw[..., 1] += roi_oy
                    frame = draw_skeleton(frame, kpts_draw)

                # Draw feedback overlay with rep counter
                if current_result is not None:
                    frame = draw_feedback(
                        frame,
                        exercise_name=current_result["exercise"],
                        is_correct=current_result["is_correct"],
                        feedback_messages=current_result["feedback"],
                        correct_reps=current_result["correct_reps"],
                        incorrect_reps=current_result["incorrect_reps"],
                        form_score=current_result.get("form_score"),
                    )

                # Display frame (convert BGR -> RGB for Streamlit)
                frame_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )

                # Update metrics below the feed
                if current_result is not None:
                    with metrics_placeholder.container():
                        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                        mc1.metric(
                            "Exercise",
                            current_result["exercise"].replace("_", " ").title(),
                        )
                        mc2.metric("Confidence", f"{current_result['confidence']:.0%}")
                        form_score = current_result.get("form_score", 100)
                        mc3.metric("Form Score", f"{form_score}/100")
                        form_label = "Correct" if current_result["is_correct"] else "Needs Fix"
                        mc4.metric("Form", form_label)
                        mc5.metric("Correct Reps", current_result["correct_reps"])
                        mc6.metric("Incorrect Reps", current_result["incorrect_reps"])

                    if current_result["feedback"]:
                        with feedback_placeholder.container():
                            for msg in current_result["feedback"]:
                                st.warning(f"\u26A0\uFE0F {msg}")
                    else:
                        feedback_placeholder.success("\u2705 Good form!")

        finally:
            cap.release()
            extractor.reset_tracking()
            st.session_state.webcam_running = False
