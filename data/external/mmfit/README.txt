MM-Fit Dataset
==============
Download from: https://mmfit.github.io/  /  Zenodo (CC BY 4.0)
Paper: "MM-Fit: Multimodal Data for Wearable-based Workout and Exercise Monitoring"

Expected files after download:
  workouts.csv         <- one row per exercise set
  OR
  annotations.csv

Required columns (any name mapping is handled in adapt_mmfit):
  subject_id, workout_id, exercise, start_frame, end_frame
  (rep_count per set is available but per-rep boundaries are NOT)

This dataset is used for EXERCISE CLASSIFICATION only.
rep_start and rep_end will be left empty (confidence=0.50).

Exercises that map to this project's 4 classes:
  bicep_curls          -> biceps
  overhead_press /
  shoulder_press       -> shoulder_press
  tricep_extensions    -> triceps
  bench_press          -> bench_press
  (squats, lunges, push-ups, etc. are excluded)
