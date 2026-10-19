Real-Time Exercise Recognition Dataset
=======================================
Download from: https://huggingface.co/datasets/...  (CC BY-NC-SA 4.0)

Expected folder structure after download:
  videos/
    bench_press/
      video_001.mp4
      ...
    bicep_curl/
      ...
    shoulder_press/
      ...
    push_up/
      ...

OR a manifest CSV:
  manifest.csv  with columns: filename, label (or class)

Used for: EXERCISE CLASSIFICATION only (no temporal rep labels).
rep_start, rep_end will be empty; confidence=0.60.

Exercises that map to this project's 4 classes:
  bench_press          -> bench_press
  barbell_bicep_curl /
  bicep_curl           -> biceps
  shoulder_press /
  overhead_press       -> shoulder_press
  tricep_extension /
  tricep_dip           -> triceps
