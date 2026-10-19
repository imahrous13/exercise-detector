Countix-Fitness Dataset
=======================
Download from: https://github.com/ryanwongsa/CountixFitness
Paper: "Counting Out Time: Class Agnostic Video Repetition Counting in the Wild"

Expected files after download:
  countix_fitness_train.csv
  countix_fitness_val.csv
  OR
  annotations/train.csv, annotations/val.csv

Required columns:
  video_id (or name), class (or label), start_frame, end_frame
  (salient pose frame annotations if available)

Used for: temporal exercise localization and rep counting.

Exercises that map to this project's 4 classes:
  bicep_curl           -> biceps
  shoulder_press       -> shoulder_press
  tricep_extension     -> triceps
  bench_press          -> bench_press
  (lunge, rope_push_down, and others are excluded)
