Roboflow Supplement Datasets
=============================
These are IMAGE-LEVEL datasets used for form/phase/mistake labels only.
Do NOT treat them as video rep sources.

Recommended datasets to download from roboflow.com:

1. "Workout Reps Counter" (CC BY 4.0)
   Classes: Squats_Start, Squats_End, Pushup_Start, Pushup_End, etc.
   -> Used as phase labels (rep_start / rep_end state images)

2. "More fit and strong" (CC BY 4.0)
   Classes: wrong-squat, wrong-lunge, wrong-push-up variants
   -> Used as incorrect-form image labels

3. "Exercise-detector" (Public Domain)
   Classes: bench_press, shoulder_press, squat, push_up, barbell_biceps_curl, etc.
   -> Used for exercise classification image pretraining

Expected structure after download (export as CSV format from Roboflow):
  workout_reps_counter/
    _annotations.csv     <- columns: filename, class, ...
  more_fit_strong/
    _annotations.csv
  exercise_detector/
    _annotations.csv

NOTE: Rows from Roboflow will have empty rep_start and rep_end.
They contribute to form_label counts only.
