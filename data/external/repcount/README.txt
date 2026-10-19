RepCount Dataset
================
Download from: https://github.com/SvipRepetitionCounting/TransRAC
Paper: "TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetition Action Counting"

Expected files after download:
  annotation_train.json
  annotation_val.json
  annotation_test.json   (optional)
  videos/                (optional — only needed if extracting skeletons)

JSON format:
  {
    "video_name.mp4": {
      "count": 10,
      "frame_number": [30, 65, 100, ...]   <- rep-end frames
    },
    ...
  }

OR newer format:
  {
    "video_name.mp4": {
      "count": 10,
      "begin": [0, 35, 70, ...],
      "end":   [30, 65, 100, ...]
    },
    ...
  }

Exercises that map to this project's 4 classes:
  bench_press  -> bench_press
  (push_up, pull_up, squat, jumping_jacks, front_raise are excluded)
