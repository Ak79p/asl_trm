# Train ASL-300
python -m train.train --dataset asl300

# Train ASL-100
python -m train.train --dataset asl100

# Test ASL - 100
python -m eval.eval_trm --dataset asl100

# Test ASL - 300
python -m eval.eval_trm --dataset asl300

# Inference
python -m inference.infer_video \
  --video inference/test2.mp4 \
  --dataset asl300

# Evaluate with custom checkpoint
python eval/eval_trm_micro.py \
  --dataset asl300 \
  --checkpoint checkpoints/asl300/best_model.pt




python -m eval.eval_trm 

python -m inference.infer_video path/to/unseen_video.mp4


asl_trm
├── checkpoints
│   └── best_model.pt
├── data
│   ├── asl100
│   └── asl300
│       ├── features_cache
│       ├── label_map.json
│       ├── test_features.csv
│       ├── train_features.csv
│       └── val_features.csv
├── eval
│   ├── __pycache__
│   │   ├── eval_trm_micro.cpython-310.pyc
│   │   └── eval_trm.cpython-310.pyc
│   └── eval_trm.py
├── features
│   ├── __pycache__
│   │   ├── build_tensor.cpython-310.pyc
│   │   ├── compute_motion.cpython-310.pyc
│   │   ├── extract_keypoints.cpython-310.pyc
│   │   ├── temporal_sampling.cpython-310.pyc
│   │   └── video_to_keypoints.cpython-310.pyc
│   ├── build_tensor.py
│   ├── compute_motion.py
│   ├── extract_and_cache.py
│   ├── extract_keypoints.py
│   ├── temporal_sampling.py
│   └── video_to_keypoints.py
├── inference
│   ├── __pycache__
│   │   └── infer_video.cpython-310.pyc
│   ├── infer_video.py
│   ├── test.mp4
│   ├── test2.mp4
│   ├── test3.mp4
│   ├── test4.mp4
│   ├── test5.mp4
│   ├── test6.mp4
│   └── test7.mp4
├── models
│   ├── __pycache__
│   │   ├── count_params.cpython-310.pyc
│   │   └── trm_micro.cpython-310.pyc
│   ├── count_params.py
│   └── trm_micro.py
├── README.md
├── requirements.txt
└── train
    ├── __pycache__
    │   └── train.cpython-310.pyc
    └── train.py
