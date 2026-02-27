# ASL-TRM: Transformer Reasoning Model for ASL Gesture Recognition

ASL-TRM is a lightweight, Transformer-based gesture recognition framework built for American Sign Language (ASL) classification using pose and motion features. The system processes ASL video input, extracts structured spatial–temporal representations, and applies transformer-based reasoning to translate signed gestures into text.

At its core, ASL-TRM leverages an efficient architecture, TRM-Micro, designed for scalability while maintaining strong recognition accuracy. By modeling temporal dependencies across frames, the framework enables reliable video-to-text translation from ASL gestures.

---

## 📦 Architecture Overview

### High-Level Pipeline
```bash
Video (.mp4)
↓
MediaPipe Holistic Keypoint Extraction
↓
Temporal Normalization + Motion Encoding
↓
Feature Tensor (Frames × Joints × Features)
↓
Input Projection
↓
Pre-Encoder Transformer
↓
Latent Z-Token Reasoning Block
↓
Post-Encoder Transformer
↓
Global Pooling
↓
Classifier → Gloss Prediction
```


### TRM-Micro (Efficient Transformer Model)

- **Input Representation:** Pose + motion features per frame
- **Pre-Encoder:** Light temporal processing
- **Latent Z-Tokens:** Small set of learnable abstract tokens for reasoning
- **Post-Encoder:** Joint reasoning + perception
- **Classifier:** Dataset-specific output (ASL-100 / ASL-300)

This architecture enables strong temporal modeling with **≈486K parameters** while still delivering high accuracy.

---

## 📊 TRM-Micro Model Size

| Vocab size | Total Parameters |
|------------------|----------------------|
| ASL 100       | 460,901             |
| ASL 300       | 486,701            |
| ASL 1000      | 577,001           |
| ASL 2000       | 706,001            |


---

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Features

Download [features](https://drive.google.com/file/d/1dED4DEU5G0KBTgFJkw0gy20561y5q98Z/view?usp=drive_link) and place the subsets in data/ folder

After unzipping features_cache in data/ 
```bash
cd data
python build_features.py --dataset asl100
```
The above command will generate feature files for model training and evaluation

---

## 🏋️ Training
Run from root

### Train on ASL-300
```bash
python -m train.train --dataset asl300
```

### Train on ASL-100
```bash
python -m train.train --dataset asl100
```

---

## 🧪 Evaluation

### Evaluate on ASL-300
```bash
python -m eval.eval_trm --dataset asl300
```

### Evaluate on ASL-100
```bash
python -m eval.eval_trm --dataset asl100
```

### Evaluate with custom checkpoint
```bash
python -m eval.eval_trm \
  --dataset asl300 \
  --checkpoint checkpoints/asl300/best_model.pt
```

---

## 🎥 Inference (Single Video)

### Run inference on a new unseen video and get top-k predictions with confidence:
```bash
python -m inference.infer_video \
  --video inference/test2.mp4 \
  --dataset asl300
```
---

## 📈 Results

### ASL-100 Evaluation
| Metric    | Score      |
| --------- | ---------- |
| Top-1     | **73.4%** |
| Top-5     | **88.63%** |
| Top-10    | **92.41%** |

### ASL-300 Evaluation

| Metric    | Score      |
| --------- | ---------- |
| Top-1     | **67.31%** |
| Top-5     | **85.13%** |
| Top-10    | **89.36%** |

### ASL-1000 Evaluation

| Metric    | Score      |
| --------- | ---------- |
| Top-1     | **64.95%** |
| Top-5     | **83.72%** |
| Top-10    | **88.23%** |

### ASL-2000 Evaluation

| Metric    | Score      |
| --------- | ---------- |
| Top-1     | **59.28%** |
| Top-5     | **80.50%** |
| Top-10    | **85.42%** |

---

## 📍 Dataset Details

This project uses the ASL-Citizen dataset created by Microsoft. It includes:

- 83,339 sign language videos
- Gloss annotations for each video
- Multiple signers and variations
- Rich vocabulary suitable for real-world ASL tasks

Dataset details and downloads available here:

🔗 https://www.microsoft.com/en-us/research/project/asl-citizen/

🔗 https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/

---

## 🛣️ Pending work

- Further experimentation with TRM-ASL model for improvements.
- Implement a sentence formation layer to convert recognized word sequences into coherent textual output.
- Deploy the user interface (currently internal) as a publicly accessible application.
- Extend the dataset to include sentence-level ASL videos and improve stop-sign detection for accurate gesture segmentation.
- Optimize system latency and robustness to support reliable real-time performance.


---

### 🫶 Acknowledgements

This work builds upon:

- Microsoft’s ASL-Citizen dataset
- MediaPipe Holistic keypoint extraction
