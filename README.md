# 🧠 ASL-TRM: Transformer Reasoning Model for ASL Gesture Recognition

ASL-TRM is a **lightweight, Transformer-based gesture recognition framework** built for American Sign Language (ASL) classification using pose and motion features. It achieves strong accuracy with an efficient model (TRM-Micro) designed for scalability and real-time use.

This project uses the **ASL-Citizen dataset** from Microsoft, a large collection of sign language video samples and corresponding gloss annotations.

🔗 **Dataset:** https://www.microsoft.com/en-us/research/project/asl-citizen/  
🔗 **Dataset Description:** https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/

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

## 📊 Model Size

| Model       | Total Parameters | Trainable Parameters |
|-------------|------------------|----------------------|
| TRM-Micro   | 486,700          | 486,700              |


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

## 🏋️ Training

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
python eval/eval_trm_micro.py \
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
=======
## TRM-Micro (6K params):
Total parameters     : 486,700
Trainable parameters : 486,700

asl300:
===== TEST RESULTS =====
Test Loss : 1.4723
Top-1     : 68.57%
Top-5     : 87.13%
Top-10    : 90.90%
========================
>>>>>>> 457afe00 (added global reasoning and updated frame extraction to 48)

Output includes:
- Top-1 predicted gloss + confidence
- Top-5 predictions
- Top-10 predictions

---

## 📈 Results

### ASL-300 Evaluation

| Metric    | Score      |
| --------- | ---------- |
| Test Loss | **1.4970** |
| Top-1     | **66.68%** |
| Top-5     | **86.00%** |
| Top-10    | **90.45%** |

### ASL-100 Evaluation
| Metric    | Score      |
| --------- | ---------- |
| Test Loss | **1.2285** |
| Top-1     | **73.05%** |
| Top-5     | **89.13%** |
| Top-10    | **93.12%** |

---

## 📍 Dataset Details

This project uses the ASL-Citizen dataset created by Microsoft. It includes:

- Thousands of sign language videos
- Gloss annotations for each video
- Multiple signers and variations
- Rich vocabulary suitable for real-world ASL tasks

Dataset details and downloads available here:

🔗 https://www.microsoft.com/en-us/research/project/asl-citizen/

🔗 https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/

---

## 🛣️ Pending work

- Support for ASL-1000 and ASL-2000
- Sentence formation layer
- Development of asl webapp

---

### 🫶 Acknowledgements

This work builds upon:

- Microsoft’s ASL-Citizen dataset
- MediaPipe Holistic keypoint extraction

<<<<<<< HEAD
=======
asl100:
===== TEST RESULTS =====
Test Loss : 1.1878
Top-1     : 72.87%
Top-5     : 89.24%
Top-10    : 92.89%
========================
>>>>>>> 457afe00 (added global reasoning and updated frame extraction to 48)
