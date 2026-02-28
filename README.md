# ASL-TRM: Transformer Reasoning Model for ASL Gesture Recognition

ASL-TRM is a lightweight, Transformer-based gesture recognition framework built for American Sign Language (ASL) classification using pose and motion features. The system processes ASL video input, extracts structured spatial–temporal representations, and applies transformer-based reasoning to translate signed gestures into text.

At its core, ASL-TRM leverages an efficient architecture, TRM-Micro, designed for scalability while maintaining strong recognition accuracy. By modeling temporal dependencies across frames, the framework enables reliable video-to-text translation from ASL gestures.

---

## Architecture Overview

![Arch diagram](./assets/arch.jpeg)

---

## Dataset:

This project uses the **ASL-Citizen dataset** from Microsoft, a large collection of sign language video samples and corresponding gloss annotations.

🔗 [**Dataset**](https://www.microsoft.com/en-us/research/project/asl-citizen/) 

🔗 [**Dataset Description**](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/)

---

## Inspiration

1. *Less is More: Recursive Reasoning with Tiny Networks*] [(link)](https://arxiv.org/pdf/2510.04871)
   
   This paper introduces a compact Transformer reasoning model designed to learn structured reasoning through latent token abstraction and recursive interaction. Although the original TRM was developed for general reasoning tasks and not for gesture recognition, its core ideas — latent reasoning slots, shared Transformer layers, and efficient recursive updates directly influenced how we built TRM-Micro for ASL. We adapted these principles to handle spatio-temporal pose and motion features, enabling lightweight yet expressive sequence reasoning in the context of sign language translation.

2. *Word-Level Deep Sign Language Recognition — TGCN Baseline* [(Link)](https://arxiv.org/pdf/1910.11006)

   This paper presents a Temporal Graph Convolutional Network (TGCN) for sign language recognition that operates on skeleton/pose features extracted from video. The TGCN models spatial dependencies between joints and temporal dynamics across frames, making it a strong baseline for pose-based action/gesture recognition tasks. It serves as our baseline model because it demonstrates how pose representations can be leveraged for sign language classification, providing a meaningful point of comparison for the performance of TRM-Micro.

---

## Model Evaluation

We evaluate TRM-Micro on multiple ASL classification benchmarks of increasing vocabulary size:
- ASL 100
- ASL 300
- ASL 1000
- ASL 2000

### Evaluation Metrics

We report Top-1, Top-5, and Top-10 accuracy:

- Top-1 → Correct label is the highest-confidence prediction
- Top-5 → Correct label appears within top 5 predictions
- Top-10 → Correct label appears within top 10 predictions

### TRM-Micro v2
| Metric     | ASL 100 | ASL 300 | ASL 1000 | ASL 2000 |
| ---------- | ------- | ------- | -------- | -------- |
| **Top-1**  | 73.89%  | 70.67%  | 67.86%   | 64.15%   |
| **Top-5**  | 89.98%  | 88.09%  | 86.38%   | 84.12%   |
| **Top-10** | 93.01%  | 91.47%  | 90.22%   | 88.03%   |


### TRM-Micro v1
| Metric     | ASL 100 | ASL 300 | ASL 1000 | ASL 2000 |
| ---------- | ------- | ------- | -------- | -------- |
| **Top-1**  | 73.40%  | 67.31%  | 64.95%   | 59.28%   |
| **Top-5**  | 88.63%  | 85.13%  | 83.72%   | 80.50%   |
| **Top-10** | 92.41%  | 89.36%  | 88.23%   | 85.42%   |


### Model Size

| Dataset      | TRM-Micro v1 | TRM-Micro v2 |
| ------------ | ------------ | ------------ |
| **ASL 100**  | 460,901      | 710,022      |
| **ASL 300**  | 486,701      | 742,222      |
| **ASL 1000** | 577,001      | 854,922      |
| **ASL 2000** | 706,001      | 1,015,922    |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/Ak79p/asl_trm.git
cd asl-trm

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit interface
streamlit run app.py
```
---

## How to Evaluate

Follow the steps below to reproduce evaluation results.

1. Download the [feature](https://drive.google.com/file/d/1226Akrt3-plnf3-8z-ot9YEye-S51LbZ/view?usp=drive_link) archive and extract it.

Place the extracted features_cache/ folder inside:
```bash
data/
```

Directory structure should look like:
```bash
asl-trm/
│
├── data/
│   ├── features_cache/
│   └── build_features.py
```

2. Build Dataset Features

```bash
cd data
python build_features.py --dataset asl100
```

3. Run Evaluation
```bash
cd ..
python -m eval.eval_trm --dataset asl100
```

### Evaluating Other Datasets
```bash
cd data
python build_features.py --dataset asl300
cd ..
python -m eval.eval_trm --dataset asl300
```

### Baseline Evaluation
To reproduce evaluation on wlasl videos, refer to the: [wlas-exp](https://github.com/Ak79p/asl_trm/tree/wlasl-exp)

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
