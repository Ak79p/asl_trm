## Pending work:
- TCGN evaluation on existing splits

---

## Feature files:
[wlasl100](https://drive.google.com/drive/folders/17-7Y7ofXFbIksLnoWpfz5lM-jAtQWI4o?usp=drive_link)
[wlasl300](https://drive.google.com/drive/folders/1HkS0fyUyg5OeiwAXJQI6RqRp0otLoxvX?usp=sharing)
[wlasl1000](https://drive.google.com/drive/folders/10Oxx3riZpxXl22d63UXjIHi8kFr2Dbhj?usp=sharing)
[wlasl2000](https://drive.google.com/drive/folders/1X4wniK7-1nVPxDguT5jl9EanehtIKbsC?usp=sharing)

Place these feature files in respective dataclass folders.

After placing the feature files you should have:
- data/asl100/features_cache/
- data/asl300/features_cache/
- data/asl1000/features_cache/
- data/asl2000/features_cache/
  
---

## WLASL on Test split

### WLASL-100
| Metric    | Trm-micro  | TGCN   |
| --------- | ---------- | ----------  |
| Top-1     | **69.65%** |  |
| Top-5     | **86.77%** | |
| Top-10    | **89.11%** ||

### WLASL-300
| Metric    | Trm-micro  | TGCN   |
| --------- | ---------- | ----------  |
| Top-1     | **58.76%** |  |
| Top-5     | **81.27%** | |
| Top-10    | **87.92%** ||

### WLASL-1000
| Metric    | Trm-micro  | TGCN   |
| --------- | ---------- | ----------  |
| Top-1     | **39.72%** |  |
| Top-5     | **66.99%** | |
| Top-10    | **75.41%** ||

### WLASL-2000
| Metric    | Trm-micro  | TGCN   |
| --------- | ---------- | ----------  |
| Top-1     | **33.36%** |  |
| Top-5     | **60.23%** | |
| Top-10    | **69.17%** ||

---

## Commands

- To visualise previous training run use this command
```bash
tensorboard --logdir=runs
```

- To evaluate test split
```bash
python -m eval.eval_trm --dataset asl300
```

- To run training
```bash
python -m train.train --dataset asl300
```

- To evaluate on custom checkpoints
```bash
python -m eval.eval_trm_micro --dataset asl300 --checkpoint checkpoints/asl300/best_model.pt
```
