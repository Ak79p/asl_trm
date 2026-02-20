## Pending work:
- TCGN evaluation on existing splits

---

## Feature files:

Place these [feature](https://drive.google.com/drive/folders/1ux0Rw--u7az2avld-O9U0cGItzGKrGw3?usp=sharing) files in respective dataclass folders.

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
python -m eval.eval_trm --dataset asl300 --checkpoint checkpoints/asl300/best_model.pt
```


