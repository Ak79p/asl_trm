## Feature files:

Place these [feature](https://drive.google.com/drive/folders/1ux0Rw--u7az2avld-O9U0cGItzGKrGw3?usp=sharing) files in respective dataclass folders.

After placing the feature files you should have:
- data/asl100/features_cache/
- data/asl300/features_cache/
- data/asl1000/features_cache/
- data/asl2000/features_cache/
  
---

## Baseline Evaluation

We evaluated TRM-Micro v1 against the TGCN baseline using the same datasets originally used for TGCN.

(**Note:** Some videos from the original dataset were missing. We evaluated on the available subset used in TGCN training and evaluation. We have contacted the authors for the missing videos but have not received a response yet.)

### Evaluation on Available Original Videos
**WLASL100**
| Model            | Top-1 | Top-5 | Top-10 |
| ---------------- | ----- | ----- | ------ |
| **TGCN**         | 55.43 | 78.68 | 87.60  |
| **TRM-Micro v1** | 69.65 | 86.77 | 89.11  |

**WLASL300**
| Model            | Top-1 | Top-5 | Top-10 |
| ---------------- | ----- | ----- | ------ |
| **TGCN**         | 38.32 | 67.51 | 79.64  |
| **TRM-Micro v1** | 58.76 | 81.27 | 87.92  |

**WLASL1000**
| Model            | Top-1 | Top-5 | Top-10 |
| ---------------- | ----- | ----- | ------ |
| **TGCN**         | 34.86 | 61.73 | 71.91  |
| **TRM-Micro v1** | 39.72 | 66.99 | 75.41  |



### Evaluation Matching Baseline Split (With Replacements)

To strictly match the original TGCN data split, missing videos were replaced using ASL-Citizen videos, and evaluation was conducted on the identical split used in the TGCN paper.
**100**
| Model            | Top-1 | Top-5 | Top-10 |
| ---------------- | ----- | ----- | ------ |
| **TGCN**         | 55.43 | 78.68 | 87.60  |
| **TRM-Micro v1** | 69.65 | 86.77 | 89.11  |

**300**
| Model            | Top-1 | Top-5 | Top-10 |
| ---------------- | ----- | ----- | ------ |
| **TGCN**         | 38.32 | 67.51 | 79.64  |
| **TRM-Micro v1** | 58.76 | 81.27 | 87.92  |

**1000**
| Model            | Top-1 | Top-5 | Top-10 |
| ---------------- | ----- | ----- | ------ |
| **TGCN**         | 34.86 | 61.73 | 71.91  |
| **TRM-Micro v1** | 39.72 | 66.99 | 75.41  |


### Model Size Comparison
| Dataset  | TGCN    | TRM-Micro v1 |
| -------- | ------- | ------------ |
| **100**  | 592,029 | 460,901      |
| **300**  | 605,029 | 486,701      |
| **1000** | 806,156 | 577,001      |

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



