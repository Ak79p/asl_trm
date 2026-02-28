## Baseline Evaluation

We evaluated TRM-Micro v1 against the TGCN baseline using the same datasets originally used for TGCN.

(**Note:** Some videos from the original dataset were missing. We evaluated on the available subset used in TGCN training and evaluation. We have contacted the authors for the missing videos but have not received a response yet.)

### Evaluation on Available Original Videos
**WLASL100**
| Model            | Top-1 | Top-5 | Top-10 |
| ---------------- | ----- | ----- | ------ |
| **TGCN**         | 37.00 | 59.00 | 66.00  |
| **TRM-Micro v1** | 49.00 | 83.00| 87.00 |

**WLASL300**
| Model            | Top-1 | Top-5 | Top-10 |
| ---------------- | ----- | ----- | ------ |
| **TGCN**         | 27.40| 54.57 | 65.62  |
| **TRM-Micro v1** | 38.30 | 70.35 | 79.18 |

**WLASL1000**
| Model            | Top-1 | Top-5 | Top-10 |
| ---------------- | ----- | ----- | ------ |
| **TGCN**         | 18.17 | 42.93 | 53.13  |
| **TRM-Micro v1** | 25.29 | 55.79| 67.48  |





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

## How to Evaluate

Follow the steps below to reproduce evaluation results.

1. Place these [feature](https://drive.google.com/drive/folders/1ux0Rw--u7az2avld-O9U0cGItzGKrGw3?usp=sharing) files in respective dataclass folders.

After placing the feature files you should have:
- data/asl100/features_cache/
- data/asl300/features_cache/
- data/asl1000/features_cache/
- data/asl2000/features_cache/

2. Run Evaluation
```bash
python -m eval.eval_trm --dataset asl100
```

### Evaluating Other Datasets
```bash
python -m eval.eval_trm --dataset asl300
```




