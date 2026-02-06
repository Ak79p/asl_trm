## Inference
python -m inference.infer_video \
  --video inference/test2.mp4 \
  --dataset asl300

## Test ASL - 300
python -m eval.eval_trm --dataset asl300

## Train ASL-300
python -m train.train --dataset asl300

## Test ASL - 100
python -m eval.eval_trm --dataset asl100

## Train ASL-100
python -m train.train --dataset asl100

# Evaluate with custom checkpoint
python eval/eval_trm_micro.py \
  --dataset asl300 \
  --checkpoint checkpoints/asl300/best_model.pt


## TRM-Micro (4K params):
Total parameters     : 486,700
Trainable parameters : 486,700

asl300:
===== TEST RESULTS =====
Test Loss : 1.4970
Top-1     : 66.68%
Top-5     : 86.00%
Top-10    : 90.45%
========================


asl100:
===== TEST RESULTS =====
Test Loss : 1.2285
Top-1     : 73.05%
Top-5     : 89.13%
Top-10    : 93.12%
========================
