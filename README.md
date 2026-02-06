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
