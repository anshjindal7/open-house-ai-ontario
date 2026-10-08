#!/usr/bin/env bash
# Fine-tune Qwen3-8B with LoRA on Apple Silicon, on data from train/build_dataset.py.
# Run from the project root:  bash train/train.sh      (ITERS=10 for a smoke test; RESUME=path/to/adapters.safetensors to continue)
#
# Defaults to one pass over the training set. Watch "Val loss": if it rises while train
# loss keeps falling, the model is memorizing; stop earlier or add data.
set -euo pipefail

BATCH=${BATCH:-2}
TRAIN_ROWS=$(wc -l < train/data/train.jsonl)
ITERS=${ITERS:-$(( TRAIN_ROWS / BATCH ))}

python -m mlx_lm lora \
  --model "${BASE:-mlx-community/Qwen3-8B-4bit}" \
  --train \
  --data train/data \
  --adapter-path "${ADAPTER_PATH:-train/adapters}" \
  --fine-tune-type lora \
  --num-layers 16 \
  --batch-size "$BATCH" \
  --iters "$ITERS" \
  --learning-rate 1e-4 \
  --max-seq-length 2560 \
  --mask-prompt \
  --grad-checkpoint \
  --steps-per-report 10 \
  --steps-per-eval "${EVAL_EVERY:-100}" \
  --val-batches "${VAL_BATCHES:-20}" \
  --save-every "${SAVE_EVERY:-100}" \
  ${RESUME:+--resume-adapter-file "$RESUME"}
