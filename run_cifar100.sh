#!/bin/bash
set -e

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "Starting CIFAR-100 experiments"
echo "========================================="

# ── CIFAR-100: 20 rounds (light) ──────────────
echo "[1/3] CIFAR-100 R=20"
python -u run_all_cases.py --cases 0-6 \
    --dataset cifar100 \
    --num_rounds 20 \
    --num_unlearn_rounds 5 \
    --num_post_training_rounds 20 \
    2>&1 | tee "$LOG_DIR/cifar100_R20.log"
echo "[1/3] CIFAR-100 R=20 done"

# ── CIFAR-100: 50 rounds (standard) ───────────
echo "[2/3] CIFAR-100 R=50"
python -u run_all_cases.py --cases 0-6 \
    --dataset cifar100 \
    --num_rounds 50 \
    --num_unlearn_rounds 5 \
    --num_post_training_rounds 30 \
    2>&1 | tee "$LOG_DIR/cifar100_R50.log"
echo "[2/3] CIFAR-100 R=50 done"

# ── CIFAR-100: 100 rounds (full) ──────────────
echo "[3/3] CIFAR-100 R=100"
python -u run_all_cases.py --cases 0-6 \
    --dataset cifar100 \
    --num_rounds 100 \
    --num_unlearn_rounds 10 \
    --num_post_training_rounds 30 \
    2>&1 | tee "$LOG_DIR/cifar100_R100.log"
echo "[3/3] CIFAR-100 R=100 done"

echo "========================================="
echo "All CIFAR-100 experiments completed"
echo "========================================="
