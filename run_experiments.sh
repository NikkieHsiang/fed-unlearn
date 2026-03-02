#!/bin/bash
set -e  # 任意命令失败则终止

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "Starting all experiments"
echo "========================================="

# ── CIFAR-10 ──────────────────────────────────
echo "[1/3] CIFAR-10 (cases 0-6)"
python -u run_all_cases.py --cases 0-6 \
    --dataset cifar10 \
    --num_rounds 20 \
    --num_unlearn_rounds 5 \
    --num_post_training_rounds 20 \
    2>&1 | tee "$LOG_DIR/cifar10.log"
echo "[1/3] CIFAR-10 done"

# ── CIFAR-100 ─────────────────────────────────
echo "[2/3] CIFAR-100 (cases 0-6)"
python -u run_all_cases.py --cases 0-6 \
    --dataset cifar100 \
    --num_rounds 50 \
    --num_unlearn_rounds 5 \
    --num_post_training_rounds 30 \
    2>&1 | tee "$LOG_DIR/cifar100.log"
echo "[2/3] CIFAR-100 done"

# ── MNIST ─────────────────────────────────────
echo "[3/3] MNIST (cases 0-6)"
python -u run_all_cases.py --cases 0-6 \
    --dataset mnist \
    --num_rounds 10 \
    --num_unlearn_rounds 5 \
    --num_post_training_rounds 10 \
    2>&1 | tee "$LOG_DIR/mnist.log"
echo "[3/3] MNIST done"

echo "========================================="
echo "All experiments completed"
echo "========================================="
