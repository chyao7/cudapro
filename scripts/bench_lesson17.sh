#!/usr/bin/env bash
# CUDA lesson17 vs PyTorch，用法: ./scripts/bench_lesson17.sh [tokens]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOKENS="${1:-4096}"

"$ROOT/scripts/build.sh" release lesson17_qkv_rmsnorm

echo "======== CUDA (Release) tokens=$TOKENS ========"
"$ROOT/build-release/bin/lesson17_qkv_rmsnorm" "$TOKENS"

echo
echo "======== PyTorch tokens=$TOKENS ========"
python3 "$ROOT/benchmarks/bench_qkv_rmsnorm.py" \
  --tokens "$TOKENS" --dtype float16 --warmup 10 --iters 50
