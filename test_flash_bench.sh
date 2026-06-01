#!/usr/bin/env bash
# FlashAttention 三代对比：lesson10 标准版 / lesson12 v1 / lesson13 v2
#
# 用法:
#   ./test_flash_bench.sh
#   ./test_flash_bench.sh 1024 2048

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

NVCC="${NVCC:-nvcc}"
OPT="${OPT:--O3}"

# GPU 架构：默认从 nvidia-smi 读取 compute_cap（如 8.6 → sm_86）
# 手动覆盖：CUDA_ARCH=sm_86 ./test_flash_bench.sh
if [[ -z "${CUDA_ARCH:-}" ]] && command -v nvidia-smi &>/dev/null; then
    CCAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [[ -n "$CCAP" ]]; then
        CUDA_ARCH="sm_${CCAP}"
    fi
fi
ARCH_FLAGS=()
if [[ -n "${CUDA_ARCH:-}" ]]; then
    ARCH_FLAGS=(-arch="$CUDA_ARCH")
fi

DEFAULT_SEQS=(512 1024 2048 4096 8192)
if [[ $# -gt 0 ]]; then
    SEQS=("$@")
else
    SEQS=("${DEFAULT_SEQS[@]}")
fi

echo "==> 检查 CUDA"
if ! command -v "$NVCC" &>/dev/null; then
    echo "错误: 找不到 $NVCC" >&2
    exit 1
fi
if ! nvidia-smi &>/dev/null; then
    echo "警告: nvidia-smi 不可用，若运行失败请先 sudo reboot 修复驱动" >&2
fi

GPU_NAME=""
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
fi
echo "==> 编译 lesson10(标准) / lesson12(v1) / lesson13(v2)"
echo "    GPU: ${GPU_NAME:-未知}  arch: ${CUDA_ARCH:-默认}"
"$NVCC" $OPT "${ARCH_FLAGS[@]}" -o lesson10 lesson10_flash_attention.cu
"$NVCC" $OPT "${ARCH_FLAGS[@]}" -o lesson12 lesson12_flash_attention_v1.cu
"$NVCC" $OPT "${ARCH_FLAGS[@]}" -o lesson13 lesson13_flash_attention_v2.cu
echo "    完成"
echo

parse_ms() {
    grep -oE '[0-9]+\.[0-9]+ ms' | head -1 | grep -oE '[0-9]+\.[0-9]+'
}

printf "%6s  %12s  %10s  %10s  %10s  %10s\n" \
    "SEQ" "baseline_ms" "v1_ms" "v2_ms" "v1/base" "v2/base"
printf "%6s  %12s  %10s  %10s  %10s  %10s\n" \
    "----" "-----------" "------" "------" "-------" "-------"

for seq in "${SEQS[@]}"; do
    t10=$(./lesson10 "$seq" 2>&1 | parse_ms || echo "nan")
    t12=$(./lesson12 "$seq" 2>&1 | parse_ms || echo "nan")
    t13=$(./lesson13 "$seq" 2>&1 | parse_ms || echo "nan")

    r1=$(awk -v a="$t10" -v b="$t12" 'BEGIN{if(a>0) printf "%.2fx", a/b; else print "n/a"}')
    r2=$(awk -v a="$t10" -v b="$t13" 'BEGIN{if(a>0) printf "%.2fx", a/b; else print "n/a"}')

    printf "%6d  %12s  %10s  %10s  %10s  %10s\n" \
        "$seq" "$t10" "$t12" "$t13" "$r1" "$r2"
done

