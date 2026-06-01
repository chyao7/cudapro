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

echo "==> 编译 lesson10(标准) / lesson12(v1) / lesson13(v2)"
"$NVCC" $OPT -o lesson10 lesson10_flash_attention.cu
"$NVCC" $OPT -o lesson12 lesson12_flash_attention_v1.cu
"$NVCC" $OPT -o lesson13 lesson13_flash_attention_v2.cu
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

echo
echo "演进: lesson10=标准(两遍KV)  lesson12=v1(Online Softmax)  lesson13=v2(FA-2 O片上累积)"
echo "      v1/base、v2/base > 1 表示比标准版快；老 GPU 上 v2 未必更快"
