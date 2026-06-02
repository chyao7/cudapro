#!/usr/bin/env python3
"""
PyTorch 版 QKV + QK-RMSNorm 分段耗时

对齐模型代码:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = self.q_norm(q.view(..., num_heads, head_dim)).view_as(q)
    k = self.k_norm(k.view(..., num_kv_heads, head_dim)).view_as(k)

默认尺寸与 lesson17一致 (hidden=512, GQA 8/4 heads, head_dim=64)

用法:
    python bench_qkv_rmsnorm.py
    python bench_qkv_rmsnorm.py --tokens 4096
    python bench_qkv_rmsnorm.py --tokens 128 --dtype float16 --warmup 20 --iters 100
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def ms_since(t0: float) -> float:
    sync()
    return (time.perf_counter() - t0) * 1000.0


class QKVRmsNormBlock(nn.Module):
    """最小 QKV + QK-RMSNorm 模块，结构对齐 HuggingFace / vLLM 常见写法。"""

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_q_heads * head_dim
        self.kv_size = num_kv_heads * head_dim

        self.qkv_proj = nn.Linear(
            hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=False,
        )
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)

    def forward_qkv_proj(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.qkv_proj(hidden)

    def forward_qk_norm(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], self.num_q_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view_as(q)

        k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view_as(k)

        return q, k, v


@torch.no_grad()
def bench_stages(
    model: QKVRmsNormBlock,
    hidden: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    # warmup 整段
    for _ in range(warmup):
        qkv = model.forward_qkv_proj(hidden)
        model.forward_qk_norm(qkv)
    sync()

    t_proj = t_qnorm = t_knorm = t_full = 0.0

    for _ in range(iters):
        t0 = time.perf_counter()
        qkv = model.forward_qkv_proj(hidden)
        t_proj += ms_since(t0)

        t0 = time.perf_counter()
        q, k, v = qkv.split(
            [model.q_size, model.kv_size, model.kv_size], dim=-1
        )
        q_by_head = q.view(*q.shape[:-1], model.num_q_heads, model.head_dim)
        q_by_head = model.q_norm(q_by_head)
        q = q_by_head.view_as(q)
        t_qnorm += ms_since(t0)

        t0 = time.perf_counter()
        k_by_head = k.view(*k.shape[:-1], model.num_kv_heads, model.head_dim)
        k_by_head = model.k_norm(k_by_head)
        k = k_by_head.view_as(k)
        _ = v  # split/view 不计入 k_norm 段
        t_knorm += ms_since(t0)

        t0 = time.perf_counter()
        qkv = model.forward_qkv_proj(hidden)
        q, k, v = model.forward_qk_norm(qkv)
        t_full += ms_since(t0)
        del q, k, v, qkv

    n = float(iters)
    return {
        "qkv_proj": t_proj / n,
        "q_norm": t_qnorm / n,
        "k_norm": t_knorm / n,
        "full": t_full / n,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyTorch QKV + QK-RMSNorm benchmark")
    p.add_argument("--tokens", type=int, default=128, help="seq len (batch×seq)")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--num-q-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="与 lesson17/18 FP16 对齐时选 float16",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda")

    q_size = args.num_q_heads * args.head_dim
    kv_size = args.num_kv_heads * args.head_dim
    qkv_dim = q_size + 2 * kv_size

    torch.manual_seed(args.seed)
    model = QKVRmsNormBlock(
        args.hidden,
        args.num_q_heads,
        args.num_kv_heads,
        args.head_dim,
    ).to(device=device, dtype=dtype)
    model.eval()

    hidden = torch.randn(args.tokens, args.hidden, device=device, dtype=dtype)

    # 一次正确性 smoke test
    with torch.no_grad():
        qkv = model.forward_qkv_proj(hidden)
        q, k, v = model.forward_qk_norm(qkv)
    assert q.shape == (args.tokens, q_size)
    assert k.shape == (args.tokens, kv_size)
    assert v.shape == (args.tokens, kv_size)

    times = bench_stages(model, hidden, args.warmup, args.iters)

    gemm_flops = 2.0 * args.tokens * args.hidden * qkv_dim
    t_proj = times["qkv_proj"]
    t_q = times["q_norm"]
    t_k = times["k_norm"]
    t_full = times["full"]
    t_norm = t_q + t_k

    props = torch.cuda.get_device_properties(device)
    print(f"PyTorch QKV + QK-RMSNorm  —  {props.name}")
    print(f"hidden [{args.tokens}×{args.hidden}] → qkv [{args.tokens}×{qkv_dim}]")
    print(
        f"  q: {args.num_q_heads} heads × {args.head_dim}  |  "
        f"k/v: {args.num_kv_heads} heads × {args.head_dim}  |  dtype={args.dtype}"
    )
    print()
    print("── 对齐 PyTorch 分段耗时 ──")
    print("  split(q,k,v) / view  : ~0 ms  (无 kernel)")
    print(f"  ① qkv_proj (Linear)  : {t_proj:8.3f} ms  ({100*t_proj/t_full:5.1f}%)")
    print(f"  ② q_norm  (RMSNorm)  : {t_q:8.3f} ms  ({100*t_q/t_full:5.1f}%)")
    print(f"  ③ k_norm  (RMSNorm)  : {t_k:8.3f} ms  ({100*t_k/t_full:5.1f}%)")
    print(f"  分段合计 ①+②+③      : {t_proj+t_norm:8.3f} ms")
    print(f"  整段 forward (实测)  : {t_full:8.3f} ms")
    print()
    print(f"  GEMM TFLOPS ≈ {gemm_flops / (t_proj * 1e6):.1f}")
    print(f"  qk-norm 占整段 ≈ {100 * t_norm / t_full:.1f}%")
    print()
    print("对比 CUDA lesson17/18: 同尺寸下可运行")
    print(f"  ./lesson17 {args.tokens}")
    print(f"  ./lesson18 {args.tokens}")


if __name__ == "__main__":
    main()
