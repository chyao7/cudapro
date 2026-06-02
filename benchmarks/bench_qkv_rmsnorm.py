#!/usr/bin/env python3
"""
PyTorch 版 QKV + QK-RMSNorm + RotaryEmbedding 分段耗时

对齐模型代码:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = self.q_norm(q.view(..., num_heads, head_dim)).view_as(q)
    k = self.k_norm(k.view(..., num_kv_heads, head_dim)).view_as(k)
    q, k = self.rotary_emb(positions, q, k)

RoPE 参数 (default):
    rope_type = "default", rope_theta = 1_000_000, is_neox_style = True

默认尺寸与 lesson17 一致 (hidden=512, GQA 8/4 heads, head_dim=64)

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


def apply_rotary_emb_neox(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """对齐 vLLM ApplyRotaryEmb.forward_static (Neox style)."""
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(nn.Module):
    """对齐 vLLM RotaryEmbeddingBase + forward_native (default RoPE)."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool = True,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float)
                / rotary_dim
            )
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.register_buffer("cos_sin_cache", cache.to(dtype), persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        q_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        q_rot = query[..., : self.rotary_dim]
        q_pass = query[..., self.rotary_dim :]
        if self.is_neox_style:
            q_rot = apply_rotary_emb_neox(q_rot, cos, sin)
        else:
            raise NotImplementedError("GPT-J style RoPE 未实现")
        query = torch.cat((q_rot, q_pass), dim=-1).reshape(q_shape)

        k_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        k_rot = key[..., : self.rotary_dim]
        k_pass = key[..., self.rotary_dim :]
        if self.is_neox_style:
            k_rot = apply_rotary_emb_neox(k_rot, cos, sin)
        else:
            raise NotImplementedError("GPT-J style RoPE 未实现")
        key = torch.cat((k_rot, k_pass), dim=-1).reshape(k_shape)
        return query, key


class QKVRmsNormRopeBlock(nn.Module):
    """最小 QKV + QK-RMSNorm + RoPE 模块，结构对齐 Qwen3 / vLLM。"""

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int = 65536,
        rope_theta: float = 1_000_000.0,
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
        self.rotary_emb = RotaryEmbedding(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
        )

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

    def forward_rotary(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb(positions, q, k)

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.forward_qkv_proj(hidden)
        q, k, v = self.forward_qk_norm(qkv)
        q, k = self.forward_rotary(positions, q, k)
        return q, k, v


@torch.no_grad()
def bench_stages(
    model: QKVRmsNormRopeBlock,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        qkv = model.forward_qkv_proj(hidden)
        q, k, v = model.forward_qk_norm(qkv)
        model.forward_rotary(positions, q, k)
    sync()

    t_proj = t_qnorm = t_knorm = t_rope = t_full = 0.0

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
        _ = v
        t_knorm += ms_since(t0)

        t0 = time.perf_counter()
        q, k = model.forward_rotary(positions, q, k)
        t_rope += ms_since(t0)

        t0 = time.perf_counter()
        q, k, v = model.forward(positions, hidden)
        t_full += ms_since(t0)
        del q, k, v

    n = float(iters)
    return {
        "qkv_proj": t_proj / n,
        "q_norm": t_qnorm / n,
        "k_norm": t_knorm / n,
        "rotary_emb": t_rope / n,
        "full": t_full / n,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PyTorch QKV + QK-RMSNorm + RoPE benchmark"
    )
    p.add_argument("--tokens", type=int, default=128, help="seq len (batch×seq)")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--num-q-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument(
        "--rope-theta",
        type=float,
        default=1_000_000.0,
        help="rope_parameters['rope_theta']，default RoPE",
    )
    p.add_argument("--max-position", type=int, default=65536)
    p.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="与 lesson17 FP16 对齐时选 float16",
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
    model = QKVRmsNormRopeBlock(
        args.hidden,
        args.num_q_heads,
        args.num_kv_heads,
        args.head_dim,
        max_position_embeddings=args.max_position,
        rope_theta=args.rope_theta,
    ).to(device=device, dtype=dtype)
    model.eval()

    hidden = torch.randn(args.tokens, args.hidden, device=device, dtype=dtype)
    positions = torch.arange(args.tokens, device=device, dtype=torch.long)

    with torch.no_grad():
        q, k, v = model.forward(positions, hidden)
    assert q.shape == (args.tokens, q_size)
    assert k.shape == (args.tokens, kv_size)
    assert v.shape == (args.tokens, kv_size)

    times = bench_stages(model, hidden, positions, args.warmup, args.iters)

    gemm_flops = 2.0 * args.tokens * args.hidden * qkv_dim
    t_proj = times["qkv_proj"]
    t_q = times["q_norm"]
    t_k = times["k_norm"]
    t_rope = times["rotary_emb"]
    t_full = times["full"]
    t_norm = t_q + t_k

    props = torch.cuda.get_device_properties(device)
    print(f"PyTorch QKV + QK-RMSNorm + RoPE  —  {props.name}")
    print(f"hidden [{args.tokens}×{args.hidden}] → qkv [{args.tokens}×{qkv_dim}]")
    print(
        f"  q: {args.num_q_heads} heads × {args.head_dim}  |  "
        f"k/v: {args.num_kv_heads} heads × {args.head_dim}  |  "
        f"dtype={args.dtype}  |  rope_theta={args.rope_theta:g}"
    )
    print()
    print("── 对齐 PyTorch 分段耗时 ──")
    print("  split(q,k,v) / view  : ~0 ms  (无 kernel)")
    print(f"  ① qkv_proj (Linear)  : {t_proj:8.3f} ms  ({100*t_proj/t_full:5.1f}%)")
    print(f"  ② q_norm  (RMSNorm)  : {t_q:8.3f} ms  ({100*t_q/t_full:5.1f}%)")
    print(f"  ③ k_norm  (RMSNorm)  : {t_k:8.3f} ms  ({100*t_k/t_full:5.1f}%)")
    print(f"  ④ rotary_emb (RoPE)  : {t_rope:8.3f} ms  ({100*t_rope/t_full:5.1f}%)")
    print(f"  分段合计 ①+②+③+④    : {t_proj+t_norm+t_rope:8.3f} ms")
    print(f"  整段 forward (实测)  : {t_full:8.3f} ms")
    print()
    print(f"  GEMM TFLOPS ≈ {gemm_flops / (t_proj * 1e6):.1f}")
    print(f"  qk-norm 占整段 ≈ {100 * t_norm / t_full:.1f}%")
    print(f"  RoPE 占整段 ≈ {100 * t_rope / t_full:.1f}%")
    print()
    print("对比 CUDA lesson17 (Release):")
    print(f"  ./build-release/bin/lesson17_qkv_rmsnorm {args.tokens}")


if __name__ == "__main__":
    main()
