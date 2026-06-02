# Lesson 17 — QKV Projection + QK-RMSNorm (CUTLASS)

使用 CUTLASS API 实现 Attention 输入侧的 **QKV 线性投影** 与 **Q/K 的 per-head RMSNorm**。

源文件：`src/lesson17_qkv_rmsnorm.cu`

> 公式使用 `$...$`（行内）和 `$$...$$`（独立一行）。Cursor / VS Code 预览需开启 Markdown 数学公式；也可直接看下方「纯文本公式」小节。

---

## 1. 对应 PyTorch 逻辑

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
q_by_head = self.q_norm(q_by_head)
q = q_by_head.view(q.shape)

k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
k_by_head = self.k_norm(k_by_head)
k = k_by_head.view(k.shape)
```

GPU 上三步有 kernel：`qkv_proj` → `q_norm` → `k_norm`；`split` / `view` 仅为指针偏移，无计算。

---

## 2. 默认模型尺寸（GQA）

| 符号 | 默认值 | 含义 |
|------|--------|------|
| $T$ | 128（可命令行改） | token 数（batch × seq） |
| $H$ | 512 | hidden size |
| $N_q$ | 8 | Query head 数 |
| $N_{kv}$ | 4 | Key/Value head 数 |
| $d$ | 64 | head dim |

派生维度：

$$
q\_size = N_q \cdot d = 512
$$

$$
kv\_size = N_{kv} \cdot d = 256
$$

$$
qkv\_dim = q\_size + 2 \cdot kv\_size = 1024
$$

纯文本：

```
q_size  = N_q  × d = 512
kv_size = N_kv × d = 256
qkv_dim = q_size + 2 × kv_size = 1024
```

---

## 3. QKV 投影（GEMM）

$$
\mathbf{QKV}[T \times qkv\_dim] = \mathbf{H}[T \times H] \times \mathbf{W}[H \times qkv\_dim]
$$

每个输出元素：

$$
\mathrm{QKV}_{t,j} = \sum_{k=0}^{H-1} H_{t,k} \cdot W_{k,j}
$$

FLOPs：

$$
\mathrm{FLOPs}_{proj} = 2 \cdot T \cdot H \cdot qkv\_dim
$$

纯文本：

```
QKV[t, j] = sum_k  H[t, k] * W[k, j]
FLOPs_proj  = 2 × T × H × qkv_dim
```

实现：`cutlass::gemm::device::Gemm`，FP16 Tensor Core，row-major。

$$
\mathbf{C} = \alpha \mathbf{A}\mathbf{B} + \beta \mathbf{C}, \quad \alpha=1,\; \beta=0
$$

---

## 4. Split（列切分）

row-major 下 `qkv` 一行内按列连续存放：

$$
\underbrace{\mathbf{q}_t}_{q\_size}
\underbrace{\mathbf{k}_t}_{kv\_size}
\underbrace{\mathbf{v}_t}_{kv\_size}
\quad \leftarrow \text{第 } t \text{ 个 token 的 } qkv\_dim \text{ 列}
$$

指针偏移（无数据搬运）：

$$
\begin{aligned}
\mathbf{q} &\leftarrow \mathrm{QKV} + 0 \\
\mathbf{k} &\leftarrow \mathrm{QKV} + q\_size \\
\mathbf{v} &\leftarrow \mathrm{QKV} + q\_size + kv\_size
\end{aligned}
$$

纯文本：

```
q ← QKV + 0
k ← QKV + q_size
v ← QKV + q_size + kv_size
```

---

## 5. View / Reshape（按 head 展开）

对第 $t$ 个 token，逻辑 shape 变换（内存不变）：

$$
\mathbf{q}_t \in \mathbb{R}^{q\_size} \;\equiv\; \mathbf{q}_t \in \mathbb{R}^{N_q \times d}
$$

$$
\mathbf{k}_t \in \mathbb{R}^{kv\_size} \;\equiv\; \mathbf{k}_t \in \mathbb{R}^{N_{kv} \times d}
$$

线性地址（第 $t$ 个 token，第 $h$ 个 head，第 $j$ 维）：

$$
\mathrm{addr}(q_{t,h,j}) = t \cdot qkv\_dim + h \cdot d + j
$$

$$
\mathrm{addr}(k_{t,h,j}) = t \cdot qkv\_dim + q\_size + h \cdot d + j
$$

纯文本：

```
addr(q[t,h,j]) = t * qkv_dim + h * d + j
addr(k[t,h,j]) = t * qkv_dim + q_size + h * d + j
```

---

## 6. RMSNorm（QK-Norm）

对每个 $(t, h)$，在 head 维度 $d$ 上做 RMSNorm（Llama / Qwen 风格）：

$$
\mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{j=0}^{d-1} x_j^2 + \varepsilon}
$$

$$
\hat{x}_j = \frac{x_j}{\mathrm{RMS}(x)} \cdot \gamma_j
$$

其中 $\gamma \in \mathbb{R}^d$ 为可学习权重（`q_weight` / `k_weight`），$\varepsilon = 10^{-5}$。

对 Q：

$$
\forall\, t \in [0,T),\; h \in [0,N_q):\quad
\hat{\mathbf{q}}_{t,h,:} = \mathrm{RMSNorm}(\mathbf{q}_{t,h,:},\, \gamma_q)
$$

对 K：

$$
\forall\, t \in [0,T),\; h \in [0,N_{kv}):\quad
\hat{\mathbf{k}}_{t,h,:} = \mathrm{RMSNorm}(\mathbf{k}_{t,h,:},\, \gamma_k)
$$

纯文本：

```
RMS(x)   = sqrt( (1/d) * sum_j(x_j^2) + eps )
x_hat_j  = x_j / RMS(x) * gamma_j

对 Q：每个 (t, h)，在 d 维上做 RMSNorm，权重 gamma_q
对 K：每个 (t, h)，在 d 维上做 RMSNorm，权重 gamma_k
V 不做 RMSNorm
```

**V 不做 RMSNorm**，GEMM 输出后保持不变。

实现：自定义 strided kernel，一次 launch 覆盖 $T \times N_q$（或 $T \times N_{kv}$）个 $(token, head)$ 行，在 `qkv` 缓冲区内 in-place 归一化。

---

## 7. 计算流程

```text
hidden [T×H]
    │
    ▼  ① CUTLASS GEMM (Tensor Core, FP16)
qkv [T×qkv_dim]
    │
    ├─ q [T×q_size]      ──► ② q_norm (RMSNorm, per token × per head)
    ├─ k [T×kv_size]      ──► ③ k_norm (RMSNorm, per token × per head)
    └─ v [T×kv_size]      ──► 不处理
```

Launch 次数：$1$ 次 GEMM + $1$ 次 q RMSNorm + $1$ 次 k RMSNorm（共 3 launch）。

q/k 仍在 `qkv` 缓冲区内 in-place；因行间 stride 为 `qkv_dim`，用 strided kernel：grid = $T \times N_{heads}$，每 block 一个 $(token, head)$。

---

## 8. CUTLASS API

**GEMM**

```cpp
cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,  // A, layout
    cutlass::half_t, cutlass::layout::RowMajor,  // B
    cutlass::half_t, cutlass::layout::RowMajor,  // C/D
    cutlass::half_t,                             // accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, ...>;

gemm({{T, qkv_dim, H}, {d_hidden, H}, {d_W, qkv_dim}, ...});
```

**RMSNorm（strided 批量）**

```cpp
// q: grid = T * NUM_Q_HEADS, 每 block 一行 (token, head)
rmsnorm_qkv_heads_strided<<<T * NUM_Q_HEADS, HEAD_DIM>>>(
    d_qkv, T, QKV_DIM, /*col_offset=*/0, NUM_Q_HEADS, HEAD_DIM,
    d_q_weight, EPS);

// k: col_offset = Q_SIZE, grid = T * NUM_KV_HEADS
rmsnorm_qkv_heads_strided<<<T * NUM_KV_HEADS, HEAD_DIM>>>(
    d_qkv, T, QKV_DIM, Q_SIZE, NUM_KV_HEADS, HEAD_DIM,
    d_k_weight, EPS);
```

头文件：`cutlass/gemm/device/gemm.h`

---

## 9. 编译与运行

需本地 CUTLASS 源码（include 路径）。

```bash
./scripts/configure.sh release
./scripts/build.sh release lesson17_qkv_rmsnorm

./build-release/bin/lesson17_qkv_rmsnorm           # 默认 T=128
./build-release/bin/lesson17_qkv_rmsnorm 4096      # 指定 token 数
```

要求：**Ampere+（sm_80+）** Tensor Core。

---

## 10. 验证

CPU 参考：FP64 累加 GEMM + 逐 token 逐 head RMSNorm。

| 输出 | 容差 |
|------|------|
| q（RMSNorm 后） | 0.05 |
| k（RMSNorm 后） | 0.05 |
| v（无 norm） | 0.05 |

---

## 11. 分段 Benchmark 输出

程序会分别计时：

| 阶段 | PyTorch 对应 |
|------|--------------|
| ① qkv_proj | `self.qkv_proj(hidden_states)` |
| ② q_norm | `self.q_norm(q.view(...))` |
| ③ k_norm | `self.k_norm(k.view(...))` |
| 合计 | 整段 forward |

并输出 GEMM TFLOPS 与 qk-norm 耗时占比。

---

## 12. 内存布局示意

```text
qkv 第 t 行 (长度 qkv_dim=1024):

|←—— q: 8×64=512 ——→|←— k: 4×64=256 —→|←— v: 4×64=256 —→|
  head0 ... head7      head0 ... head3     head0 ... head3
```

row-major 下 q/k 各 head 的 $d$ 维连续，可直接 `view` 为 $[N_q, d]$ 或 $[N_{kv}, d]$ 做 RMSNorm。
