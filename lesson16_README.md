# Lesson 16 — Tensor Core GEMM (WMMA)

对比三种矩阵乘法实现：**FP32 tiled（CUDA Core）**、**手写 WMMA（Tensor Core）**、**cuBLAS Tensor Op**。

源文件：`lesson16_tensor_gemm.cu`

> 公式使用 `$...$`（行内）和 `$$...$$`（独立一行）。Cursor / VS Code 预览需开启 Markdown 数学公式。

---

## 1. 问题定义

标准 GEMM（row-major）：

$$
\mathbf{C}[M \times N] = \mathbf{A}[M \times K] \times \mathbf{B}[K \times N]
$$

每个输出元素：

$$
C_{ij} = \sum_{t=0}^{K-1} A_{it} \cdot B_{tj}, \quad 0 \le i < M,\; 0 \le j < N
$$

线性地址（row-major）：

$$
A_{ij} \rightarrow A[i \cdot K + j], \quad
B_{ij} \rightarrow B[i \cdot N + j], \quad
C_{ij} \rightarrow C[i \cdot N + j]
$$

FLOPs（乘加各算 1 次）：

$$
\mathrm{FLOPs} = 2MNK
$$

$$
\mathrm{GFLOPS} = \frac{2MNK}{t_{\mathrm{ms}} \times 10^{6}}
$$

---

## 2. 三种实现

| 版本 | 硬件 | A / B 类型 | 累加 / C 类型 | 说明 |
|------|------|------------|---------------|------|
| FP32 tiled | CUDA Core | `float` | `float` | 同 lesson03 |
| WMMA | Tensor Core | `half` (FP16) | `float` | 手写 `mma.sync` |
| cuBLAS Tensor Op | Tensor Core | `half` | `float` | `cublasGemmEx` |

Tensor 路径的数值含义：

$$
C_{ij} = \sum_{t=0}^{K-1} \mathrm{FP32}\!\left(\mathrm{FP16}(A_{it}) \cdot \mathrm{FP16}(B_{tj})\right)
$$

即 **FP16 输入、FP32 累加**，与训练里常用的 mixed-precision GEMM 一致。

---

## 3. FP32 Tiled（CUDA Core）

与 lesson03 相同：每个 thread 算 $C$ 的一个元素，$K$ 维分 tile 载入 shared memory。

**Block**：`16×16 = 256` threads，每个 thread 负责输出 1 个元素。

**K 维循环**（第 $t$ 个 tile）：

$$
\text{sum} \mathrel{+}= sA[\mathrm{ty}][i] \cdot sB[i][\mathrm{tx}], \quad i = 0..15
$$

**Launch**：

```text
grid  = ( ⌈N/16⌉, ⌈M/16⌉ )
block = ( 16, 16 )
```

---

## 4. WMMA Tensor Core

### 4.1 基本 MMA 块

Volta+ 上 WMMA 固定一次处理 **16×16×16**：

$$
\mathbf{D}_{16 \times 16} = \mathbf{A}_{16 \times 16} \times \mathbf{B}_{16 \times 16} + \mathbf{C}_{16 \times 16}
$$

本 lesson 中 $\mathbf{A},\mathbf{B}$ 为 FP16，累加器 $\mathbf{C},\mathbf{D}$ 为 FP32。

对应 PTX 指令：`mma.sync.aligned.m16n8k16`（Ampere 上常见 16×8×16 tile；API 层仍用 16×16×16 fragment）。

### 4.2 K 维外循环

全局 $K$ 按 16 分块累加：

$$
\mathbf{C}_{\mathrm{tile}} \mathrel{+}= \mathbf{A}_{\mathrm{tile}}^{(i)} \times \mathbf{B}_{\mathrm{tile}}^{(i)}, \quad i = 0, 16, 32, \ldots, K-16
$$

### 4.3 线程组织

```text
1 block = 1 warp（32 threads）
1 block 负责 C 上 1 个 16×16 tile

grid = ( N/16, M/16 )
block = ( 32 )
```

### 4.4 Shape 变换

```text
Global C[M×N]
    ↓ block (tile_m, tile_n)
C[row : row+16, col : col+16]     row = tile_m×16, col = tile_n×16
    ↓ K 循环，每次 i += 16
A[row : row+16, i : i+16]  ──load──→  a_frag [16×16]
B[i : i+16, col : col+16]  ──load──→  b_frag [16×16]
    ↓ mma_sync
acc [16×16] (FP32)
    ↓ store
写回 C[row : row+16, col : col+16]
```

---

## 5. cuBLAS Tensor Op

row-major 下调用（与 lesson15 列主序约定一致）：

$$
\mathbf{C} = \alpha \mathbf{A}\mathbf{B} + \beta \mathbf{C}, \quad \alpha=1,\; \beta=0
$$

API：`cublasGemmEx`，`CUDA_R_16F` 输入，`CUDA_R_32F` 输出，`CUBLAS_GEMM_DEFAULT_TENSOR_OP`。

内部仍走 Tensor Core，并含多 stage pipeline、tile 自动选择等优化，通常快于手写 WMMA。

---

## 6. 编译与运行

```bash
nvcc -O3 -arch=sm_86 -o lesson16 lesson16_tensor_gemm.cu -lcublas

./lesson16          # 默认 1024³
./lesson16 2048     # 自定义 M=N=K（须为 16 的倍数）
```

要求：**Volta+（sm_70+）**，推荐 Ampere（sm_80+）。`M, N, K` 必须是 **16 的倍数**。

---

## 7. 验证

| 版本 | 参考 | 容差 |
|------|------|------|
| FP32 tiled | CPU FP64 累加 | 1e-2 |
| WMMA / cuBLAS | CPU FP32 GEMM | 2.0（FP16 舍入） |

---

## 8. 参考性能（RTX 3080 Ti, sm_86）

| 版本 | 1024³ ms | 1024³ GFLOPS | vs FP32 |
|------|----------|--------------|---------|
| FP32 tiled | 0.92 | 2330 | 1.0× |
| WMMA | 0.15 | 14141 | **6.1×** |
| cuBLAS Tensor | 0.05 | 45789 | **19.7×** |

| 版本 | 2048³ ms | 2048³ GFLOPS | vs FP32 |
|------|----------|--------------|---------|
| FP32 tiled | 6.09 | 2821 | 1.0× |
| WMMA | 1.06 | 16192 | **5.7×** |
| cuBLAS Tensor | 0.27 | 64392 | **22.8×** |

---

## 9. 与前后 lesson 的关系

```text
lesson03  FP32 tiled GEMM        ← CUDA Core 基线
lesson15  FP32 tiled vs cuBLAS   ← 库优化能快多少
lesson16  FP32 vs WMMA vs cuBLAS Tensor  ← Tensor Core 引入
```

**CUDA Core vs Tensor Core**：

| | CUDA Core (lesson03/16 FP32) | Tensor Core (lesson16 WMMA) |
|--|------------------------------|-----------------------------|
| 指令 | 每 thread 一次 FMA | 一条 `mma.sync` 算 16×16×16 |
| 精度 | FP32 | FP16 入、FP32 累加 |
| block | 256 threads / 16×16 输出 | 32 threads (1 warp) / 16×16 输出 |

手写 WMMA 比 FP32 快约 **6×**，但距 cuBLAS 仍有约 **3×** 差距——后者有多 warp block、double buffering、自动调参等完整优化。

---

## 10. 关键 API

```cpp
#include <mma.h>   // nvcuda::wmma

nvcuda::wmma::fragment<..., half, row_major> a_frag, b_frag;
nvcuda::wmma::fragment<..., float> acc;

nvcuda::wmma::load_matrix_sync(a_frag, ptr_A, lda);
nvcuda::wmma::load_matrix_sync(b_frag, ptr_B, ldb);
nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
nvcuda::wmma::store_matrix_sync(ptr_C, acc, ldc, mem_row_major);
```

命名空间是 **`nvcuda::wmma`**，不是裸 `wmma::`。
