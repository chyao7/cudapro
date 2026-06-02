/*
 * QKV Projection + QK-RMSNorm (CUTLASS API)
 *
 * 对应 PyTorch 逻辑:
 *   qkv = hidden @ W_qkv
 *   q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
 *   q = q_norm(q.view(..., num_q_heads, head_dim))
 *   k = k_norm(k.view(..., num_kv_heads, head_dim))
 *
 * 内存布局 (row-major): view/reshape 不改变数据，q/k 区段本身已连续
 *   q: [tokens, q_size]  ≡ [tokens*num_q_heads, head_dim]
 *   k: [tokens, kv_size] ≡ [tokens*num_kv_heads, head_dim]
 *
 * 编译:
 *   nvcc -O3 -arch=sm_86 \
 *     -I/path/to/cutlass/include \
 *     -I/path/to/cutlass/tools/util/include \
 *     -o lesson17 lesson17_qkv_rmsnorm.cu
 *
 * 运行: ./lesson17
 *       ./lesson17 4096        # 指定 seq len (batch×seq)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/half.h"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define CUTLASS_CHECK(status)                                                \
    do {                                                                     \
        cutlass::Status st = (status);                                       \
        if (st != cutlass::Status::kSuccess) {                               \
            fprintf(stderr, "CUTLASS error at %s:%d: %d\n", __FILE__,        \
                    __LINE__, (int)st);                                      \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ── 模型尺寸 (GQA)，tokens 可由命令行覆盖 ──
static int g_tokens = 128;
#define TOKENS g_tokens
#define HIDDEN 512
#define NUM_Q_HEADS 8
#define NUM_KV_HEADS 4
#define HEAD_DIM 64

#define Q_SIZE (NUM_Q_HEADS * HEAD_DIM)
#define KV_SIZE (NUM_KV_HEADS * HEAD_DIM)
#define QKV_DIM (Q_SIZE + KV_SIZE + KV_SIZE)
#define EPS 1e-5f

using Element = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;
using ElementAcc = cutlass::half_t;

using CutlassGemm = cutlass::gemm::device::Gemm<
    Element, Layout, Element, Layout, Element, Layout, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        Element, 128 / cutlass::sizeof_bits<Element>::value, ElementAcc,
        ElementAcc>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

static float to_f(Element x) { return static_cast<float>(x); }

static Element to_h(float x) { return Element(x); }

__device__ float elem_to_f(Element x) { return static_cast<float>(x); }

// qkv 缓冲区内 q/k 跨 token 不连续（行间 stride = QKV_DIM）。
// 一次 launch：grid = T×N_heads，每 block 处理一个 (token, head) 行。
__global__ void rmsnorm_qkv_heads_strided(
    Element *qkv, int tokens, int qkv_dim, int col_offset, int num_heads,
    int head_dim, const Element *weight, float eps) {
    int const idx = blockIdx.x;
    int const t = idx / num_heads;
    int const h = idx % num_heads;
    if (t >= tokens) {
        return;
    }

    Element *row = qkv + t * qkv_dim + col_offset + h * head_dim;
    int const tid = threadIdx.x;
    extern __shared__ float smem_reduce[];
    float local_sq = 0.0f;

    for (int j = tid; j < head_dim; j += blockDim.x) {
        float v = elem_to_f(row[j]);
        local_sq += v * v;
    }

    smem_reduce[tid] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_reduce[tid] += smem_reduce[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(smem_reduce[0] / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();

    for (int j = tid; j < head_dim; j += blockDim.x) {
        row[j] = Element(elem_to_f(row[j]) * s_inv_rms * elem_to_f(weight[j]));
    }
}

static void launch_rmsnorm_qkv_heads(Element *d_qkv, int col_offset, int num_heads,
                                     const Element *d_weight,
                                     cudaStream_t stream) {
    int const rows = TOKENS * num_heads;
    int const block = HEAD_DIM;
    size_t const smem = static_cast<size_t>(block) * sizeof(float);
    rmsnorm_qkv_heads_strided<<<rows, block, smem, stream>>>(
        d_qkv, TOKENS, QKV_DIM, col_offset, num_heads, HEAD_DIM, d_weight, EPS);
    CUDA_CHECK(cudaGetLastError());
}

// ── CPU 参考 ──
static void gemm_cpu(const Element *a, const Element *b, Element *c, int m,
                     int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int t = 0; t < k; ++t) {
                sum += (double)to_f(a[i * k + t]) * to_f(b[t * n + j]);
            }
            c[i * n + j] = to_h((float)sum);
        }
    }
}

static void rmsnorm_cpu_rows(const Element *in, const Element *weight,
                             Element *out, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const Element *row_in = in + r * cols;
        Element *row_out = out + r * cols;
        double sum_sq = 0.0;
        for (int j = 0; j < cols; ++j) {
            float v = to_f(row_in[j]);
            sum_sq += (double)v * v;
        }
        float inv_rms = rsqrtf((float)(sum_sq / cols) + eps);
        for (int j = 0; j < cols; ++j) {
            row_out[j] =
                to_h(to_f(row_in[j]) * inv_rms * to_f(weight[j]));
        }
    }
}

static void qkv_rmsnorm_cpu(const Element *hidden, const Element *W,
                            const Element *q_weight, const Element *k_weight,
                            Element *qkv, Element *v_out) {
    gemm_cpu(hidden, W, qkv, TOKENS, QKV_DIM, HIDDEN);

    for (int t = 0; t < TOKENS; ++t) {
        Element *row = qkv + t * QKV_DIM;
        rmsnorm_cpu_rows(row, q_weight, row, NUM_Q_HEADS, HEAD_DIM, EPS);
        rmsnorm_cpu_rows(row + Q_SIZE, k_weight, row + Q_SIZE, NUM_KV_HEADS,
                         HEAD_DIM, EPS);
    }

    const Element *v = qkv + Q_SIZE + KV_SIZE;

    const int v_elems = TOKENS * KV_SIZE;
    for (int i = 0; i < v_elems; ++i) {
        v_out[i] = v[i];
    }
}

// ── CUTLASS QKV + QK-RMSNorm ──
struct QkvRmsNormOp {
    Element *d_hidden{};
    Element *d_W{};
    Element *d_qkv{};
    Element *d_q_weight{};
    Element *d_k_weight{};

    CutlassGemm gemm;

    void qkv_proj(cudaStream_t stream = nullptr) {
        typename CutlassGemm::Arguments args(
            {TOKENS, QKV_DIM, HIDDEN},
            {d_hidden, HIDDEN},
            {d_W, QKV_DIM},
            {d_qkv, QKV_DIM},
            {d_qkv, QKV_DIM},
            {ElementAcc(1.0f), ElementAcc(0.0f)});
        CUTLASS_CHECK(gemm(args));
    }

    // 批量 RMSNorm：2 launch（q + k），对齐 PyTorch view(T, N_heads, d)
    void q_norm(cudaStream_t stream = nullptr) {
        launch_rmsnorm_qkv_heads(d_qkv, 0, NUM_Q_HEADS, d_q_weight, stream);
    }

    void k_norm(cudaStream_t stream = nullptr) {
        launch_rmsnorm_qkv_heads(d_qkv, Q_SIZE, NUM_KV_HEADS, d_k_weight,
                                 stream);
    }

    void run(cudaStream_t stream = nullptr) {
        qkv_proj(stream);
        q_norm(stream);
        k_norm(stream);
    }
};

static bool verify(const Element *gpu, const Element *ref, int n, float tol) {
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_err = fmaxf(max_err, fabsf(to_f(gpu[i]) - to_f(ref[i])));
    }
    printf("  max_err = %.4f %s\n", max_err, max_err < tol ? "(OK)" : "(FAIL)");
    return max_err < tol;
}

static float bench_stage(void (*fn)(QkvRmsNormOp &, cudaStream_t), QkvRmsNormOp &op,
                         int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) {
        fn(op, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        fn(op, nullptr);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

static void stage_qkv_proj(QkvRmsNormOp &op, cudaStream_t s) { op.qkv_proj(s); }
static void stage_q_norm(QkvRmsNormOp &op, cudaStream_t s) { op.q_norm(s); }
static void stage_k_norm(QkvRmsNormOp &op, cudaStream_t s) { op.k_norm(s); }
static void stage_full(QkvRmsNormOp &op, cudaStream_t s) { op.run(s); }

static float bench(QkvRmsNormOp &op, int warmup, int repeats) {
    return bench_stage(stage_full, op, warmup, repeats);
}

int main(int argc, char **argv) {
    if (argc > 1) {
        g_tokens = atoi(argv[1]);
        if (g_tokens <= 0) {
            fprintf(stderr, "用法: %s [tokens]\n", argv[0]);
            return 1;
        }
    }
    cudaDeviceProp prop; // 获取设备属性
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 8) {
        fprintf(stderr, "需要 Ampere+ (sm_80+) Tensor Core\n");
        return 1;
    }

    printf("QKV + QK-RMSNorm (CUTLASS)  —  %s\n\n", prop.name);
    printf("hidden [%d × %d]  @  W [%d × %d]  →  qkv [%d × %d]\n", TOKENS,
           HIDDEN, HIDDEN, QKV_DIM, TOKENS, QKV_DIM);
    printf("  q: %d heads × %d  |  k/v: %d heads × %d each\n\n", NUM_Q_HEADS,
           HEAD_DIM, NUM_KV_HEADS, HEAD_DIM);

    const size_t bytes_hidden = (size_t)TOKENS * HIDDEN * sizeof(Element);
    const size_t bytes_W = (size_t)HIDDEN * QKV_DIM * sizeof(Element);
    const size_t bytes_qkv = (size_t)TOKENS * QKV_DIM * sizeof(Element);
    const size_t bytes_v = (size_t)TOKENS * KV_SIZE * sizeof(Element);
    const size_t bytes_qw = (size_t)HEAD_DIM * sizeof(Element);
    const size_t bytes_kw = (size_t)HEAD_DIM * sizeof(Element);

    Element *h_hidden = (Element *)malloc(bytes_hidden);
    Element *h_W = (Element *)malloc(bytes_W);
    Element *h_qkv = (Element *)malloc(bytes_qkv);
    Element *h_qkv_ref = (Element *)malloc(bytes_qkv);
    Element *h_v = (Element *)malloc(bytes_v);
    Element *h_v_ref = (Element *)malloc(bytes_v);
    Element *h_q_weight = (Element *)malloc(bytes_qw);
    Element *h_k_weight = (Element *)malloc(bytes_kw);

    for (int i = 0; i < TOKENS * HIDDEN; ++i) {
        h_hidden[i] = to_h((float)(i % 17) * 0.01f);
    }
    for (int i = 0; i < HIDDEN * QKV_DIM; ++i) {
        h_W[i] = to_h((float)(i % 13) * 0.01f);
    }
    for (int i = 0; i < HEAD_DIM; ++i) {
        h_q_weight[i] = to_h(1.0f + 0.01f * (float)(i % 7));
        h_k_weight[i] = to_h(1.0f + 0.01f * (float)(i % 5));
    }

    qkv_rmsnorm_cpu(h_hidden, h_W, h_q_weight, h_k_weight, h_qkv_ref, h_v_ref);

    Element *d_hidden, *d_W, *d_qkv;
    Element *d_q_weight, *d_k_weight;
    CUDA_CHECK(cudaMalloc(&d_hidden, bytes_hidden));
    CUDA_CHECK(cudaMalloc(&d_W, bytes_W));
    CUDA_CHECK(cudaMalloc(&d_qkv, bytes_qkv));
    CUDA_CHECK(cudaMalloc(&d_q_weight, bytes_qw));
    CUDA_CHECK(cudaMalloc(&d_k_weight, bytes_kw));
    CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden, bytes_hidden, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W, bytes_W, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_weight, h_q_weight, bytes_qw, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_weight, h_k_weight, bytes_kw, cudaMemcpyHostToDevice));

    QkvRmsNormOp op{d_hidden, d_W, d_qkv, d_q_weight, d_k_weight};
    op.run();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_qkv, d_qkv, bytes_qkv, cudaMemcpyDeviceToHost));

    const Element *h_q = h_qkv;
    const Element *h_k = h_qkv + Q_SIZE;
    const Element *h_v_slice = h_qkv + Q_SIZE + KV_SIZE;
    const Element *h_q_ref = h_qkv_ref;
    const Element *h_k_ref = h_qkv_ref + Q_SIZE;

    printf("验证 q (RMSNorm): %s\n",
           verify(h_q, h_q_ref, TOKENS * Q_SIZE, 0.05f) ? "通过" : "失败");
    printf("验证 k (RMSNorm): %s\n",
           verify(h_k, h_k_ref, TOKENS * KV_SIZE, 0.05f) ? "通过" : "失败");
    printf("验证 v (无 norm): %s\n",
           verify(h_v_slice, h_v_ref, TOKENS * KV_SIZE, 0.05f) ? "通过" : "失败");

    const float t_proj = bench_stage(stage_qkv_proj, op, 5, 50);
    const float t_qnorm = bench_stage(stage_q_norm, op, 5, 50);
    const float t_knorm = bench_stage(stage_k_norm, op, 5, 50);
    const float t_full = bench_stage(stage_full, op, 5, 50);
    const double gemm_flops = 2.0 * TOKENS * HIDDEN * QKV_DIM;

    printf("\n── 对齐 PyTorch 分段耗时 (批量 RMSNorm, 2 launch) ──\n");
    printf("  split(q,k,v)     : 0 ms  (view，无 kernel)\n");
    printf("%28s  %8.3f ms  %5.1f%%\n", "① qkv_proj (GEMM)", t_proj,
           100.f * t_proj / t_full);
    printf("%28s  %8.3f ms  %5.1f%%\n", "② q_norm (RMSNorm)", t_qnorm,
           100.f * t_qnorm / t_full);
    printf("%28s  %8.3f ms  %5.1f%%\n", "③ k_norm (RMSNorm)", t_knorm,
           100.f * t_knorm / t_full);
    printf("%28s  %8.3f ms\n", "合计 (①+②+③)", t_full);
    printf("  GEMM TFLOPS ≈ %.1f  |  qk-norm 占 %.1f%%\n",
           gemm_flops / (t_proj * 1e6), 100.f * (t_qnorm + t_knorm) / t_full);
    printf("\n提示: 与 lesson18 融合版对比请运行 ./lesson18\n");

    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_q_weight));
    CUDA_CHECK(cudaFree(d_k_weight));
    free(h_hidden);
    free(h_W);
    free(h_qkv);
    free(h_qkv_ref);
    free(h_v);
    free(h_v_ref);
    free(h_q_weight);
    free(h_k_weight);
    return 0;
}
