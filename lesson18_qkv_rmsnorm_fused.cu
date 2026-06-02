/*
 * QKV + QK-RMSNorm — 融合 Epilogue 版 (lesson18)
 *
 * 对比 lesson17 (GEMM + 2× cutlass::rmsnorm):
 *   ① CUTLASS GEMM + EpilogueVisitor: 写 qkv 同时 atomic 累加 q/k 的 sum(x²)
 *   ② finalize kernel: rsqrt + weight，in-place 归一化 q/k
 *
 * 参考: cutlass/examples/37_gemm_layernorm_gemm_fusion
 *
 * 编译:
 *   nvcc -O3 -arch=sm_86 --expt-relaxed-constexpr \
 *     -I/home/chyao/projects/cutlass/include \
 *     -I/home/chyao/projects/cutlass/tools/util/include \
 *     -I/home/chyao/projects/cutlass/examples/37_gemm_layernorm_gemm_fusion \
 *     -o lesson18 lesson18_qkv_rmsnorm_fused.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/util/device_rmsnorm.h"
#include "cutlass/half.h"

#include "gemm_with_epilogue_visitor.h"
#include "qk_rmsnorm_epilogue_visitor.h"

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
using ElementCompute = cutlass::half_t;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
    Element, 128 / cutlass::sizeof_bits<Element>::value, ElementCompute,
    ElementCompute>;

using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
    Element, Layout, 128 / cutlass::sizeof_bits<Element>::value, Element,
    Layout, 128 / cutlass::sizeof_bits<Element>::value, Element, Layout,
    ElementCompute, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape, EpilogueFunctorOp,
    SwizzleThreadBlock, 4, true,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Element, Element,
        Element, ElementCompute>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone>::GemmKernel;

using EpilogueVisitor = cutlass::kernel::EpilogueVisitorQkRmsNorm<
    ThreadblockShape, DefaultGemmKernel::kThreadCount,
    typename DefaultGemmKernel::Epilogue::OutputTileIterator,
    typename DefaultGemmKernel::Epilogue::AccumulatorFragmentIterator::
        AccumulatorTile,
    ElementCompute, ElementCompute, EpilogueFunctorOp>;

using Epilogue =
    typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        EpilogueVisitor, typename DefaultGemmKernel::Epilogue>::Epilogue;

using GemmEpilogueFusion = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
    typename DefaultGemmKernel::Mma, Epilogue, SwizzleThreadBlock>;

static __host__ __device__ float to_f(Element x) {
    return static_cast<float>(x);
}

static __host__ __device__ Element to_h(float x) { return Element(x); }

// ── 分离版 pass1: 从 qkv 计算 sum(x²) ──
__global__ void compute_sum_sq_qk(const Element *qkv, int tokens, int qkv_dim,
                                  int q_size, int kv_size, int head_dim,
                                  int num_q_heads, int num_kv_heads,
                                  float *sum_sq_q, float *sum_sq_k) {
    int const total_q_rows = tokens * num_q_heads;
    int const total_k_rows = tokens * num_kv_heads;
    int const idx = blockIdx.x;

    if (idx < total_q_rows) {
        int const token = idx / num_q_heads;
        int const head = idx % num_q_heads;
        const Element *row = qkv + token * qkv_dim + head * head_dim;
        float local = 0.f;
        for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
            float v = to_f(row[j]);
            local += v * v;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            local += __shfl_down_sync(0xffffffff, local, offset);
        }
        if (threadIdx.x == 0) {
            sum_sq_q[idx] = local;
        }
    } else {
        int const k_idx = idx - total_q_rows;
        if (k_idx >= total_k_rows) {
            return;
        }
        int const token = k_idx / num_kv_heads;
        int const head = k_idx % num_kv_heads;
        const Element *row = qkv + token * qkv_dim + q_size + head * head_dim;
        float local = 0.f;
        for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
            float v = to_f(row[j]);
            local += v * v;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            local += __shfl_down_sync(0xffffffff, local, offset);
        }
        if (threadIdx.x == 0) {
            sum_sq_k[k_idx] = local;
        }
    }
}

// ── finalize: 读 sum_sq，in-place 归一化 q/k ──
__global__ void apply_qk_rmsnorm_finalize(
    Element *qkv, int tokens, int qkv_dim, int q_size, int kv_size, int head_dim,
    int num_q_heads, int num_kv_heads, const float *sum_sq_q,
    const float *sum_sq_k, const Element *q_weight, const Element *k_weight,
    float eps) {
    int const total_q_rows = tokens * num_q_heads;
    int const total_k_rows = tokens * num_kv_heads;
    int const idx = blockIdx.x;

    if (idx < total_q_rows) {
        int const token = idx / num_q_heads;
        int const head = idx % num_q_heads;
        float const inv =
            rsqrtf(sum_sq_q[idx] / static_cast<float>(head_dim) + eps);
        Element *row = qkv + token * qkv_dim + head * head_dim;
        for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
            row[j] = to_h(to_f(row[j]) * inv * to_f(q_weight[j]));
        }
    } else {
        int const k_idx = idx - total_q_rows;
        if (k_idx >= total_k_rows) {
            return;
        }
        int const token = k_idx / num_kv_heads;
        int const head = k_idx % num_kv_heads;
        float const inv =
            rsqrtf(sum_sq_k[k_idx] / static_cast<float>(head_dim) + eps);
        Element *row = qkv + token * qkv_dim + q_size + head * head_dim;
        for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
            row[j] = to_h(to_f(row[j]) * inv * to_f(k_weight[j]));
        }
    }
}

namespace qkv_fusion {

class QkvRmsNormFusedOp {
public:
    Element *d_hidden{};
    Element *d_W{};
    Element *d_qkv{};
    Element *d_q_weight{};
    Element *d_k_weight{};
    float *d_sum_sq_q{};
    float *d_sum_sq_k{};

    cutlass::Status run(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaMemsetAsync(d_sum_sq_q, 0,
                                   (size_t)TOKENS * NUM_Q_HEADS * sizeof(float),
                                   stream));
        CUDA_CHECK(cudaMemsetAsync(d_sum_sq_k, 0,
                                   (size_t)TOKENS * NUM_KV_HEADS * sizeof(float),
                                   stream));

        typename GemmEpilogueFusion::Arguments gemm_args(
            cutlass::gemm::GemmUniversalMode::kGemm, {TOKENS, QKV_DIM, HIDDEN},
            {d_hidden, HIDDEN}, {d_W, QKV_DIM},
            typename EpilogueVisitor::Arguments(
                typename EpilogueFunctorOp::Params(
                    ElementCompute(1.0f), ElementCompute(0.0f)),
                {d_qkv, QKV_DIM}, {d_qkv, QKV_DIM}, d_sum_sq_q, d_sum_sq_k,
                Q_SIZE, KV_SIZE, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS));

        typename GemmEpilogueFusion::Params params(gemm_args);
        dim3 grid = SwizzleThreadBlock().get_grid_shape(params.grid_tiled_shape);
        dim3 block(GemmEpilogueFusion::kThreadCount, 1, 1);
        int smem = int(sizeof(typename GemmEpilogueFusion::SharedStorage));

        if (smem >= 48 * 1024) {
            CUDA_CHECK(cudaFuncSetAttribute(
                cutlass::Kernel<GemmEpilogueFusion>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        }

        cutlass::Kernel<GemmEpilogueFusion>
            <<<grid, block, smem, stream>>>(params);
        CUDA_CHECK(cudaGetLastError());

        int const total_rows = TOKENS * (NUM_Q_HEADS + NUM_KV_HEADS);
        apply_qk_rmsnorm_finalize<<<total_rows, HEAD_DIM, 0, stream>>>(
            d_qkv, TOKENS, QKV_DIM, Q_SIZE, KV_SIZE, HEAD_DIM, NUM_Q_HEADS,
            NUM_KV_HEADS, d_sum_sq_q, d_sum_sq_k, d_q_weight, d_k_weight, EPS);
        CUDA_CHECK(cudaGetLastError());
        return cutlass::Status::kSuccess;
    }
};

} // namespace qkv_fusion

// ── CPU 参考 (同 lesson17) ──
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
                            Element *qkv) {
    gemm_cpu(hidden, W, qkv, TOKENS, QKV_DIM, HIDDEN);
    for (int t = 0; t < TOKENS; ++t) {
        Element *row = qkv + t * QKV_DIM;
        rmsnorm_cpu_rows(row, q_weight, row, NUM_Q_HEADS, HEAD_DIM, EPS);
        rmsnorm_cpu_rows(row + Q_SIZE, k_weight, row + Q_SIZE, NUM_KV_HEADS,
                         HEAD_DIM, EPS);
    }
}

// ── lesson17 分离版 (对比 benchmark) ──
struct QkvRmsNormSeparateOp {
    using CutlassGemm = cutlass::gemm::device::Gemm<
        Element, Layout, Element, Layout, Element, Layout, ElementCompute,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ThreadblockShape,
        WarpShape, InstructionShape, EpilogueFunctorOp, SwizzleThreadBlock, 4>;

    Element *d_hidden{}, *d_W{}, *d_qkv{}, *d_q_weight{}, *d_k_weight{};
    float *d_sum_sq_q{}, *d_sum_sq_k{};
    CutlassGemm gemm;

    void run(cudaStream_t stream = nullptr) {
        typename CutlassGemm::Arguments args(
            {TOKENS, QKV_DIM, HIDDEN}, {d_hidden, HIDDEN}, {d_W, QKV_DIM},
            {d_qkv, QKV_DIM}, {d_qkv, QKV_DIM},
            {ElementCompute(1.0f), ElementCompute(0.0f)});
        CUTLASS_CHECK(gemm(args));

        int const total_rows = TOKENS * (NUM_Q_HEADS + NUM_KV_HEADS);
        compute_sum_sq_qk<<<total_rows, HEAD_DIM, 0, stream>>>(
            d_qkv, TOKENS, QKV_DIM, Q_SIZE, KV_SIZE, HEAD_DIM, NUM_Q_HEADS,
            NUM_KV_HEADS, d_sum_sq_q, d_sum_sq_k);
        apply_qk_rmsnorm_finalize<<<total_rows, HEAD_DIM, 0, stream>>>(
            d_qkv, TOKENS, QKV_DIM, Q_SIZE, KV_SIZE, HEAD_DIM, NUM_Q_HEADS,
            NUM_KV_HEADS, d_sum_sq_q, d_sum_sq_k, d_q_weight, d_k_weight, EPS);
        CUDA_CHECK(cudaGetLastError());
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

static float bench_fused(qkv_fusion::QkvRmsNormFusedOp &op, int warmup,
                         int repeats) {
    for (int i = 0; i < warmup; ++i) {
        CUTLASS_CHECK(op.run());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        CUTLASS_CHECK(op.run());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

static float bench_separate(QkvRmsNormSeparateOp &op, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) {
        op.run();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        op.run();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

int main(int argc, char **argv) {
    if (argc > 1) {
        g_tokens = atoi(argv[1]);
        if (g_tokens <= 0) {
            fprintf(stderr, "用法: %s [tokens]\n", argv[0]);
            return 1;
        }
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 8) {
        fprintf(stderr, "需要 Ampere+ (sm_80+)\n");
        return 1;
    }

    printf("QKV + QK-RMSNorm 融合 Epilogue  —  %s\n\n", prop.name);
    printf("Pipeline: GEMM(EpilogueVisitor sum_sq) → finalize(RMSNorm)\n");
    printf("hidden [%d×%d] @ W [%d×%d] → qkv [%d×%d]\n\n", TOKENS, HIDDEN,
           HIDDEN, QKV_DIM, TOKENS, QKV_DIM);

    const size_t bytes_hidden = (size_t)TOKENS * HIDDEN * sizeof(Element);
    const size_t bytes_W = (size_t)HIDDEN * QKV_DIM * sizeof(Element);
    const size_t bytes_qkv = (size_t)TOKENS * QKV_DIM * sizeof(Element);
    const size_t bytes_w = (size_t)HEAD_DIM * sizeof(Element);
    const size_t bytes_sq_q = (size_t)TOKENS * NUM_Q_HEADS * sizeof(float);
    const size_t bytes_sq_k = (size_t)TOKENS * NUM_KV_HEADS * sizeof(float);

    Element *h_hidden = (Element *)malloc(bytes_hidden);
    Element *h_W = (Element *)malloc(bytes_W);
    Element *h_qkv = (Element *)malloc(bytes_qkv);
    Element *h_ref = (Element *)malloc(bytes_qkv);
    Element *h_qw = (Element *)malloc(bytes_w);
    Element *h_kw = (Element *)malloc(bytes_w);

    for (int i = 0; i < (int)(TOKENS * HIDDEN); ++i) {
        h_hidden[i] = to_h((float)(i % 17) * 0.01f);
    }
    for (int i = 0; i < (int)(HIDDEN * QKV_DIM); ++i) {
        h_W[i] = to_h((float)(i % 13) * 0.01f);
    }
    for (int i = 0; i < HEAD_DIM; ++i) {
        h_qw[i] = to_h(1.0f + 0.01f * (float)(i % 7));
        h_kw[i] = to_h(1.0f + 0.01f * (float)(i % 5));
    }

    qkv_rmsnorm_cpu(h_hidden, h_W, h_qw, h_kw, h_ref);

    Element *d_hidden, *d_W, *d_qkv, *d_qw, *d_kw;
    float *d_sq_q, *d_sq_k;
    CUDA_CHECK(cudaMalloc(&d_hidden, bytes_hidden));
    CUDA_CHECK(cudaMalloc(&d_W, bytes_W));
    CUDA_CHECK(cudaMalloc(&d_qkv, bytes_qkv));
    CUDA_CHECK(cudaMalloc(&d_qw, bytes_w));
    CUDA_CHECK(cudaMalloc(&d_kw, bytes_w));
    CUDA_CHECK(cudaMalloc(&d_sq_q, bytes_sq_q));
    CUDA_CHECK(cudaMalloc(&d_sq_k, bytes_sq_k));
    CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden, bytes_hidden, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W, bytes_W, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qw, h_qw, bytes_w, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kw, h_kw, bytes_w, cudaMemcpyHostToDevice));

    qkv_fusion::QkvRmsNormFusedOp fused{d_hidden, d_W,     d_qkv,
                                        d_qw,     d_kw,     d_sq_q,
                                        d_sq_k};

    CUTLASS_CHECK(fused.run());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_qkv, d_qkv, bytes_qkv, cudaMemcpyDeviceToHost));

    printf("验证 q: %s\n",
           verify(h_qkv, h_ref, TOKENS * Q_SIZE, 0.08f) ? "通过" : "失败");
    printf("验证 k: %s\n",
           verify(h_qkv + Q_SIZE, h_ref + Q_SIZE, TOKENS * KV_SIZE, 0.08f)
               ? "通过"
               : "失败");
    printf("验证 v: %s\n",
           verify(h_qkv + Q_SIZE + KV_SIZE, h_ref + Q_SIZE + KV_SIZE,
                  TOKENS * KV_SIZE, 0.08f)
               ? "通过"
               : "失败");

    QkvRmsNormSeparateOp separate{d_hidden, d_W, d_qkv, d_qw, d_kw,
                                  d_sq_q,     d_sq_k};
    const float t_fused = bench_fused(fused, 5, 50);
    const float t_sep = bench_separate(separate, 5, 50);

    printf("\n── PyTorch 等价端到端对比 (含 qkv_proj + qk_norm) ──\n");
    printf("  split/view       : 0 ms\n");
    printf("%28s  %8.3f ms\n", "lesson17 分离 (3 stage)", t_sep);
    printf("%28s  %8.3f ms  (%.2fx)\n", "lesson18 融合 epilogue", t_fused,
           t_sep / t_fused);
    printf("  → 对应 self.qkv_proj + q_norm + k_norm 整段\n");

    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_qw));
    CUDA_CHECK(cudaFree(d_kw));
    CUDA_CHECK(cudaFree(d_sq_q));
    CUDA_CHECK(cudaFree(d_sq_k));
    free(h_hidden);
    free(h_W);
    free(h_qkv);
    free(h_ref);
    free(h_qw);
    free(h_kw);
    return 0;
}
