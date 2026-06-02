/*
 * QKV Projection + QK-RMSNorm + RotaryEmbedding (CUTLASS API)
 *
 * post-GEMM 融合 kernel: 每个 (token, head) 一次完成 RMSNorm → RoPE；
 * RoPE 的 cos/sin 在 kernel 内按 position 即时计算。
 *
 * 运行: ./build-release/bin/lesson17_qkv_rmsnorm [tokens]
 */

#include <stdio.h>
#include <stdlib.h>

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
#define MAX_SEQ_LEN 65536
#define ROPE_THETA 1000000.0f
#define ROTARY_DIM HEAD_DIM

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

static Element to_h(float x) { return Element(x); }

__device__ float elem_to_f(Element x) { return static_cast<float>(x); }

__device__ void rope_cos_sin_at(int lane, int pos, int rotary_dim, float base,
                                float *cos_out, float *sin_out) {
    float const inv_freq =
        1.0f / powf(base, (2.0f * lane) / static_cast<float>(rotary_dim));
    float const angle = static_cast<float>(pos) * inv_freq;
    *cos_out = cosf(angle);
    *sin_out = sinf(angle);
}

__global__ void qk_rmsnorm_rope_fused(
    Element *qkv, int tokens, int qkv_dim, int q_offset, int q_num_heads,
    int k_offset, int k_num_heads, int head_dim, int rotary_dim,
    const Element *q_weight, const Element *k_weight, const int *positions,
    float rope_theta, float eps) {
    int const t = blockIdx.x;
    if (t >= tokens) {
        return;
    }

    int const warp_id = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const embed_dim = rotary_dim / 2;
    int const total_heads = q_num_heads + k_num_heads;

    __shared__ float cos_smem[32];
    __shared__ float sin_smem[32];

    if (lane < embed_dim && warp_id == 0) {
        int const pos = positions[t];
        float cos_val, sin_val;
        rope_cos_sin_at(lane, pos, rotary_dim, rope_theta, &cos_val, &sin_val);
        cos_smem[lane] = cos_val;
        sin_smem[lane] = sin_val;
    }
    __syncthreads();

    if (warp_id >= total_heads) {
        return;
    }

    int col_offset;
    int h;
    Element const *weight;
    if (warp_id < q_num_heads) {
        col_offset = q_offset;
        h = warp_id;
        weight = q_weight;
    } else {
        col_offset = k_offset;
        h = warp_id - q_num_heads;
        weight = k_weight;
    }

    Element *row = qkv + t * qkv_dim + col_offset + h * head_dim;

    float const x = elem_to_f(row[lane]);
    float const y = elem_to_f(row[lane + embed_dim]);
    float sq = x * x + y * y;

    for (int offset = 16; offset > 0; offset >>= 1) {
        sq += __shfl_down_sync(0xffffffff, sq, offset);
    }
    float const sum_sq = __shfl_sync(0xffffffff, sq, 0);
    float const inv_rms = rsqrtf(sum_sq / static_cast<float>(head_dim) + eps);

    float nx = x * inv_rms * elem_to_f(weight[lane]);
    float ny = y * inv_rms * elem_to_f(weight[lane + embed_dim]);
    float const cos = cos_smem[lane];
    float const sin = sin_smem[lane];

    row[lane] = Element(nx * cos - ny * sin);
    row[lane + embed_dim] = Element(ny * cos + nx * sin);
}

static void launch_qk_rmsnorm_rope_fused(Element *d_qkv, const int *d_positions,
                                       const Element *d_q_weight,
                                       const Element *d_k_weight,
                                       float rope_theta,
                                       cudaStream_t stream) {
    int const warps_per_block = NUM_Q_HEADS + NUM_KV_HEADS;
    int const block = warps_per_block * 32;
    qk_rmsnorm_rope_fused<<<TOKENS, block, 0, stream>>>(
        d_qkv, TOKENS, QKV_DIM, 0, NUM_Q_HEADS, Q_SIZE, NUM_KV_HEADS, HEAD_DIM,
        ROTARY_DIM, d_q_weight, d_k_weight, d_positions, rope_theta, EPS);
    CUDA_CHECK(cudaGetLastError());
}

struct QkvRmsNormRopeOp {
    Element *d_hidden{};
    Element *d_W{};
    Element *d_qkv{};
    Element *d_q_weight{};
    Element *d_k_weight{};
    int *d_positions{};

    CutlassGemm gemm;

    void qkv_proj(cudaStream_t stream = nullptr) {
        (void)stream;
        typename CutlassGemm::Arguments args(
            {TOKENS, QKV_DIM, HIDDEN},
            {d_hidden, HIDDEN},
            {d_W, QKV_DIM},
            {d_qkv, QKV_DIM},
            {d_qkv, QKV_DIM},
            {ElementAcc(1.0f), ElementAcc(0.0f)});
        CUTLASS_CHECK(gemm(args));
    }

    void qk_norm_rope(cudaStream_t stream = nullptr) {
        launch_qk_rmsnorm_rope_fused(d_qkv, d_positions, d_q_weight, d_k_weight,
                                     ROPE_THETA, stream);
    }

    void run(cudaStream_t stream = nullptr) {
        qkv_proj(stream);
        qk_norm_rope(stream);
    }
};

static float bench_stage(void (*fn)(QkvRmsNormRopeOp &, cudaStream_t),
                         QkvRmsNormRopeOp &op, int warmup, int repeats) {
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

static void stage_qkv_proj(QkvRmsNormRopeOp &op, cudaStream_t s) {
    op.qkv_proj(s);
}
static void stage_qk(QkvRmsNormRopeOp &op, cudaStream_t s) {
    op.qk_norm_rope(s);
}
static void stage_full(QkvRmsNormRopeOp &op, cudaStream_t s) { op.run(s); }

int main(int argc, char **argv) {
    if (argc > 1) {
        g_tokens = atoi(argv[1]);
        if (g_tokens <= 0) {
            fprintf(stderr, "用法: %s [tokens]\n", argv[0]);
            return 1;
        }
    }
    if (g_tokens > MAX_SEQ_LEN) {
        fprintf(stderr, "tokens (%d) 超过 MAX_SEQ_LEN (%d)\n", g_tokens,
                MAX_SEQ_LEN);
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 8) {
        fprintf(stderr, "需要 Ampere+ (sm_80+) Tensor Core\n");
        return 1;
    }

    printf("QKV + QK-RMSNorm + RoPE (CUTLASS)  —  %s\n\n", prop.name);
    printf("hidden [%d × %d]  @  W [%d × %d]  →  qkv [%d × %d]\n", TOKENS,
           HIDDEN, HIDDEN, QKV_DIM, TOKENS, QKV_DIM);
    printf("  q: %d heads × %d  |  k/v: %d heads × %d each  |  rope_theta=%.0f\n\n",
           NUM_Q_HEADS, HEAD_DIM, NUM_KV_HEADS, HEAD_DIM, ROPE_THETA);

    const size_t bytes_hidden = (size_t)TOKENS * HIDDEN * sizeof(Element);
    const size_t bytes_W = (size_t)HIDDEN * QKV_DIM * sizeof(Element);
    const size_t bytes_qw = (size_t)HEAD_DIM * sizeof(Element);
    const size_t bytes_kw = (size_t)HEAD_DIM * sizeof(Element);
    const size_t bytes_positions = (size_t)TOKENS * sizeof(int);

    Element *h_hidden = (Element *)malloc(bytes_hidden);
    Element *h_W = (Element *)malloc(bytes_W);
    Element *h_q_weight = (Element *)malloc(bytes_qw);
    Element *h_k_weight = (Element *)malloc(bytes_kw);
    int *h_positions = (int *)malloc(bytes_positions);

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
    for (int i = 0; i < TOKENS; ++i) {
        h_positions[i] = i;
    }

    Element *d_hidden, *d_W, *d_qkv;
    Element *d_q_weight, *d_k_weight;
    int *d_positions;
    CUDA_CHECK(cudaMalloc(&d_hidden, bytes_hidden));
    CUDA_CHECK(cudaMalloc(&d_W, (size_t)HIDDEN * QKV_DIM * sizeof(Element)));
    CUDA_CHECK(cudaMalloc(&d_qkv, (size_t)TOKENS * QKV_DIM * sizeof(Element)));
    CUDA_CHECK(cudaMalloc(&d_q_weight, bytes_qw));
    CUDA_CHECK(cudaMalloc(&d_k_weight, bytes_kw));
    CUDA_CHECK(cudaMalloc(&d_positions, bytes_positions));
    CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden, bytes_hidden, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W, bytes_W, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_weight, h_q_weight, bytes_qw, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_weight, h_k_weight, bytes_kw, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_positions, h_positions, bytes_positions,
                          cudaMemcpyHostToDevice));

    QkvRmsNormRopeOp op{d_hidden,   d_W,        d_qkv,
                        d_q_weight, d_k_weight, d_positions};

    const float t_proj = bench_stage(stage_qkv_proj, op, 5, 50);
    const float t_qk = bench_stage(stage_qk, op, 5, 50);
    const float t_full = bench_stage(stage_full, op, 5, 50);
    const double gemm_flops = 2.0 * TOKENS * HIDDEN * QKV_DIM;

    printf("── 分段耗时 ──\n");
    printf("  split(q,k,v)     : 0 ms  (view，无 kernel)\n");
    printf("%28s  %8.3f ms  %5.1f%%\n", "① qkv_proj (GEMM)", t_proj,
           100.f * t_proj / t_full);
    printf("%28s  %8.3f ms  %5.1f%%\n", "② qk_norm+rope (1 launch)", t_qk,
           100.f * t_qk / t_full);
    printf("%28s  %8.3f ms\n", "合计 (①+②)", t_full);
    printf("  GEMM TFLOPS ≈ %.1f  |  post-GEMM 占 %.1f%%\n",
           gemm_flops / (t_proj * 1e6), 100.f * t_qk / t_full);

    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_q_weight));
    CUDA_CHECK(cudaFree(d_k_weight));
    CUDA_CHECK(cudaFree(d_positions));
    free(h_hidden);
    free(h_W);
    free(h_q_weight);
    free(h_k_weight);
    free(h_positions);
    return 0;
}
