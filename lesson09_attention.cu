/*
 * 阶段 5 - 练习 3：Scaled Dot-Product Attention
 *
 * 对比两种 softmax 规约实现：
 *   1. attention_shared — shared memory 树形规约（lesson04/05）
 *   2. attention_warp   — warp shuffle 规约（lesson08）
 *
 * 编译：nvcc -o lesson09 lesson09_attention.cu
 * 运行：./lesson09
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <float.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define SEQ 2048
#define DIM 512
#define BLOCK 256
#define WARP 32

// ── shared memory 树形规约 ──
__device__ float block_reduce_max_shared(float val) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    return sdata[0];
}

__device__ float block_reduce_sum_shared(float val) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    return sdata[0];
}

// ── warp shuffle 规约 ──
__device__ float warp_reduce_max(float val) {
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float block_reduce_max_warp(float val) {
    __shared__ float warp_sums[BLOCK / WARP];
    int tid = threadIdx.x;
    int lane = tid % WARP;
    int wid = tid / WARP;

    val = warp_reduce_max(val);
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();

    val = (tid < blockDim.x / WARP) ? warp_sums[lane] : -FLT_MAX;
    if (wid == 0) val = warp_reduce_max(val);

    __shared__ float result;
    if (tid == 0) result = val;
    __syncthreads();
    return result;
}

__device__ float block_reduce_sum_warp(float val) {
    __shared__ float warp_sums[BLOCK / WARP];
    int tid = threadIdx.x;
    int lane = tid % WARP;
    int wid = tid / WARP;

    val = warp_reduce_sum(val);
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();

    val = (tid < blockDim.x / WARP) ? warp_sums[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);

    __shared__ float result;
    if (tid == 0) result = val;
    __syncthreads();
    return result;
}

__device__ void attention_phase1_scores(const float *q, const float *K,
                                        float *smem, int seq, int dim,
                                        float scale, int tid) {
    for (int t = tid; t < seq; t += blockDim.x) {
        const float *k = K + t * dim;
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot += q[d] * k[d];
        }
        smem[t] = dot * scale;
    }
}

__device__ void attention_softmax_shared(float *smem, int seq, int tid) {
    float local_max = -FLT_MAX;
    for (int t = tid; t < seq; t += blockDim.x) {
        local_max = fmaxf(local_max, smem[t]);
    }
    float row_max = block_reduce_max_shared(local_max);

    float local_sum = 0.0f;
    for (int t = tid; t < seq; t += blockDim.x) {
        smem[t] = expf(smem[t] - row_max);
        local_sum += smem[t];
    }
    __syncthreads();

    float row_sum = block_reduce_sum_shared(local_sum);
    for (int t = tid; t < seq; t += blockDim.x) {
        smem[t] /= row_sum;
    }
}

__device__ void attention_softmax_warp(float *smem, int seq, int tid) {
    float local_max = -FLT_MAX;
    for (int t = tid; t < seq; t += blockDim.x) {
        local_max = fmaxf(local_max, smem[t]);
    }
    float row_max = block_reduce_max_warp(local_max);

    float local_sum = 0.0f;
    for (int t = tid; t < seq; t += blockDim.x) {
        smem[t] = expf(smem[t] - row_max);
        local_sum += smem[t];
    }
    __syncthreads();

    float row_sum = block_reduce_sum_warp(local_sum);
    for (int t = tid; t < seq; t += blockDim.x) {
        smem[t] /= row_sum;
    }
}

__device__ void attention_phase3_output(const float *smem, const float *V,
                                        float *o, int seq, int dim, int tid) {
    for (int d = tid; d < dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t < seq; ++t) {
            sum += smem[t] * V[t * dim + d];
        }
        o[d] = sum;
    }
}

__global__ void attention_shared(const float *Q, const float *K, const float *V,
                                 float *out, int seq, int dim) {
    extern __shared__ float smem[];
    int s = blockIdx.x;
    int tid = threadIdx.x;
    const float *q = Q + s * dim;
    float scale = rsqrtf((float)dim);

    attention_phase1_scores(q, K, smem, seq, dim, scale, tid);
    __syncthreads();
    attention_softmax_shared(smem, seq, tid);
    __syncthreads();
    attention_phase3_output(smem, V, out + s * dim, seq, dim, tid);
}

__global__ void attention_warp(const float *Q, const float *K, const float *V,
                               float *out, int seq, int dim) {
    extern __shared__ float smem[];
    int s = blockIdx.x;
    int tid = threadIdx.x;
    const float *q = Q + s * dim;
    float scale = rsqrtf((float)dim);

    attention_phase1_scores(q, K, smem, seq, dim, scale, tid);
    __syncthreads();
    attention_softmax_warp(smem, seq, tid);
    __syncthreads();
    attention_phase3_output(smem, V, out + s * dim, seq, dim, tid);
}

static float dot_cpu(const float *a, const float *b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) s += a[i] * b[i];
    return s;
}

void attention_cpu(const float *Q, const float *K, const float *V, float *out,
                   int seq, int dim) {
    float *scores = (float *)malloc(seq * sizeof(float));
    float scale = rsqrtf((float)dim);

    for (int s = 0; s < seq; ++s) {
        const float *q = Q + s * dim;
        for (int t = 0; t < seq; ++t) {
            scores[t] = dot_cpu(q, K + t * dim, dim) * scale;
        }

        float row_max = -FLT_MAX;
        for (int t = 0; t < seq; ++t) row_max = fmaxf(row_max, scores[t]);

        double row_sum = 0.0;
        for (int t = 0; t < seq; ++t) {
            scores[t] = expf(scores[t] - row_max);
            row_sum += scores[t];
        }
        for (int t = 0; t < seq; ++t) scores[t] /= (float)row_sum;

        float *o = out + s * dim;
        for (int d = 0; d < dim; ++d) {
            double sum = 0.0;
            for (int t = 0; t < seq; ++t) {
                sum += (double)scores[t] * V[t * dim + d];
            }
            o[d] = (float)sum;
        }
    }
    free(scores);
}

bool verify(const float *gpu, const float *cpu, int n, float tol) {
    for (int i = 0; i < n; ++i) {
        if (fabsf(gpu[i] - cpu[i]) > tol) return false;
    }
    return true;
}

float benchmark(void (*kernel)(const float *, const float *, const float *,
                               float *, int, int),
                const float *d_Q, const float *d_K, const float *d_V,
                float *d_out, int seq, int dim, size_t smem_bytes, int warmup,
                int repeats) {
    for (int i = 0; i < warmup; ++i) {
        kernel<<<seq, BLOCK, smem_bytes>>>(d_Q, d_K, d_V, d_out, seq, dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        kernel<<<seq, BLOCK, smem_bytes>>>(d_Q, d_K, d_V, d_out, seq, dim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

int main() {
    const int N = SEQ * DIM;
    const size_t bytes = N * sizeof(float);
    const size_t smem_bytes = SEQ * sizeof(float);

    float *h_Q = (float *)malloc(bytes);
    float *h_K = (float *)malloc(bytes);
    float *h_V = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < N; ++i) {
        h_Q[i] = (float)(rand() % 200 - 100) / 50.0f;
        h_K[i] = (float)(rand() % 200 - 100) / 50.0f;
        h_V[i] = (float)(rand() % 200 - 100) / 50.0f;
    }

    float *d_Q, *d_K, *d_V, *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_Q, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_K, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_V, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));

    attention_cpu(h_Q, h_K, h_V, h_ref, SEQ, DIM);

    printf("Attention: SEQ=%d, DIM=%d (单头简化版)\n", SEQ, DIM);
    printf("launch: grid=%d, block=%d, smem=%.1f KB/block\n\n", SEQ, BLOCK,
           smem_bytes / 1024.0f);
    printf("阶段2 差异: shared 规约 vs warp shuffle 规约\n");
    printf("阶段1/3 完全相同\n\n");

    attention_shared<<<SEQ, BLOCK, smem_bytes>>>(d_Q, d_K, d_V, d_out, SEQ,
                                                  DIM);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("shared 规约 验证: %s\n",
           verify(h_out, h_ref, N, 1e-2f) ? "通过" : "失败");

    float t_shared = benchmark(attention_shared, d_Q, d_K, d_V, d_out, SEQ, DIM,
                               smem_bytes, 3, 30);
    printf("shared 规约 耗时: %.3f ms\n\n", t_shared);

    attention_warp<<<SEQ, BLOCK, smem_bytes>>>(d_Q, d_K, d_V, d_out, SEQ, DIM);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("warp shuffle 验证: %s\n",
           verify(h_out, h_ref, N, 1e-2f) ? "通过" : "失败");

    float t_warp = benchmark(attention_warp, d_Q, d_K, d_V, d_out, SEQ, DIM,
                             smem_bytes, 3, 30);
    printf("warp shuffle 耗时: %.3f ms\n", t_warp);
    printf("加速比 (warp vs shared): %.2fx\n", t_shared / t_warp);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out));
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_out);
    free(h_ref);
    return 0;
}
