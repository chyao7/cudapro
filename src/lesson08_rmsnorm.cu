/*
 * 阶段 5 - 练习 2：RMSNorm + Warp Shuffle 规约
 *
 * 阶段 4 回顾：Softmax 行级规约、多阶段 kernel
 * 新知识点：
 *   1. Warp shuffle (__shfl_down_sync) — LLM kernel 常用原语
 *   2. Fused RMSNorm — Llama / Qwen 等模型的归一化层
 *
 * 对 in[rows × cols] 每一行：
 *   rms = rsqrt(mean(x²) + ε)
 *   out[i] = x[i] * rms * weight[i]
 *
 * 编译：nvcc -o lesson08 lesson08_rmsnorm.cu
 * 运行：./lesson08
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define BLOCK 256
#define WARP 32
#define EPS 1e-6f

// ── lesson04 风格：shared memory 树形 sum 规约 ──
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

// ── 新：warp 内 shuffle 规约（无 shared memory）──
__device__ float warp_reduce_sum(float val) {
    // 0xffffffff = 整个 warp 32 个 thread 都参与
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // lane 0 持有 warp 内 sum
}

// block 级 sum：先 warp 内规约，再 warp 间规约
__device__ float block_reduce_sum_warp(float val) {
    __shared__ float warp_sums[BLOCK / WARP];  // 256/32 = 8 个 warp

    int tid = threadIdx.x;
    int lane = tid % WARP;
    int wid = tid / WARP;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        warp_sums[wid] = val;
    }
    __syncthreads();

    // 第一个 warp 汇总各 warp 的部分和（仅 lane 0 持有正确结果）
    val = (tid < blockDim.x / WARP) ? warp_sums[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    // 广播 block 总和给所有 thread（否则只有 tid==0 正确，其余 thread 的 rms 会错）
    __shared__ float block_sum;
    if (tid == 0) {
        block_sum = val;
    }
    __syncthreads();
    return block_sum;
}

// 版本 1：shared memory 规约（对比基准）
__global__ void rmsnorm_shared(const float *in, const float *weight, float *out,
                               int cols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *row_in = in + row * cols;
    float *row_out = out + row * cols;

    float local_sq = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        local_sq += row_in[j] * row_in[j];
    }
    float sum_sq = block_reduce_sum_shared(local_sq);
    float rms = rsqrtf(sum_sq / cols + eps);

    for (int j = tid; j < cols; j += blockDim.x) {
        row_out[j] = row_in[j] * rms * weight[j];
    }
}

// 版本 2：warp shuffle 规约（vLLM 常用思路）
__global__ void rmsnorm_warp(const float *in, const float *weight, float *out,
                             int cols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *row_in = in + row * cols;
    float *row_out = out + row * cols;

    float local_sq = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        local_sq += row_in[j] * row_in[j];
    }
    float sum_sq = block_reduce_sum_warp(local_sq);
    float rms = rsqrtf(sum_sq / cols + eps);

    for (int j = tid; j < cols; j += blockDim.x) {
        row_out[j] = row_in[j] * rms * weight[j];
    }
}

void rmsnorm_cpu(const float *in, const float *weight, float *out, int rows,
                 int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float *row_in = in + r * cols;
        float *row_out = out + r * cols;

        double sum_sq = 0.0;
        for (int j = 0; j < cols; ++j) {
            sum_sq += (double)row_in[j] * row_in[j];
        }
        float rms = rsqrtf((float)(sum_sq / cols) + eps);

        for (int j = 0; j < cols; ++j) {
            row_out[j] = row_in[j] * rms * weight[j];
        }
    }
}

bool verify(const float *gpu, const float *cpu, int n, float tol) {
    for (int i = 0; i < n; ++i) {
        if (fabsf(gpu[i] - cpu[i]) > tol) return false;
    }
    return true;
}

float benchmark(void (*kernel)(const float *, const float *, float *, int,
                               float),
                const float *d_in, const float *d_weight, float *d_out,
                int rows, int cols, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) {
        kernel<<<rows, BLOCK>>>(d_in, d_weight, d_out, cols, EPS);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        kernel<<<rows, BLOCK>>>(d_in, d_weight, d_out, cols, EPS);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

int main() {
    const int ROWS = 4096;   // batch × seq 或 token 数
    const int COLS = 4096;   // hidden size
    const int N = ROWS * COLS;
    const size_t bytes = N * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_weight = (float *)malloc(COLS * sizeof(float));
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < N; ++i) {
        h_in[i] = (float)(rand() % 1000 - 500) / 100.0f;
    }
    for (int j = 0; j < COLS; ++j) {
        h_weight[j] = 1.0f;
    }

    float *d_in, *d_weight, *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_weight, COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, COLS * sizeof(float),
                           cudaMemcpyHostToDevice));

    printf("RMSNorm: %d 行 × %d 列 (hidden=%d)\n", ROWS, COLS, COLS);
    printf("launch: grid=%d (每 block 一行), block=%d\n\n", ROWS, BLOCK);

    rmsnorm_shared<<<ROWS, BLOCK>>>(d_in, d_weight, d_out, COLS, EPS);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    rmsnorm_cpu(h_in, h_weight, h_ref, ROWS, COLS, EPS);
    printf("shared 规约 验证: %s\n",
           verify(h_out, h_ref, N, 1e-3f) ? "通过" : "失败");

    float t_shared = benchmark(rmsnorm_shared, d_in, d_weight, d_out, ROWS,
                               COLS, 3, 50);
    printf("shared 规约 耗时: %.3f ms\n\n", t_shared);

    rmsnorm_warp<<<ROWS, BLOCK>>>(d_in, d_weight, d_out, COLS, EPS);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("warp shuffle 验证: %s\n",
           verify(h_out, h_ref, N, 1e-3f) ? "通过" : "失败");

    float t_warp =
        benchmark(rmsnorm_warp, d_in, d_weight, d_out, ROWS, COLS, 3, 50);
    printf("warp shuffle 耗时: %.3f ms\n", t_warp);
    printf("加速比 (warp vs shared): %.2fx\n", t_shared / t_warp);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_weight);
    free(h_out);
    free(h_ref);
    return 0;
}
