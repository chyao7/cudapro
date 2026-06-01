/*
 * 阶段 2 - 练习 2：并行规约 (Reduction)
 *
 * 阶段 2 回顾：shared memory、block 内协作
 * 新知识点：
 *   1. 树形规约 (tree reduction)
 *   2. block 级部分和 + atomicAdd 合并
 *   3. 多级规约 (multi-pass) — 避免大量 atomic 竞争
 *
 * 问题：对长度为 N 的数组求和
 *
 * 编译：nvcc -o lesson04 lesson04_reduction.cu
 * 运行：./lesson04
 */

#include <stdio.h>
#include <stdlib.h>
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

// 版本 1：block 内树形规约，block 结果用 atomicAdd 累加到 global
__global__ void reduce_atomic(const float *in, float *out, int n) {
    __shared__ float sdata[BLOCK];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    // 树形规约：每轮 active thread 减半
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// 版本 2：第一趟 block 规约 → 部分和数组；第二趟规约到最终结果（无 atomic 竞争）
__global__ void reduce_to_partials(const float *in, float *partials, int n) {
    __shared__ float sdata[BLOCK];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_final(const float *partials, float *out, int n) {
    __shared__ float sdata[BLOCK];

    int tid = threadIdx.x;
    int i = tid;

    sdata[tid] = (i < n) ? partials[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out = sdata[0];
    }
}

void reduce_multipass(const float *d_in, float *d_out, float *d_buf_a,
                      float *d_buf_b, int n, int num_blocks) {
    const float *src = d_in;
    float *dst = d_buf_a;
    int count = n;

    while (count > BLOCK) {
        int grid = (count + BLOCK - 1) / BLOCK;
        reduce_to_partials<<<grid, BLOCK>>>(src, dst, count);
        count = grid;
        src = dst;
        dst = (dst == d_buf_a) ? d_buf_b : d_buf_a;
    }

    reduce_final<<<1, BLOCK>>>(src, d_out, count);
    CUDA_CHECK(cudaDeviceSynchronize());
}

float reduce_multipass_result(const float *d_in, float *d_out, float *d_buf_a,
                              float *d_buf_b, int n, int num_blocks) {
    reduce_multipass(d_in, d_out, d_buf_a, d_buf_b, n, num_blocks);
    float result = 0;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

float reduce_cpu(const float *in, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += in[i];
    }
    return (float)sum;
}

float benchmark_atomic(const float *d_in, float *d_out, int n, int num_blocks,
                       int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        reduce_atomic<<<num_blocks, BLOCK>>>(d_in, d_out, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        reduce_atomic<<<num_blocks, BLOCK>>>(d_in, d_out, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

float benchmark_multipass(const float *d_in, float *d_out, float *d_buf_a,
                          float *d_buf_b, int n, int num_blocks, int warmup,
                          int repeats) {
    for (int i = 0; i < warmup; ++i) {
        reduce_multipass(d_in, d_out, d_buf_a, d_buf_b, n, num_blocks);
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        reduce_multipass(d_in, d_out, d_buf_a, d_buf_b, n, num_blocks);
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
    const int N = 1 << 24;  // 16777216 个元素
    const size_t bytes = N * sizeof(float);
    const int num_blocks = (N + BLOCK - 1) / BLOCK;

    float *h_in = (float *)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f;  // 期望和 = N
    }

    float cpu_sum = reduce_cpu(h_in, N);

    float *d_in, *d_out, *d_buf_a, *d_buf_b;
    CUDA_CHECK(cudaMalloc((void **)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_buf_a, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_buf_b, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    printf("并行规约: N = %d (%.1f M 元素), block = %d, grid = %d\n\n", N,
           N / 1e6f, BLOCK, num_blocks);
    printf("CPU 参考和: %.0f\n\n", cpu_sum);

    // atomic 版本
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    reduce_atomic<<<num_blocks, BLOCK>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_atomic = 0;
    CUDA_CHECK(cudaMemcpy(&gpu_atomic, d_out, sizeof(float),
                           cudaMemcpyDeviceToHost));
    printf("atomic 验证: %s (GPU=%.0f)\n", gpu_atomic == cpu_sum ? "通过" : "失败",
           gpu_atomic);

    float t_atomic = benchmark_atomic(d_in, d_out, N, num_blocks, 3, 50);
    printf("atomic 耗时: %.3f ms\n\n", t_atomic);

    // multi-pass 版本（N=16M → 65536 部分和，需多级规约）
    float gpu_multipass = reduce_multipass_result(d_in, d_out, d_buf_a, d_buf_b,
                                                  N, num_blocks);
    printf("multipass 验证: %s (GPU=%.0f)\n",
           gpu_multipass == cpu_sum ? "通过" : "失败", gpu_multipass);

    float t_multipass = benchmark_multipass(d_in, d_out, d_buf_a, d_buf_b, N,
                                            num_blocks, 3, 50);
    printf("multipass 耗时: %.3f ms\n", t_multipass);
    printf("加速比 (multipass vs atomic): %.1fx\n", t_atomic / t_multipass);

    float bw_gb = bytes / (t_multipass * 1e6);
    printf("multipass 有效带宽: %.1f GB/s\n", bw_gb);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_buf_a));
    CUDA_CHECK(cudaFree(d_buf_b));
    free(h_in);
    return 0;
}
