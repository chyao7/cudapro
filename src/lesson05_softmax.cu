/*
 * 阶段 4 - 练习 1：Softmax
 *
 * 阶段 2 回顾：树形规约 (max / sum)
 * 阶段 4 新内容：
 *   - 多阶段 kernel（max → exp+sum → normalize）
 *   - 数值稳定性（减去 max 防止 exp 溢出）
 *   - 批处理：每个 block 处理矩阵的一行
 *
 * 对矩阵 in[rows × cols] 的每一行做 softmax：
 *   out[i][j] = exp(in[i][j] - row_max) / row_sum
 *
 * 编译：nvcc -o lesson05 lesson05_softmax.cu
 * 运行：./lesson05
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

#define BLOCK 256

// block 内树形 max 规约
__device__ float block_reduce_max(float val) {
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

// block 内树形 sum 规约
__device__ float block_reduce_sum(float val) {
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

// 每个 block 处理一行；thread 用 stride 循环覆盖 cols > BLOCK 的情况
__global__ void softmax_rows(const float *in, float *out, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *row_in = in + row * cols;//指向当前行
    float *row_out = out + row * cols;//指向当前行

    // 阶段 1：找行最大值（数值稳定性的关键）
    float local_max = -FLT_MAX;
    for (int j = tid; j < cols; j += blockDim.x) {
        local_max = fmaxf(local_max, row_in[j]);
    }
    float row_max = block_reduce_max(local_max);

    // 阶段 2：exp(x - max) 并求和
    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        local_sum += expf(row_in[j] - row_max);
    }
    float row_sum = block_reduce_sum(local_sum);

    // 阶段 3：归一化写出
    for (int j = tid; j < cols; j += blockDim.x) {
        row_out[j] = expf(row_in[j] - row_max) / row_sum;
    }
}

void softmax_cpu(const float *in, float *out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float *row_in = in + r * cols;
        float *row_out = out + r * cols;

        float row_max = -FLT_MAX;
        for (int j = 0; j < cols; ++j) {
            row_max = fmaxf(row_max, row_in[j]);
        }

        double row_sum = 0.0;
        for (int j = 0; j < cols; ++j) {
            row_sum += exp((double)row_in[j] - row_max);
        }

        for (int j = 0; j < cols; ++j) {
            row_out[j] = (float)(exp((double)row_in[j] - row_max) / row_sum);
        }
    }
}

bool verify_softmax(const float *gpu, const float *cpu, int n, float tol) {
    for (int i = 0; i < n; ++i) {
        if (fabsf(gpu[i] - cpu[i]) > tol) {
            return false;
        }
    }
    return true;
}

bool verify_rows_sum_to_one(const float *out, int rows, int cols, float tol) {
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            sum += out[r * cols + j];
        }
        if (fabsf(sum - 1.0f) > tol) {
            printf("行 %d 求和 = %f (期望 1.0)\n", r, sum);
            return false;
        }
    }
    return true;
}

float benchmark_softmax(const float *d_in, float *d_out, int rows, int cols,
                        int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) {
        softmax_rows<<<rows, BLOCK>>>(d_in, d_out, cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        softmax_rows<<<rows, BLOCK>>>(d_in, d_out, cols);
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
    const int ROWS = 4096;
    const int COLS = 1024;
    const int N = ROWS * COLS;
    const size_t bytes = N * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < N; ++i) {
        h_in[i] = (float)(rand() % 2000 - 1000);  // [-1000, 999]
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    printf("Softmax: %d 行 × %d 列 = %d 元素\n", ROWS, COLS, N);
    printf("launch: grid=%d (每 block 一行), block=%d\n\n", ROWS, BLOCK);

    softmax_rows<<<ROWS, BLOCK>>>(d_in, d_out, COLS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    softmax_cpu(h_in, h_ref, ROWS, COLS);

    const float tol = 1e-3f;
    printf("数值验证 (vs CPU): %s\n",
           verify_softmax(h_out, h_ref, N, tol) ? "通过" : "失败");
    printf("行和验证 (每行 ≈ 1): %s\n",
           verify_rows_sum_to_one(h_out, ROWS, COLS, 1e-3f) ? "通过" : "失败");

    printf("\n示例 行0 前 5 个: ");
    for (int j = 0; j < 5; ++j) {
        printf("%.4f ", h_out[j]);
    }
    printf("\n");

    float ms = benchmark_softmax(d_in, d_out, ROWS, COLS, 3, 50);
    printf("\nGPU 耗时: %.3f ms\n", ms);
    printf("吞吐: %.1f M 元素/s\n", N / (ms * 1e3f));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    free(h_ref);
    return 0;
}
