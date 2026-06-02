/*
 * 阶段 0 - 练习 1：向量加法
 *
 * 目标：理解 CUDA 程序的标准流程
 *   Host 准备数据 → 拷贝到 Device → 启动 Kernel → 拷回结果 → 验证
 *
 * 编译：nvcc -o lesson00 lesson00_vector_add.cu
 * 运行：./lesson00
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 检查 CUDA API 调用是否成功（阶段 0 必学习惯）
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ── Kernel：在 GPU 上执行的函数 ──
// __global__ 表示这是从 CPU 调用、在 GPU 上运行的函数
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    // 每个 thread 负责一个元素
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// CPU 版本，用于验证 GPU 结果是否正确
void vector_add_cpu(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1 << 20;  // 1048576 个元素，约 4MB
    const size_t bytes = N * sizeof(float);

    // ── 1. Host 端分配并初始化数据 ──
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // ── 2. Device 端分配显存 ──
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    // ── 3. Host → Device 拷贝 ──
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // ── 4. 配置并启动 Kernel ──
    int threads_per_block = 64;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    printf("N = %d, threads/block = %d, blocks = %d\n", N, threads_per_block,
           blocks_per_grid);
    printf("总 thread 数 = %d (>= N 才能覆盖所有元素)\n",
           blocks_per_grid * threads_per_block);

    // 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vector_add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("GPU kernel 耗时: %.3f ms\n", ms);

    // ── 5. Device → Host 拷贝结果 ──
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // ── 6. CPU 参考结果 + 验证 ──
    vector_add_cpu(h_a, h_b, h_ref, N);

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_ref[i]) {
            if (errors < 5) {
                printf("错误 at [%d]: GPU=%f, CPU=%f\n", i, h_c[i], h_ref[i]);
            }
            ++errors;
        }
    }

    if (errors == 0) {
        printf("验证通过！前 5 个结果: ");
        for (int i = 0; i < 5; ++i) {
            printf("%.0f ", h_c[i]);
        }
        printf("...\n");
    } else {
        printf("验证失败，共 %d 个错误\n", errors);
    }

    // ── 7. 释放资源 ──
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_ref);

    return errors == 0 ? 0 : 1;
}
