/*
 * 阶段 1 - 练习 1.2：矩阵加法
 *
 * 阶段 0 回顾：1D indexing，vector_add<<<blocks, threads>>>
 * 阶段 1 新内容：
 *   - 2D Grid / 2D Block（dim3）
 *   - 行主序 (row-major) 内存布局
 *   - 用 (row, col) 定位矩阵元素
 *
 * 编译：nvcc -o lesson01 lesson01_matrix_add.cu
 * 运行：./lesson01
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

// 行主序：A[row][col] 在内存中是 A[row * cols + col]
__host__ __device__ inline int index2d(int row, int col, int cols) {
    return row * cols + col;
}

__global__ void matrix_add(const float *a, const float *b, float *c, int rows,
                           int cols) {
    // 2D thread 坐标 → 矩阵 (row, col)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = index2d(row, col, cols);
        c[idx] = a[idx] + b[idx];
    }
}

void matrix_add_cpu(const float *a, const float *b, float *c, int rows,
                    int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int col = 0; col < cols; ++col) {
            int idx = r * cols + col;
            c[idx] = a[idx] + b[idx];
        }
    }
}

int main() {
    const int ROWS = 1024;
    const int COLS = 1024;
    const int N = ROWS * COLS;
    const size_t bytes = N * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i % 100);
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 2D launch 配置：每个 block 是 16×16 的 thread 方阵
    dim3 threads(16, 16);  // (x=col方向, y=row方向)
    dim3 blocks((COLS + threads.x-1) / threads.x,
                (ROWS + threads.y-1) / threads.y);

    printf("矩阵大小: %d × %d = %d 元素\n", ROWS, COLS, N);
    printf("block: (%d, %d), grid: (%d, %d)\n", threads.x, threads.y, blocks.x,
           blocks.y);
    int total_threads = blocks.x * blocks.y * threads.x * threads.y;
    printf("总 thread 数 = %d (覆盖 %d 元素，含边界保护)\n", total_threads, N);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matrix_add<<<blocks, threads>>>(d_a, d_b, d_c, ROWS, COLS);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("GPU kernel 耗时: %.3f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    matrix_add_cpu(h_a, h_b, h_ref, ROWS, COLS);

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_ref[i]) {
            if (errors < 5) {
                printf("错误 at flat[%d]: GPU=%f, CPU=%f\n", i, h_c[i],
                       h_ref[i]);
            }
            ++errors;
        }
    }

    if (errors == 0) {
        printf("验证通过！\n");
        printf("C[0][0]=%.0f  C[0][1]=%.0f  C[1][0]=%.0f  C[%d][%d]=%.0f\n",
               h_c[0], h_c[1], h_c[COLS], ROWS - 1, COLS - 1,
               h_c[index2d(ROWS - 1, COLS - 1, COLS)]);
    } else {
        printf("验证失败，共 %d 个错误\n", errors);
    }

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
