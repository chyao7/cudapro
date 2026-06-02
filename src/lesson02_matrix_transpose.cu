/*
 * 阶段 1 - 练习 2：矩阵转置
 *
 * 新知识点：
 *   1. Memory Coalescing（合并访存）— 阶段 1 核心概念
 *   2. naive 版本 vs shared memory 版本性能对比
 *
 * 编译：nvcc -o lesson02 lesson02_matrix_transpose.cu
 * 运行：./lesson02
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

#define TILE 32

// 版本 1：naive 转置 — 读 coalesced，写 uncoalesced（慢）
__global__ void transpose_naive(const float *in, float *out, int rows,
                                int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col];
    }
}

// 版本 2：shared memory tiling — 读写都 coalesced
__global__ void transpose_tiled(const float *in, float *out, int rows,
                                int cols) {
    __shared__ float tile[TILE][TILE + 1];  // +1 避免 bank conflict

    int col_in = blockIdx.x * TILE + threadIdx.x;
    int row_in = blockIdx.y * TILE + threadIdx.y;

    // 合并读 global memory → shared memory
    if (row_in < rows && col_in < cols) {
        tile[threadIdx.y][threadIdx.x] = in[row_in * cols + col_in];
    }
    __syncthreads();

    // 转置坐标：block 在输出矩阵中的位置
    int col_out = blockIdx.y * TILE + threadIdx.x;
    int row_out = blockIdx.x * TILE + threadIdx.y;

    if (row_out < cols && col_out < rows) {
        out[row_out * rows + col_out] = tile[threadIdx.x][threadIdx.y];
    }
}

void transpose_cpu(const float *in, float *out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out[c * rows + r] = in[r * cols + c];
        }
    }
}

float benchmark(void (*kernel)(const float *, float *, int, int),
                const float *d_in, float *d_out, int rows, int cols,
                dim3 grid, dim3 block, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) {
        kernel<<<grid, block>>>(d_in, d_out, rows, cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        kernel<<<grid, block>>>(d_in, d_out, rows, cols);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

bool verify(const float *gpu, const float *cpu, int n) {
    for (int i = 0; i < n; ++i) {
        if (gpu[i] != cpu[i]) return false;
    }
    return true;
}

int main() {
    const int ROWS = 4096;
    const int COLS = 4096;
    const int N = ROWS * COLS;
    const size_t bytes = N * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    for (int i = 0; i < N; ++i) h_in[i] = (float)i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((COLS + TILE - 1) / TILE, (ROWS + TILE - 1) / TILE);

    printf("矩阵转置: %d × %d\n\n", ROWS, COLS);

    // naive
    transpose_naive<<<grid, block>>>(d_in, d_out, ROWS, COLS);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    transpose_cpu(h_in, h_ref, ROWS, COLS);
    printf("naive  验证: %s\n", verify(h_out, h_ref, N) ? "通过" : "失败");

    float t_naive =
        benchmark(transpose_naive, d_in, d_out, ROWS, COLS, grid, block, 3, 20);
    printf("naive  耗时: %.3f ms\n\n", t_naive);

    // tiled
    transpose_tiled<<<grid, block>>>(d_in, d_out, ROWS, COLS);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("tiled  验证: %s\n", verify(h_out, h_ref, N) ? "通过" : "失败");

    float t_tiled =
        benchmark(transpose_tiled, d_in, d_out, ROWS, COLS, grid, block, 3, 20);
    printf("tiled  耗时: %.3f ms\n", t_tiled);
    printf("加速比: %.1fx\n", t_naive / t_tiled);

    // 带宽估算（读+写各 N 个 float）
    float bw_gb = 2.0f * bytes / (t_tiled * 1e6);  // GB/s
    printf("tiled 有效带宽: %.1f GB/s\n", bw_gb);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    free(h_ref);
    return 0;
}
