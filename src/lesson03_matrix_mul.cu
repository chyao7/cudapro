/*
 * 阶段 2 - 练习 1：矩阵乘法 (GEMM)
 *
 * 阶段 1 回顾：shared memory tiling、合并访存
 * 阶段 2 新内容：
 *   - 计算密集型 kernel（FLOPs >> 内存访问）
 *   - shared memory 数据复用（同一元素服务 block 内多个 thread）
 *   - naive GEMM vs tiled GEMM 性能对比
 *
 * C[M×N] = A[M×K] × B[K×N]
 *
 * 编译：nvcc -o lesson03 lesson03_matrix_mul.cu
 * 运行：./lesson03
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

#define TILE 16

// 版本 1：naive — 每个 thread 算 C 的一个元素，内层循环反复读 global memory
__global__ void matmul_naive(const float *a, const float *b, float *c, int m,
                             int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int t = 0; t < k; ++t) {
            sum += a[row * k + t] * b[t * n + col];
        }
        c[row * n + col] = sum;
    }
}

// 版本 2：tiled — 把 K 维分块，每块 tile 载入 shared memory 后复用
__global__ void matmul_tiled(const float *a, const float *b, float *c, int m,
                             int n, int k) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;

    float sum = 0.0f;
    int num_tiles = (k + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        // 合并读 A 的子块 (TILE×TILE) 和 B 的子块 (TILE×TILE)
        int a_col = t * TILE + threadIdx.x;
        int a_row = row;
        if (a_row < m && a_col < k)
            sA[threadIdx.y][threadIdx.x] = a[a_row * k + a_col];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        int b_row = t * TILE + threadIdx.y;
        int b_col = col;
        if (b_row < k && b_col < n)
            sB[threadIdx.y][threadIdx.x] = b[b_row * n + b_col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // 在 shared memory 里做 TILE 次乘加
        for (int i = 0; i < TILE; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

void matmul_cpu(const float *a, const float *b, float *c, int m, int n, int k) {
    for (int r = 0; r < m; ++r) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int t = 0; t < k; ++t) {
                sum += a[r * k + t] * b[t * n + col];
            }
            c[r * n + col] = sum;
        }
    }
}

float benchmark(void (*kernel)(const float *, const float *, float *, int, int,
                               int),
                const float *d_a, const float *d_b, float *d_c, int m, int n,
                int k, dim3 grid, dim3 block, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) {
        kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

bool verify(const float *gpu, const float *cpu, int size) {
    for (int i = 0; i < size; ++i) {
        if (gpu[i] != cpu[i]) return false;
    }
    return true;
}

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    const int size_c = M * N;
    const size_t bytes_a = M * K * sizeof(float);
    const size_t bytes_b = K * N * sizeof(float);
    const size_t bytes_c = size_c * sizeof(float);

    float *h_a = (float *)malloc(bytes_a);
    float *h_b = (float *)malloc(bytes_b);
    float *h_c = (float *)malloc(bytes_c);
    float *h_ref = (float *)malloc(bytes_c);

    for (int i = 0; i < M * K; ++i) h_a[i] = (float)(i % 17);
    for (int i = 0; i < K * N; ++i) h_b[i] = (float)(i % 13);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes_a));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes_b));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes_c));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    printf("矩阵乘法: A[%d×%d] × B[%d×%d] = C[%d×%d]\n\n", M, K, K, N, M, N);

    matmul_naive<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost));
    matmul_cpu(h_a, h_b, h_ref, M, N, K);
    printf("naive  验证: %s\n", verify(h_c, h_ref, size_c) ? "通过" : "失败");

    float t_naive = benchmark(matmul_naive, d_a, d_b, d_c, M, N, K, grid,
                              block, 3, 20);
    printf("naive  耗时: %.3f ms\n\n", t_naive);

    matmul_tiled<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost));
    printf("tiled  验证: %s\n", verify(h_c, h_ref, size_c) ? "通过" : "失败");

    float t_tiled = benchmark(matmul_tiled, d_a, d_b, d_c, M, N, K, grid,
                               block, 3, 20);
    printf("tiled  耗时: %.3f ms\n", t_tiled);
    printf("加速比: %.1fx\n", t_naive / t_tiled);

    // GFLOPS = 2*M*N*K / time
    double flops = 2.0 * M * N * K;
    printf("tiled 算力: %.1f GFLOPS\n", flops / (t_tiled * 1e6));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_ref);
    return 0;
}
