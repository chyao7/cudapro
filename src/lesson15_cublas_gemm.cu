/*
 * 手写 tiled GEMM (lesson03) vs cuBLAS SGEMM
 * C[M×N] = A[M×K] × B[K×N]  row-major
 *
 * nvcc -O3 -arch=sm_86 -o lesson15 lesson15_cublas_gemm.cu -lcublas
 * ./lesson15
 * ./lesson15 2048
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t st = (call);                                          \
        if (st != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error at %s:%d: code %d\n", __FILE__,   \
                    __LINE__, (int)st);                                      \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define DEFAULT_MNK 1024
#define TILE 16

__global__ void matmul_tiled(const float *a, const float *b, float *c, int m,
                             int n, int k) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int col = blockIdx.x * TILE + threadIdx.x;
    const int row = blockIdx.y * TILE + threadIdx.y;
    float sum = 0.0f;
    const int num_tiles = (k + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE + threadIdx.x;
        if (row < m && a_col < k)
            sA[threadIdx.y][threadIdx.x] = a[row * k + a_col];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        const int b_row = t * TILE + threadIdx.y;
        if (b_row < k && col < n)
            sB[threadIdx.y][threadIdx.x] = b[b_row * n + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

static void matmul_cpu(const float *a, const float *b, float *c, int m, int n,
                       int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int t = 0; t < k; ++t) {
                sum += (double)a[i * k + t] * b[t * n + j];
            }
            c[i * n + j] = (float)sum;
        }
    }
}

static void cublas_gemm(cublasHandle_t handle, int m, int n, int k,
                        const float *d_A, const float *d_B, float *d_C) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                             d_B, n, d_A, k, &beta, d_C, n));
}

static bool verify(const float *gpu, const float *ref, int count, float tol) {
    float max_err = 0.0f;
    for (int i = 0; i < count; ++i) {
        max_err = fmaxf(max_err, fabsf(gpu[i] - ref[i]));
    }
    printf("  max_err = %.6f %s\n", max_err, max_err < tol ? "(OK)" : "(FAIL)");
    return max_err < tol;
}

static float gflops(int m, int n, int k, float ms) {
    return (float)(2.0 * m * n * k / (ms * 1e6));
}

static float benchmark_tiled(const float *d_A, const float *d_B, float *d_C,
                             int m, int n, int k, dim3 grid, dim3 block,
                             int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) {
        matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

static float benchmark_cublas(cublasHandle_t handle, const float *d_A,
                              const float *d_B, float *d_C, int m, int n, int k,
                              int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) {
        cublas_gemm(handle, m, n, k, d_A, d_B, d_C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        cublas_gemm(handle, m, n, k, d_A, d_B, d_C);
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
    int mnk = DEFAULT_MNK;
    if (argc > 1) {
        mnk = atoi(argv[1]);
        if (mnk <= 0) {
            fprintf(stderr, "用法: %s [M=N=K]\n", argv[0]);
            return 1;
        }
    }

    const int M = mnk;
    const int N = mnk;
    const int K = mnk;
    const int count_C = M * N;
    const size_t bytes_A = (size_t)M * K * sizeof(float);
    const size_t bytes_B = (size_t)K * N * sizeof(float);
    const size_t bytes_C = (size_t)count_C * sizeof(float);

    printf("GEMM: C[%d×%d] = A[%d×%d] × B[%d×%d]\n\n", M, N, M, K, K, N);

    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    float *h_C = (float *)malloc(bytes_C);
    float *h_ref = (float *)malloc(bytes_C);

    for (int i = 0; i < M * K; ++i) h_A[i] = (float)(i % 17);
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)(i % 13);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const dim3 block(TILE, TILE);
    const dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    if (M * N * K <= 1024LL * 1024 * 1024) {
        matmul_cpu(h_A, h_B, h_ref, M, N, K);
    }

    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    if (M * N * K <= 1024LL * 1024 * 1024) {
        printf("tiled  验证: %s\n",
               verify(h_C, h_ref, count_C, 1e-2f) ? "通过" : "失败");
    }

    cublas_gemm(handle, M, N, K, d_A, d_B, d_C);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    if (M * N * K <= 1024LL * 1024 * 1024) {
        printf("cuBLAS 验证: %s\n",
               verify(h_C, h_ref, count_C, 1e-2f) ? "通过" : "失败");
    }
    printf("\n");

    const float t_tiled =
        benchmark_tiled(d_A, d_B, d_C, M, N, K, grid, block, 3, 20);
    const float t_cublas =
        benchmark_cublas(handle, d_A, d_B, d_C, M, N, K, 3, 20);

    printf("%8s  %10s  %10s  %10s\n", "版本", "ms", "GFLOPS", "vs tiled");
    printf("%8s  %10s  %10s  %10s\n", "----", "--", "------", "--------");
    printf("%8s  %10.3f  %10.1f  %10.2fx\n", "tiled", t_tiled,
           gflops(M, N, K, t_tiled), 1.0f);
    printf("%8s  %10.3f  %10.1f  %10.2fx\n", "cuBLAS", t_cublas,
           gflops(M, N, K, t_cublas), t_tiled / t_cublas);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);
    return 0;
}
