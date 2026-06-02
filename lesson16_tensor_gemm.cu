/*
 * Tensor Core GEMM (WMMA) vs FP32 tiled (lesson03)
 *
 * WMMA: A/B 为 FP16，累加 FP32，16×16×16 mma.sync → Tensor Core
 * 要求 M/N/K 为 16 的倍数；Ampere+ (sm_80+)
 *
 * nvcc -O3 -arch=sm_86 -o lesson16 lesson16_tensor_gemm.cu -lcublas
 * ./lesson16
 * ./lesson16 2048
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>

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
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP 32

/* lesson03 风格 FP32 tiled — CUDA core */
__global__ void matmul_tiled_fp32(const float *a, const float *b, float *c,
                                  int m, int n, int k) {
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

/*
 * WMMA Tensor Core GEMM
 * 每个 block = 1 warp，负责 C 上 16×16 tile
 * C[M×N] = A[M×K] × B[K×N], row-major, half × half → float acc
 */
__global__ void wmma_gemm(const half *A, const half *B, float *C, int M, int N,
                          int K) {
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int row = tile_m * WMMA_M;
    const int col = tile_n * WMMA_N;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           float>
        acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int i = 0; i < K; i += WMMA_K) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               half, nvcuda::wmma::row_major>
            a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               half, nvcuda::wmma::row_major>
            b_frag;

        nvcuda::wmma::load_matrix_sync(a_frag, A + row * K + i, K);
        nvcuda::wmma::load_matrix_sync(b_frag, B + i * N + col, N);
        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    nvcuda::wmma::store_matrix_sync(C + row * N + col, acc, N,
                                    nvcuda::wmma::mem_row_major);
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

static void float_to_half(const float *src, half *dst, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] = __float2half(src[i]);
    }
}

static void cublas_gemm_tensor(cublasHandle_t handle, int m, int n, int k,
                               const half *d_A, const half *d_B, float *d_C) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                              d_B, CUDA_R_16F, n, d_A, CUDA_R_16F, k, &beta,
                              d_C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static bool verify(const float *gpu, const float *ref, int count, float tol) {
    float max_err = 0.0f;
    for (int i = 0; i < count; ++i) {
        max_err = fmaxf(max_err, fabsf(gpu[i] - ref[i]));
    }
    printf("  max_err = %.4f %s\n", max_err, max_err < tol ? "(OK)" : "(FAIL)");
    return max_err < tol;
}

static float gflops(int m, int n, int k, float ms) {
    return (float)(2.0 * m * n * k / (ms * 1e6));
}

static float bench_fp32_tiled(const float *d_A, const float *d_B, float *d_C,
                              int m, int n, int k, int warmup, int repeats) {
    const dim3 block(TILE, TILE);
    const dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    for (int i = 0; i < warmup; ++i) {
        matmul_tiled_fp32<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        matmul_tiled_fp32<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

static float bench_wmma(const half *d_A, const half *d_B, float *d_C, int m,
                        int n, int k, int warmup, int repeats) {
    const dim3 grid(n / WMMA_N, m / WMMA_M);
    const dim3 block(WARP);
    for (int i = 0; i < warmup; ++i) {
        wmma_gemm<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        wmma_gemm<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

static float bench_cublas_tensor(cublasHandle_t handle, const half *d_A,
                                 const half *d_B, float *d_C, int m, int n,
                                 int k, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) {
        cublas_gemm_tensor(handle, m, n, k, d_A, d_B, d_C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        cublas_gemm_tensor(handle, m, n, k, d_A, d_B, d_C);
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
        if (mnk <= 0 || mnk % 16 != 0) {
            fprintf(stderr, "用法: %s [M=N=K]  (须为 16 的倍数)\n", argv[0]);
            return 1;
        }
    }

    const int M = mnk;
    const int N = mnk;
    const int K = mnk;
    const int count_C = M * N;
    const size_t bytes_f = (size_t)count_C * sizeof(float);
    const size_t bytes_A = (size_t)M * K * sizeof(float);
    const size_t bytes_B = (size_t)K * N * sizeof(float);
    const size_t bytes_h = (size_t)M * K * sizeof(half);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) {
        fprintf(stderr, "需要 Volta+ (Tensor Core)\n");
        return 1;
    }

    printf("Tensor Core GEMM  demo  (%s)\n", prop.name);
    printf("C[%d×%d] = A[%d×%d] × B[%d×%d]\n\n", M, N, M, K, K, N);

    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    float *h_C = (float *)malloc(bytes_f);
    float *h_ref = (float *)malloc(bytes_f);
    half *h_Ah = (half *)malloc(bytes_h);
    half *h_Bh = (half *)malloc((size_t)K * N * sizeof(half));

    for (int i = 0; i < M * K; ++i) h_A[i] = (float)(i % 17);
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)(i % 13);
    float_to_half(h_A, h_Ah, M * K);
    float_to_half(h_B, h_Bh, K * N);

    float *d_A, *d_B, *d_C;
    half *d_Ah, *d_Bh;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_f));
    CUDA_CHECK(cudaMalloc(&d_Ah, bytes_h));
    CUDA_CHECK(cudaMalloc(&d_Bh, (size_t)K * N * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Ah, h_Ah, bytes_h, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bh, h_Bh, (size_t)K * N * sizeof(half),
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    if (M * N * K <= 1024LL * 1024 * 1024) {
        matmul_cpu(h_A, h_B, h_ref, M, N, K);
    }

    matmul_tiled_fp32<<<dim3(N / TILE, M / TILE), dim3(TILE, TILE)>>>(
        d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_f, cudaMemcpyDeviceToHost));
    if (M * N * K <= 1024LL * 1024 * 1024) {
        printf("FP32 tiled (CUDA core) 验证: %s\n",
               verify(h_C, h_ref, count_C, 1e-2f) ? "通过" : "失败");
    }

    wmma_gemm<<<dim3(N / WMMA_N, M / WMMA_M), dim3(WARP)>>>(d_Ah, d_Bh, d_C, M,
                                                           N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_f, cudaMemcpyDeviceToHost));
    if (M * N * K <= 1024LL * 1024 * 1024) {
        printf("WMMA Tensor Core      验证: %s\n",
               verify(h_C, h_ref, count_C, 2.0f) ? "通过" : "失败");
    }

    cublas_gemm_tensor(handle, M, N, K, d_Ah, d_Bh, d_C);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_f, cudaMemcpyDeviceToHost));
    if (M * N * K <= 1024LL * 1024 * 1024) {
        printf("cuBLAS Tensor Op      验证: %s\n",
               verify(h_C, h_ref, count_C, 2.0f) ? "通过" : "失败");
    }
    printf("\n");

    const float t_fp32 = bench_fp32_tiled(d_A, d_B, d_C, M, N, K, 3, 20);
    const float t_wmma = bench_wmma(d_Ah, d_Bh, d_C, M, N, K, 3, 20);
    const float t_cublas =
        bench_cublas_tensor(handle, d_Ah, d_Bh, d_C, M, N, K, 3, 20);

    printf("%18s  %8s  %10s  %12s\n", "版本", "ms", "GFLOPS", "vs FP32 tiled");
    printf("%18s  %8s  %10s  %12s\n", "----", "--", "------", "------------");
    printf("%18s  %8.3f  %10.1f  %12.2fx\n", "FP32 tiled", t_fp32,
           gflops(M, N, K, t_fp32), 1.0f);
    printf("%18s  %8.3f  %10.1f  %12.2fx\n", "WMMA Tensor Core", t_wmma,
           gflops(M, N, K, t_wmma), t_fp32 / t_wmma);
    printf("%18s  %8.3f  %10.1f  %12.2fx\n", "cuBLAS Tensor Op", t_cublas,
           gflops(M, N, K, t_cublas), t_fp32 / t_cublas);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_Ah));
    CUDA_CHECK(cudaFree(d_Bh));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);
    free(h_Ah);
    free(h_Bh);
    return 0;
}
