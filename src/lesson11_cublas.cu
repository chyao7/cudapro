/*
 * 阶段 6 - 练习 1：cuBLAS 入门 (SGEMM)
 *
 * 前置回顾：
 *   lesson03 — 手写 tiled GEMM
 *   lesson10 — FlashAttention 里的 Q@K^T、P@V 本质都是 GEMM
 *
 * cuBLAS 是什么：
 *   NVIDIA 闭源线性代数库，GEMM 已高度优化（Tensor Core、tiling、autotune）
 *   日常开发：能调库就不手写 kernel
 *
 * 本课目标：
 *   1. cublasHandle 生命周期
 *   2. cublasSgemm 参数含义（尤其 leading dimension）
 *   3. row-major 数据如何调用 col-major 的 cuBLAS
 *   4. 与 lesson03 手写 GEMM 性能对比
 *
 * 编译：nvcc -o lesson11 lesson11_cublas.cu -lcublas
 * 运行：./lesson11
 *
 * ── cuBLAS 学习路线（建议顺序）──
 *   [本课]  cublasSgemm              — 单精度 GEMM
 *   下一步  cublasGemmEx             — FP16/BF16/TF32 + Tensor Core
 *   再下一步 cublasGemmStridedBatched — multi-head / batch matmul
 *   进阶    cuBLASLt                — 可配置 epilogue（bias、GELU）
 *   定制    CUTLASS                  — cuBLAS 不够用时自己拼 kernel
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

#define M 2048
#define N 2048
#define K 512
#define TILE 16

// ── lesson03 风格 tiled GEMM，用于性能对比 ──
__global__ void matmul_tiled(const float *a, const float *b, float *c, int m,
                             int n, int k) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;

    float sum = 0.0f;
    int num_tiles = (k + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        if (row < m && a_col < k)
            sA[threadIdx.y][threadIdx.x] = a[row * k + a_col];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        int b_row = t * TILE + threadIdx.y;
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

void matmul_cpu(const float *a, const float *b, float *c, int m, int n, int k) {
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

/*
 * row-major GEMM: C[M×N] = alpha * A[M×K] * B[K×N] + beta * C
 *
 * cuBLAS 原生 col-major，等价变换：
 *   C^T = B^T * A^T
 * 调用时使用 OP_T + OP_T，并交换 m/n 角色。
 *
 * 数据仍按 row-major 存放，lda/ldb/ldc 取 row-major 的「行宽」。
 */
void cublas_gemm_rowmajor(cublasHandle_t handle, int m, int n, int k,
                          const float *d_A, const float *d_B, float *d_C,
                          float alpha, float beta) {
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                             d_B, n, d_A, k, &beta, d_C, n));
}

bool verify(const float *ref, const float *got, int count, float tol) {
    float max_err = 0.0f;
    for (int i = 0; i < count; ++i) {
        float err = fabsf(got[i] - ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("  max_err = %.6f %s\n", max_err, max_err < tol ? "(OK)" : "(FAIL)");
    return max_err < tol;
}

float benchmark_tiled(const float *d_A, const float *d_B, float *d_C, int m,
                      int n, int k, int warmup, int repeats) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);

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

float benchmark_cublas(cublasHandle_t handle, const float *d_A,
                       const float *d_B, float *d_C, int m, int n, int k,
                       int warmup, int repeats) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int i = 0; i < warmup; ++i) {
        cublas_gemm_rowmajor(handle, m, n, k, d_A, d_B, d_C, alpha, beta);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        cublas_gemm_rowmajor(handle, m, n, k, d_A, d_B, d_C, alpha, beta);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

int main() {
    const int count_A = M * K;
    const int count_B = K * N;
    const int count_C = M * N;

    printf("=== cuBLAS 入门：SGEMM ===\n");
    printf("C[%d×%d] = A[%d×%d] × B[%d×%d]  (row-major 存储)\n\n", M, N, M, K, K,
           N);

    float *h_A = (float *)malloc(count_A * sizeof(float));
    float *h_B = (float *)malloc(count_B * sizeof(float));
    float *h_C = (float *)malloc(count_C * sizeof(float));
    float *h_ref = (float *)malloc(count_C * sizeof(float));
    float *h_out = (float *)malloc(count_C * sizeof(float));

    srand(42);
    for (int i = 0; i < count_A; ++i) h_A[i] = (float)(rand() % 200 - 100) / 50.0f;
    for (int i = 0; i < count_B; ++i) h_B[i] = (float)(rand() % 200 - 100) / 50.0f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, count_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, count_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_C, count_C * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, count_A * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, count_B * sizeof(float),
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    printf("── 1. CPU 参考 ──\n");
    matmul_cpu(h_A, h_B, h_ref, M, N, K);

    printf("── 2. cuBLAS 验证 ──\n");
    cublas_gemm_rowmajor(handle, M, N, K, d_A, d_B, d_C, 1.0f, 0.0f);
    CUDA_CHECK(cudaMemcpy(h_out, d_C, count_C * sizeof(float),
                          cudaMemcpyDeviceToHost));
    verify(h_ref, h_out, count_C, 1e-2f);

    printf("── 3. lesson03 tiled 验证 ──\n");
    matmul_tiled<<<dim3((N + TILE - 1) / TILE, (M + TILE - 1) / TILE),
                   dim3(TILE, TILE)>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaMemcpy(h_out, d_C, count_C * sizeof(float),
                          cudaMemcpyDeviceToHost));
    verify(h_ref, h_out, count_C, 1e-2f);

    printf("── 4. 性能对比 ──\n");
    float t_tiled = benchmark_tiled(d_A, d_B, d_C, M, N, K, 2, 20);
    float t_cublas = benchmark_cublas(handle, d_A, d_B, d_C, M, N, K, 2, 20);
    double flops = 2.0 * M * N * K;
    printf("  lesson03 tiled : %.3f ms  (%.1f GFLOPS)\n", t_tiled,
           flops / t_tiled / 1e6);
    printf("  cuBLAS sgemm   : %.3f ms  (%.1f GFLOPS)\n", t_cublas,
           flops / t_cublas / 1e6);
    printf("  加速比 (cuBLAS / tiled): %.2fx\n", t_tiled / t_cublas);

    printf("\n── cuBLAS 核心 API 速查 ──\n");
    printf("  cublasCreate(&handle)          创建句柄\n");
    printf("  cublasDestroy(handle)          销毁句柄\n");
    printf("  cublasSetStream(handle, stream) 绑定 CUDA stream\n");
    printf("  cublasSgemm(handle, OP_N, OP_N, n, m, k,\n");
    printf("              &alpha, d_B, n, d_A, k, &beta, d_C, n)\n");
    printf("  row-major: 内存布局等同 col-major 的 C^T=B^T*A^T\n");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);
    free(h_out);
    return 0;
}
