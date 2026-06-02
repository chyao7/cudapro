/*
 * 阶段 5 - 练习 1：2D 卷积 (Conv2D)
 *
 * 综合前面所学：
 *   - 2D grid / block（lesson01）
 *   - shared memory + halo 加载（lesson02/03）
 *   - 计算密集型 kernel + 带宽对比
 *
 * 输入 in[H×W]，卷积核 kernel[K×K]，输出 out[H×W]（same padding）
 *
 * 编译：nvcc -o lesson07 lesson07_conv2d.cu
 * 运行：./lesson07
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

#define K 5          // 5×5 卷积核
#define R (K / 2)    // radius = 2
#define TILE 16      // 每个 block 处理 16×16 输出 tile

// 版本 1：naive — 每个 thread 直接读 global memory 做 K×K 次乘加
__global__ void conv2d_naive(const float *in, const float *kernel, float *out,
                             int h, int w) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= h || col >= w) return;

    float sum = 0.0f;
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            int ir = row + ky - R;
            int ic = col + kx - R;
            if (ir >= 0 && ir < h && ic >= 0 && ic < w) {
                sum += in[ir * w + ic] * kernel[ky * K + kx];
            }
        }
    }
    out[row * w + col] = sum;
}

// 版本 2：shared memory tiling — 协作加载 tile+halo，减少重复读 global
__global__ void conv2d_tiled(const float *in, const float *kernel, float *out,
                             int h, int w) {
    __shared__ float tile[TILE + 2 * R][TILE + 2 * R];

    int out_col = blockIdx.x * TILE + threadIdx.x;
    int out_row = blockIdx.y * TILE + threadIdx.y;
    int tile_col = threadIdx.x;
    int tile_row = threadIdx.y;

    // block 左上角在输入中的位置（含 halo 偏移 -R）
    int base_row = blockIdx.y * TILE - R;
    int base_col = blockIdx.x * TILE - R;

    // 协作加载 (TILE+2R) × (TILE+2R) 的 tile（含 halo）到 shared memory
    // 每个 thread 可能负责多个元素（当 tile 大于 block 时 stride 循环）
    for (int lr = tile_row; lr < TILE + 2 * R; lr += blockDim.y) {
        for (int lc = tile_col; lc < TILE + 2 * R; lc += blockDim.x) {
            int gr = base_row + lr;
            int gc = base_col + lc;
            if (gr >= 0 && gr < h && gc >= 0 && gc < w)
                tile[lr][lc] = in[gr * w + gc];
            else
                tile[lr][lc] = 0.0f;  // 边界外 zero padding
        }
    }
    __syncthreads();

    if (out_row >= h || out_col >= w) return;

    // 在 shared memory 里做卷积（tile 内坐标 = thread 坐标 + R）
    float sum = 0.0f;
    int lr = tile_row + R;
    int lc = tile_col + R;
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            sum += tile[lr + ky - R][lc + kx - R] * kernel[ky * K + kx];
        }
    }
    out[out_row * w + out_col] = sum;
}

void conv2d_cpu(const float *in, const float *kernel, float *out, int h,
                int w) {
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            float sum = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    int ir = row + ky - R;
                    int ic = col + kx - R;
                    if (ir >= 0 && ir < h && ic >= 0 && ic < w) {
                        sum += in[ir * w + ic] * kernel[ky * K + kx];
                    }
                }
            }
            out[row * w + col] = sum;
        }
    }
}

float benchmark(void (*kernel)(const float *, const float *, float *, int, int),
                const float *d_in, const float *d_kernel, float *d_out, int h,
                int w, dim3 grid, dim3 block, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) {
        kernel<<<grid, block>>>(d_in, d_kernel, d_out, h, w);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        kernel<<<grid, block>>>(d_in, d_kernel, d_out, h, w);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

bool verify(const float *gpu, const float *cpu, int n, float tol) {
    for (int i = 0; i < n; ++i) {
        if (fabsf(gpu[i] - cpu[i]) > tol) return false;
    }
    return true;
}

int main() {
    const int H = 2048;
    const int W = 2048;
    const int N = H * W;
    const size_t bytes = N * sizeof(float);
    const size_t kbytes = K * K * sizeof(float);

    float h_kernel[K * K];
    for (int i = 0; i < K * K; ++i) {
        h_kernel[i] = 1.0f / (K * K);  // 均值滤波
    }

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    for (int i = 0; i < N; ++i) h_in[i] = (float)(i % 256);

    float *d_in, *d_out, *d_kernel;
    CUDA_CHECK(cudaMalloc((void **)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_kernel, kbytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kbytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);

    printf("2D 卷积: %d × %d, 卷积核 %d×%d (same padding)\n", H, W, K, K);
    printf("block: (%d,%d), grid: (%d,%d)\n", TILE, TILE, grid.x, grid.y);
    printf("tiled shared tile: (%d×%d) = %.1f KB\n\n", TILE + 2 * R,
           TILE + 2 * R, (TILE + 2 * R) * (TILE + 2 * R) * sizeof(float) / 1024.0f);

    conv2d_naive<<<grid, block>>>(d_in, d_kernel, d_out, H, W);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    conv2d_cpu(h_in, h_kernel, h_ref, H, W);
    printf("naive  验证: %s\n", verify(h_out, h_ref, N, 1e-3f) ? "通过" : "失败");

    float t_naive = benchmark(conv2d_naive, d_in, d_kernel, d_out, H, W, grid,
                              block, 3, 20);
    printf("naive  耗时: %.3f ms\n\n", t_naive);

    conv2d_tiled<<<grid, block>>>(d_in, d_kernel, d_out, H, W);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("tiled  验证: %s\n", verify(h_out, h_ref, N, 1e-3f) ? "通过" : "失败");

    float t_tiled = benchmark(conv2d_tiled, d_in, d_kernel, d_out, H, W, grid,
                              block, 3, 20);
    printf("tiled  耗时: %.3f ms\n", t_tiled);
    printf("加速比: %.1fx\n", t_naive / t_tiled);

    double ops = (double)N * K * K * 2;  // 每像素 K² 乘加
    printf("tiled 算力: %.1f GFLOPS\n", ops / (t_tiled * 1e6));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_kernel));
    free(h_in);
    free(h_out);
    free(h_ref);
    return 0;
}
