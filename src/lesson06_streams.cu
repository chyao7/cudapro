/*
 * 阶段 4 - 练习 2：CUDA Stream 异步流水线
 *
 * 对比三种方式处理 NUM_BATCHES 批数据：
 *   1. 同步串行：cudaMemcpy + kernel + cudaMemcpy（lesson00 风格）
 *   2. 异步单 stream：cudaMemcpyAsync，但仍顺序等待
 *   3. 双 stream 流水线：拷贝与计算重叠
 *
 * 关键依赖：锁页内存 cudaMallocHost（普通 malloc 无法真正异步 DMA）
 *
 * 编译：nvcc -o lesson06 lesson06_streams.cu
 * 运行：./lesson06
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

#define BATCH (1 << 22)  // 每批 4M 元素 ≈ 16 MB
#define NUM_BATCHES 8
#define THREADS 256

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

struct BatchBuffers {
    float *h_a[NUM_BATCHES];
    float *h_b[NUM_BATCHES];
    float *h_c[NUM_BATCHES];
    float *d_a[2];
    float *d_b[2];
    float *d_c[2];
};

static int blocks_per_batch() { return (BATCH + THREADS - 1) / THREADS; }

static void init_batches(BatchBuffers *buf) {
    const size_t bytes = BATCH * sizeof(float);
    for (int i = 0; i < NUM_BATCHES; ++i) {
        CUDA_CHECK(cudaMallocHost((void **)&buf->h_a[i], bytes));
        CUDA_CHECK(cudaMallocHost((void **)&buf->h_b[i], bytes));
        CUDA_CHECK(cudaMallocHost((void **)&buf->h_c[i], bytes));
        for (int j = 0; j < BATCH; ++j) {
            buf->h_a[i][j] = (float)(i * 100 + j);
            buf->h_b[i][j] = (float)(j % 17);
        }
    }
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMalloc((void **)&buf->d_a[i], bytes));
        CUDA_CHECK(cudaMalloc((void **)&buf->d_b[i], bytes));
        CUDA_CHECK(cudaMalloc((void **)&buf->d_c[i], bytes));
    }
}

static void free_batches(BatchBuffers *buf) {
    for (int i = 0; i < NUM_BATCHES; ++i) {
        CUDA_CHECK(cudaFreeHost(buf->h_a[i]));
        CUDA_CHECK(cudaFreeHost(buf->h_b[i]));
        CUDA_CHECK(cudaFreeHost(buf->h_c[i]));
    }
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaFree(buf->d_a[i]));
        CUDA_CHECK(cudaFree(buf->d_b[i]));
        CUDA_CHECK(cudaFree(buf->d_c[i]));
    }
}

static bool verify_batches(const BatchBuffers *buf) {
    for (int i = 0; i < NUM_BATCHES; ++i) {
        for (int j = 0; j < BATCH; j += BATCH / 8) {
            float expected = buf->h_a[i][j] + buf->h_b[i][j];
            if (buf->h_c[i][j] != expected) {
                printf("验证失败 batch=%d j=%d: got=%f expected=%f\n", i, j,
                       buf->h_c[i][j], expected);
                return false;
            }
        }
    }
    return true;
}

// 方式 1：完全同步，每批阻塞等待
static float run_sync(const BatchBuffers *buf) {
    const size_t bytes = BATCH * sizeof(float);
    const int blocks = blocks_per_batch();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int b = 0; b < NUM_BATCHES; ++b) {
        CUDA_CHECK(cudaMemcpy(buf->d_a[0], buf->h_a[b], bytes,
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf->d_b[0], buf->h_b[b], bytes,
                               cudaMemcpyHostToDevice));
        vector_add<<<blocks, THREADS>>>(buf->d_a[0], buf->d_b[0], buf->d_c[0],
                                        BATCH);
        CUDA_CHECK(cudaMemcpy(buf->h_c[b], buf->d_c[0], bytes,
                               cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

// 方式 2：异步 API，但所有操作挤在同一个 stream，仍顺序执行
static float run_async_single_stream(const BatchBuffers *buf) {
    const size_t bytes = BATCH * sizeof(float);
    const int blocks = blocks_per_batch();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int b = 0; b < NUM_BATCHES; ++b) {
        CUDA_CHECK(cudaMemcpyAsync(buf->d_a[0], buf->h_a[b], bytes,
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(buf->d_b[0], buf->h_b[b], bytes,
                                    cudaMemcpyHostToDevice, stream));
        vector_add<<<blocks, THREADS, 0, stream>>>(buf->d_a[0], buf->d_b[0],
                                                   buf->d_c[0], BATCH);
        CUDA_CHECK(cudaMemcpyAsync(buf->h_c[b], buf->d_c[0], bytes,
                                    cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

// 方式 3：双 buffer + 双 stream 流水线
//   stream[cur]  上算当前 batch
//   stream[next] 上预取下一 batch 的 H2D，与计算重叠
static float run_pipeline(const BatchBuffers *buf) {
    const size_t bytes = BATCH * sizeof(float);
    const int blocks = blocks_per_batch();

    cudaStream_t stream[2];
    CUDA_CHECK(cudaStreamCreate(&stream[0]));
    CUDA_CHECK(cudaStreamCreate(&stream[1]));

    // 先把 batch 0 拷到 buffer 0
    CUDA_CHECK(cudaMemcpyAsync(buf->d_a[0], buf->h_a[0], bytes,
                                cudaMemcpyHostToDevice, stream[0]));
    CUDA_CHECK(cudaMemcpyAsync(buf->d_b[0], buf->h_b[0], bytes,
                                cudaMemcpyHostToDevice, stream[0]));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int b = 0; b < NUM_BATCHES; ++b) {
        int cur = b % 2;
        int next = (b + 1) % 2;

        // 在另一个 stream 上预取下一批（与当前 batch 计算并行）
        if (b + 1 < NUM_BATCHES) {
            CUDA_CHECK(cudaMemcpyAsync(buf->d_a[next], buf->h_a[b + 1], bytes,
                                        cudaMemcpyHostToDevice, stream[next]));
            CUDA_CHECK(cudaMemcpyAsync(buf->d_b[next], buf->h_b[b + 1], bytes,
                                        cudaMemcpyHostToDevice, stream[next]));
        }

        vector_add<<<blocks, THREADS, 0, stream[cur]>>>(
            buf->d_a[cur], buf->d_b[cur], buf->d_c[cur], BATCH);
        CUDA_CHECK(cudaMemcpyAsync(buf->h_c[b], buf->d_c[cur], bytes,
                                    cudaMemcpyDeviceToHost, stream[cur]));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream[0]));
    CUDA_CHECK(cudaStreamSynchronize(stream[1]));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaStreamDestroy(stream[0]));
    CUDA_CHECK(cudaStreamDestroy(stream[1]));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

int main() {
    BatchBuffers buf = {};

    printf("CUDA Stream 流水线实验\n");
    printf("批次数: %d, 每批: %d 元素 (%.1f MB), 总计: %.1f MB\n\n",
           NUM_BATCHES, BATCH, BATCH * sizeof(float) / (1024.0f * 1024.0f),
           NUM_BATCHES * BATCH * sizeof(float) * 3 / (1024.0f * 1024.0f));

    init_batches(&buf);

    float t_sync = run_sync(&buf);
    printf("1) 同步串行:       %.2f ms  (基准)\n", t_sync);
    printf("   验证: %s\n\n", verify_batches(&buf) ? "通过" : "失败");

    float t_async1 = run_async_single_stream(&buf);
    printf("2) 异步单 stream:  %.2f ms  (%.2fx vs 同步)\n", t_async1,
           t_sync / t_async1);
    printf("   验证: %s\n\n", verify_batches(&buf) ? "通过" : "失败");

    float t_pipe = run_pipeline(&buf);
    printf("3) 双 stream 流水线: %.2f ms  (%.2fx vs 同步)\n", t_pipe,
           t_sync / t_pipe);
    printf("   验证: %s\n\n", verify_batches(&buf) ? "通过" : "失败");

    printf("--- 解读 ---\n");
    printf("• 同步 vs 异步单 stream: 耗时接近，因为同一 stream 内仍顺序执行\n");
    printf("• 流水线加速来自: H2D(下一批) 与 kernel(当前批) 在不同 stream 重叠\n");
    printf("• 若加速不明显: kernel 太快或 GTX 1050 Copy Engine 与 SM 重叠有限\n");

    free_batches(&buf);
    return 0;
}
