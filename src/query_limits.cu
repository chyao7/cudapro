#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));    \
            return 1;                                                        \
        }                                                                    \
    } while (0)

int main() {
    int ndev = 0;
    cudaError_t err = cudaGetDeviceCount(&ndev);
    if (err != cudaSuccess || ndev == 0) {
        printf("未检测到 CUDA GPU: %s\n", cudaGetErrorString(err));
        return 1;
    }

    for (int d = 0; d < ndev; ++d) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, d));

        printf("=== GPU %d: %s ===\n", d, prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("\n--- Block (每个 block 的 thread 上限) ---\n");
        printf("maxThreadsDim[0/1/2]: (%d, %d, %d)\n", prop.maxThreadsDim[0],
               prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("maxThreadsPerBlock:  %d\n", prop.maxThreadsPerBlock);
        printf("\n--- Grid (grid 维度上限) ---\n");
        printf("maxGridSize[0/1/2]:   (%u, %u, %u)\n", prop.maxGridSize[0],
               prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n--- 其他常用限制 ---\n");
        printf("multiprocessorCount (SM数): %d\n", prop.multiProcessorCount);
        printf("warpSize:                   %d\n", prop.warpSize);
        printf("sharedMemPerBlock:          %.1f KB\n",
               prop.sharedMemPerBlock / 1024.0f);
        printf("sharedMemPerMultiprocessor: %.1f KB\n",
               prop.sharedMemPerMultiprocessor / 1024.0f);
        printf("maxBlocksPerMultiProcessor: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("totalGlobalMem:             %.2f GB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("\n--- 理论最大总 thread 数 (grid × block) ---\n");
        unsigned long long max_block =
            (unsigned long long)prop.maxThreadsDim[0] *
            prop.maxThreadsDim[1] * prop.maxThreadsDim[2];
        unsigned long long max_grid =
            (unsigned long long)prop.maxGridSize[0] * prop.maxGridSize[1] *
            prop.maxGridSize[2];
        printf("单 block 最多 thread: %llu\n", max_block);
        printf("grid 最多 block 数:   %llu\n", max_grid);
        printf("理论最大 thread 总数: %llu (实际受显存/任务限制)\n",
               max_block * max_grid);
        if (d + 1 < ndev) printf("\n");
    }
    return 0;
}
