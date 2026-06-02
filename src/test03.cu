#include <stdio.h>
#include <cuda_runtime.h>

// 定义边界框结构
typedef struct {
    float x1, y1, x2, y2, score;
} Box;

// CUDA核函数，用于执行NMS
__global__ void nms(Box *boxes, int *keep, int box_num, float nms_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= box_num) return;

    // 对边界框进行排序
    Box *sorted_boxes = boxes + blockIdx.x * blockDim.x;
    for (int i = 0; i < blockDim.x; ++i) {
        for (int j = i + 1; j < blockDim.x; ++j) {
            if (sorted_boxes[i].score < sorted_boxes[j].score) {
                Box temp = sorted_boxes[i];
                sorted_boxes[i] = sorted_boxes[j];
                sorted_boxes[j] = temp;
            }
        }
    }

    // 对排序后的边界框进行NMS
    for (int i = 0; i < blockDim.x; ++i) {
        if (i == threadIdx.x) continue;

        float inter_x1 = max(sorted_boxes[threadIdx.x].x1, sorted_boxes[i].x1);
        float inter_y1 = max(sorted_boxes[threadIdx.x].y1, sorted_boxes[i].y1);
        float inter_x2 = min(sorted_boxes[threadIdx.x].x2, sorted_boxes[i].x2);
        float inter_y2 = min(sorted_boxes[threadIdx.x].y2, sorted_boxes[i].y2);

        float inter_area = max(inter_x2 - inter_x1 + 1, 0.0f) * max(inter_y2 - inter_y1 + 1, 0.0f);

        float box_area1 = (sorted_boxes[threadIdx.x].x2 - sorted_boxes[threadIdx.x].x1 + 1) * (sorted_boxes[threadIdx.x].y2 - sorted_boxes[threadIdx.x].y1 + 1);
        float box_area2 = (sorted_boxes[i].x2 - sorted_boxes[i].x1 + 1) * (sorted_boxes[i].y2 - sorted_boxes[i].y1 + 1);

        float iou = inter_area / (box_area1 + box_area2 - inter_area);
        printf("Box %d vs Box %d: IOU = %f\n", threadIdx.x, i, iou);
        if (iou > nms_threshold) {
            atomicExch(&keep[i], 0);
        }
    }
}

int main() {
    // 假设我们有10个边界框
    int box_num = 1000;
    float nms_threshold = 0.5f;

    // 在主机上创建边界框数组
    Box *h_boxes = (Box *)malloc(box_num * sizeof(Box));
    // 初始化边界框数据
    for (int i = 0; i < box_num; ++i) {
        h_boxes[i].x1 = i * 10;
        h_boxes[i].y1 = i * 10;
        h_boxes[i].x2 = i * 10 + 100;
        h_boxes[i].y2 = i * 10 + 100;
        h_boxes[i].score = i * 0.1f;
    }

    // 在设备上创建边界框数组
    Box *d_boxes;
    cudaMalloc((void **)&d_boxes, box_num * sizeof(Box));
    cudaMemcpy(d_boxes, h_boxes, box_num * sizeof(Box), cudaMemcpyHostToDevice);

    // 在设备上创建keep数组
    int *d_keep;
    cudaMalloc((void **)&d_keep, box_num * sizeof(int));
    cudaMemset(d_keep, 1, box_num * sizeof(int));

    // 定义CUDA核函数的执行配置
    int threads_per_block = 256;
    int blocks_per_grid = (box_num + threads_per_block - 1) / threads_per_block;

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始时间
    cudaEventRecord(start, 0);

    // 调用CUDA核函数
    nms<<<blocks_per_grid, threads_per_block>>>(d_boxes, d_keep, box_num, nms_threshold);

    // 记录结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 打印执行时间
    printf("NMS kernel execution time: %f ms\n", milliseconds);

    // 释放CUDA事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 在主机上创建keep数组
    int *h_keep = (int *)malloc(box_num * sizeof(int));
    cudaMemcpy(h_keep, d_keep, box_num * sizeof(int), cudaMemcpyDeviceToHost);

    // 输出保留的边界框
    for (int i = 0; i < box_num; ++i) {
        if (h_keep[i] == 1) {
            printf("Box %d: (%f, %f) - (%f, %f), Score: %f\n", i, h_boxes[i].x1, h_boxes[i].y1, h_boxes[i].x2, h_boxes[i].y2, h_boxes[i].score);
        }
    }

    // 释放内存
    free(h_boxes);
    free(h_keep);
    cudaFree(d_boxes);
    cudaFree(d_keep);

    return 0;
}
