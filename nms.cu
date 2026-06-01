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