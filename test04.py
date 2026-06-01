import os
import ctypes
import numpy as np



# 编译CUDA代码
# os.system("nvcc -c -o nms.o nms.cu")

# 加载编译后的CUDA代码
nms = ctypes.CDLL("./nsm.o")

# 定义CUDA核函数的参数类型
nms.nms.argtypes = [ctypes.POINTER(Box), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_float]

# 假设我们有10个边界框
box_num = 10
nms_threshold = 0.5

# 在主机上创建边界框数组
h_boxes = [Box(0, 0, 100, 100, 0.1), Box(10, 10, 110, 110, 0.2), Box(20, 20, 120, 120, 0.3), Box(30, 30, 130, 130, 0.4), Box(40, 40, 140, 140, 0.5), Box(50, 50, 150, 150, 0.6), Box(60, 60, 160, 160, 0.7), Box(70, 70, 170, 170, 0.8), Box(80, 80, 180, 180, 0.9), Box(90, 90, 190, 190, 1.0)]

# 在设备上创建边界框数组
d_boxes = np.zeros(box_num, dtype=np.dtype([('x1', np.float32), ('y1', np.float32), ('x2', np.float32), ('y2', np.float32), ('score', np.float32)]))
d_boxes['x1'] = [box.x1 for box in h_boxes]
d_boxes['y1'] = [box.y1 for box in h_boxes]
d_boxes['x2'] = [box.x2 for box in h_boxes]
d_boxes['y2'] = [box.y2 for box in h_boxes]
d_boxes['score'] = [box.score for box in h_boxes]

# 在设备上创建keep数组
d_keep = np.ones(box_num, dtype=np.int32)

# 调用CUDA核函数
nms.nms(d_boxes.ctypes.data_as(ctypes.POINTER(Box)), d_keep.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), box_num, nms_threshold)

# 输出保留的边界框
for i in range(box_num):
    if d_keep[i] == 1:
        print("Box %d: (%f, %f) - (%f, %f), Score: %f" % (i, h_boxes[i].x1, h_boxes[i].y1, h_boxes[i].x2, h_boxes[i].y2, h_boxes[i].score))
