# cudapro

CUDA 入门到 LLM 算子实战的练习仓库：从 vector add、GEMM、RMSNorm、Attention 到 CUTLASS Tensor Core 与 QKV+RoPE 融合 kernel。

## 目录结构

```
cudapro/
├── src/                 # 课程 .cu（lesson00 … lesson17、工具样例）
├── benchmarks/          # 性能对比（PyTorch / flash 等）
├── include/             # 共享头文件
├── docs/                # 各 lesson 说明（如 lesson17）
├── cmake/               # CMake 模块
├── scripts/             # 配置与编译脚本
├── build/               # Debug 构建目录（本地生成，不入库）
└── build-release/       # Release 构建目录（本地生成）
```

可执行文件统一输出到：`<build-dir>/bin/<源文件名>`，例如 `build-release/bin/lesson17_qkv_rmsnorm`。

## 环境要求

- Linux + NVIDIA GPU（Ampere `sm_80+` 用于 lesson16/17 Tensor Core）
- CMake ≥ 3.18、CUDA Toolkit（建议 12+）
- [CUTLASS](https://github.com/NVIDIA/cutlass)（lesson17）：默认 `../cutlass`，或设置 `CUTLASS_DIR`
- PyTorch + CUDA（仅跑 `benchmarks/` 时需要）

## 快速开始

```bash
# 方式一：脚本（推荐）
./scripts/configure.sh release
./scripts/build.sh release lesson17_qkv_rmsnorm
./build-release/bin/lesson17_qkv_rmsnorm 4096

# 方式二：CMake Preset
cmake --preset release
cmake --build --preset release --target lesson17_qkv_rmsnorm -j
```

Debug 调试（cuda-gdb / VS Code）：

```bash
./scripts/configure.sh debug
./scripts/build.sh debug lesson00_vector_add
./build/bin/lesson00_vector_add
```

## 常用选项

| 变量 / 参数 | 说明 |
|-------------|------|
| `CUDAPRO_CUDA_ARCH` | GPU 算力，默认 `86`（3080 Ti 等） |
| `CUTLASS_DIR` | CUTLASS 根目录 |
| `-DCMAKE_BUILD_TYPE=Release` | 性能测试务必用 Release |

编译全部 target：`./scripts/build.sh release` 或 `cmake --build build-release --target all_cuda -j`。

## 与 PyTorch 对比（lesson17）

```bash
./scripts/bench_lesson17.sh 4096
# 或手动：
python3 benchmarks/bench_qkv_rmsnorm.py --tokens 4096 --dtype float16
./build-release/bin/lesson17_qkv_rmsnorm 4096
```

详见 [docs/lesson17_README.md](docs/lesson17_README.md)。

## VS Code

已配置 `.vscode/tasks.json`（Debug/Release 配置与编译）与 `launch.json`（cuda-gdb）。扩展建议见 `extensions.json`。
