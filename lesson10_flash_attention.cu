/*
 * 阶段 5 - 练习 4：FlashAttention 标准版（Baseline）
 *
 * 标准 Attention（三 kernel，8 次 HBM 读写）：
 *   S = Q K^T → P = softmax(S) → O = P V
 *   中间 S,P 各 O(N^2) 在 HBM
 *
 * 本文件：分块 + Kernel 融合，但 **不用 Online Softmax**（那是 lesson12 v1）
 *
 * 两遍 KV 扫描（Safe Softmax，全局 m_i 已知后再算 P）：
 *   Pass 1: 遍历所有 KV tile，S_ij = Q_i K_j^T，更新 m_i = max(m_i, rowmax(S_ij))
 *   Pass 2: 再遍历 KV tile，P_ij = exp(S_ij - m_i)，l_i += rowsum(P_ij)，O_i += P_ij V_j
 *   最后 O_i /= l_i
 *
 * IO 优化（相对标准 Attention）：
 *   - S_tile = [Br×Bc] 只在 SRAM，不写回 HBM
 *   - 单 kernel 融合 QK^T + softmax + P@V
 *   - smem 固定 ~28KB，与 seq 无关
 *
 * 代价：KV 扫两遍（v1 的 Online Softmax 只需一遍）
 *
 * 演进：lesson10=标准分块  lesson12=v1(Online Softmax)  lesson13=v2(warp)
 *
 * 编译：nvcc -O3 -o lesson10 lesson10_flash_attention.cu
 * 运行：./lesson10 [seq] [--gpu-only]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <float.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define DEFAULT_SEQ 2048
#define DIM 512
#define BR 32
#define BC 32
#define TD 64
#define BLOCK 256

static size_t flash_smem_bytes(void) {
    return (BR * TD + BC * TD + BC * TD + BR * BC) * sizeof(float);
}

static float dot_cpu(const float *a, const float *b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) s += a[i] * b[i];
    return s;
}

void attention_cpu(const float *Q, const float *K, const float *V, float *out,
                   int seq, int dim) {
    float *scores = (float *)malloc((size_t)seq * sizeof(float));
    const float scale = rsqrtf((float)dim);

    for (int s = 0; s < seq; ++s) {
        const float *q = Q + (size_t)s * dim;
        for (int t = 0; t < seq; ++t) {
            scores[t] = dot_cpu(q, K + (size_t)t * dim, dim) * scale;
        }

        float row_max = -FLT_MAX;
        for (int t = 0; t < seq; ++t) row_max = fmaxf(row_max, scores[t]);

        double row_sum = 0.0;
        for (int t = 0; t < seq; ++t) {
            scores[t] = expf(scores[t] - row_max);
            row_sum += scores[t];
        }
        for (int t = 0; t < seq; ++t) scores[t] /= (float)row_sum;

        float *o = out + (size_t)s * dim;
        for (int d = 0; d < dim; ++d) {
            double sum = 0.0;
            for (int t = 0; t < seq; ++t) {
                sum += (double)scores[t] * V[(size_t)t * dim + d];
            }
            o[d] = (float)sum;
        }
    }
    free(scores);
}

/* 计算片上 S[br×bc] = scale * Q_i K_j^T（d 分块累加） */
__device__ void compute_s_tile(const float *Q, const float *K, float *Qs,
                               float *Ks, float *S, int q_start, int kv_start,
                               int br, int bc, int dim, int tid, float scale) {
    for (int idx = tid; idx < br * BC; idx += blockDim.x) {
        S[idx] = 0.0f;
    }
    __syncthreads();

    for (int td = 0; td < dim; td += TD) {
        const int td_size = min(TD, dim - td);

        for (int idx = tid; idx < br * td_size; idx += blockDim.x) {
            const int r = idx / td_size;
            const int t = idx % td_size;
            Qs[r * TD + t] = Q[(q_start + r) * dim + td + t];
        }
        for (int idx = tid; idx < bc * td_size; idx += blockDim.x) {
            const int c = idx / td_size;
            const int t = idx % td_size;
            Ks[c * TD + t] = K[(kv_start + c) * dim + td + t];
        }
        __syncthreads();

        for (int idx = tid; idx < br * bc; idx += blockDim.x) {
            const int r = idx / bc;
            const int c = idx % bc;
            float dot = 0.0f;
            for (int t = 0; t < td_size; ++t) {
                dot += Qs[r * TD + t] * Ks[c * TD + t];
            }
            S[r * BC + c] += dot;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < br * bc; idx += blockDim.x) {
        S[idx] *= scale;
    }
    __syncthreads();
}

/*
 * 标准版 FlashAttention：两遍 KV 扫描，无 Online Softmax
 */
__global__ void flash_attention(const float *Q, const float *K, const float *V,
                                float *O, int seq, int dim) {
    const int q_start = blockIdx.x * BR;
    if (q_start >= seq) return;

    const int br = min(BR, seq - q_start);
    const int tid = threadIdx.x;
    const float scale = rsqrtf((float)dim);

    __shared__ float m_i[BR];
    __shared__ float l_i[BR];

    extern __shared__ float smem[];
    float *Qs = smem;
    float *Ks = Qs + BR * TD;
    float *Vs = Ks + BC * TD;
    float *S = Vs + BC * TD;

    if (tid < br) {
        m_i[tid] = -FLT_MAX;
        l_i[tid] = 0.0f;
    }
    __syncthreads();

    const int Tc = (seq + BC - 1) / BC;

    /* ── Pass 1: 全局 row max（Safe Softmax 第一步） ── */
    for (int j = 0; j < Tc; ++j) {
        const int kv_start = j * BC;
        const int bc = min(BC, seq - kv_start);

        compute_s_tile(Q, K, Qs, Ks, S, q_start, kv_start, br, bc, dim, tid,
                       scale);

        for (int r = tid; r < br; r += blockDim.x) {
            float m_ij = -FLT_MAX;
            for (int c = 0; c < bc; ++c) {
                m_ij = fmaxf(m_ij, S[r * BC + c]);
            }
            m_i[r] = fmaxf(m_i[r], m_ij);
        }
        __syncthreads();
    }

    /* ── Pass 2: 固定 m_i，累加 l_i 和 O_i ── */
    for (int j = 0; j < Tc; ++j) {
        const int kv_start = j * BC;
        const int bc = min(BC, seq - kv_start);

        compute_s_tile(Q, K, Qs, Ks, S, q_start, kv_start, br, bc, dim, tid,
                       scale);

        for (int r = tid; r < br; r += blockDim.x) {
            const float m_row = m_i[r];
            float l_add = 0.0f;
            for (int c = 0; c < bc; ++c) {
                const float p = expf(S[r * BC + c] - m_row);
                S[r * BC + c] = p;
                l_add += p;
            }
            l_i[r] += l_add;
        }
        __syncthreads();

        for (int td = 0; td < dim; td += TD) {
            const int td_size = min(TD, dim - td);

            for (int idx = tid; idx < bc * td_size; idx += blockDim.x) {
                const int c = idx / td_size;
                const int t = idx % td_size;
                Vs[c * TD + t] = V[(kv_start + c) * dim + td + t];
            }
            __syncthreads();

            for (int idx = tid; idx < br * td_size; idx += blockDim.x) {
                const int r = idx / td_size;
                const int t = idx % td_size;
                float pv = 0.0f;
                for (int c = 0; c < bc; ++c) {
                    pv += S[r * BC + c] * Vs[c * TD + t];
                }
                float *o = O + (q_start + r) * dim + td + t;
                *o += pv;
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < br * dim; idx += blockDim.x) {
        const int r = idx / dim;
        const int d = idx % dim;
        O[(q_start + r) * dim + d] /= l_i[r];
    }
}

bool verify(const float *gpu, const float *cpu, int n, float tol) {
    int bad = 0;
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float err = fabsf(gpu[i] - cpu[i]);
        if (err > max_err) max_err = err;
        if (err > tol) {
            if (bad < 5) {
                fprintf(stderr, "  mismatch[%d]: gpu=%.6f cpu=%.6f err=%.6f\n", i,
                        gpu[i], cpu[i], err);
            }
            ++bad;
        }
    }
    if (bad > 0) {
        fprintf(stderr, "  total mismatches: %d / %d, max_err=%.6f\n", bad, n,
                max_err);
        return false;
    }
    return true;
}

float benchmark_flash(const float *d_Q, const float *d_K, const float *d_V,
                      float *d_out, int seq, int dim, size_t smem_bytes,
                      int warmup, int repeats) {
    const int grid = (seq + BR - 1) / BR;

    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, (size_t)seq * dim * sizeof(float)));
        flash_attention<<<grid, BLOCK, smem_bytes>>>(d_Q, d_K, d_V, d_out, seq,
                                                     dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, (size_t)seq * dim * sizeof(float)));
        flash_attention<<<grid, BLOCK, smem_bytes>>>(d_Q, d_K, d_V, d_out, seq,
                                                     dim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeats;
}

int main(int argc, char **argv) {
    int seq = DEFAULT_SEQ;
    bool skip_cpu = false;
    if (argc > 1) {
        seq = atoi(argv[1]);
        if (seq <= 0) {
            fprintf(stderr, "用法: %s [seq] [--gpu-only]\n", argv[0]);
            return 1;
        }
    }
    if (argc > 2 && strcmp(argv[2], "--gpu-only") == 0) {
        skip_cpu = true;
    }
    if (seq > 4096) skip_cpu = true;

    const int N = seq * DIM;
    const size_t bytes = (size_t)N * sizeof(float);
    const size_t smem_bytes = flash_smem_bytes();

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("FlashAttention 标准版 — 分块融合，两遍 Safe Softmax (单头)\n");
    printf("SEQ=%d, DIM=%d, B_r=%d, B_c=%d, T_d=%d\n", seq, DIM, BR, BC, TD);
    printf("GPU: %s, smem/block=%.1f KB\n\n", prop.name,
           smem_bytes / 1024.0f);
    printf("── 算法 ──\n");
    printf("  Pass1: 扫 KV tile → 求全局 m_i = rowmax(S)\n");
    printf("  Pass2: 再扫 KV tile → P=exp(S-m_i), O+=PV, O/=l_i\n");
    printf("  无 Online Softmax（v1/lesson12 才引入，一遍 KV 即可）\n");
    printf("  S_tile=[%d×%d] 仅在 SRAM，不写 HBM\n\n", BR, BC);

    float *h_Q = (float *)malloc(bytes);
    float *h_K = (float *)malloc(bytes);
    float *h_V = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < N; ++i) {
        h_Q[i] = (float)(rand() % 200 - 100) / 50.0f;
        h_K[i] = (float)(rand() % 200 - 100) / 50.0f;
        h_V[i] = (float)(rand() % 200 - 100) / 50.0f;
    }

    float *d_Q, *d_K, *d_V, *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_Q, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_K, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_V, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));

    if (!skip_cpu) {
        printf("CPU 标准 Attention 参考计算中...\n");
        attention_cpu(h_Q, h_K, h_V, h_ref, seq, DIM);
    }

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    flash_attention<<<(seq + BR - 1) / BR, BLOCK, smem_bytes>>>(
        d_Q, d_K, d_V, d_out, seq, DIM);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    if (!skip_cpu) {
        printf("FlashAttention 标准版验证: %s\n",
               verify(h_out, h_ref, N, 1e-2f) ? "通过" : "失败");
    }

    const float t_flash =
        benchmark_flash(d_Q, d_K, d_V, d_out, seq, DIM, smem_bytes, 2, 10);
    printf("FlashAttention 标准版耗时: %.3f ms\n", t_flash);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out));
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_out);
    free(h_ref);
    return 0;
}
