/*
 * 阶段 5 - 练习 6：FlashAttention v2 — 论文 FA-2 Algorithm 1
 *
 * 论文循环顺序（Dao 2023）：
 *   for i = 1..Tr:           外层 Q tile（grid block = Q_i）
 *     初始化 O_i, m_i, ℓ_i 在片上
 *     for j = 1..Tc:         内层 KV tile
 *       S_ij = Q_i K_j^T → online softmax → 更新 O_i（片上）
 *     最后写 O_i, L_i = m_i + log(ℓ_i) 到 HBM（各一次）
 *
 * 相对 FA-1 (lesson12)：
 *   - 循环顺序对调：外 i 内 j
 *   - O/m/ℓ 不在每个 j 读写 HBM（O 在寄存器 o_acc[] 跨 j 累积）
 *
 * 编译：nvcc -O3 -o lesson13 lesson13_flash_attention_v2.cu
 * 运行：./lesson13 [seq]
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
#define O_CHUNK ((BR * DIM + BLOCK - 1) / BLOCK)

static size_t smem_bytes(void) {
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

/*
 * FA-2 Algorithm 1：外 i (blockIdx.x = Q tile)，内 j (KV loop)
 * O_i 在 o_acc[] 跨 j 累积；m_i, ℓ_i 在 smem；j 结束后写 O, L
 */
__global__ void flash_attention_v2(const float *Q, const float *K, const float *V,
                                   float *O, float *L, int seq, int dim) {
    const int q_start = blockIdx.x * BR;
    if (q_start >= seq) return;

    const int br = min(BR, seq - q_start);
    const int tid = threadIdx.x;
    const float scale = rsqrtf((float)dim);

    /* 论文：O_i^(0)=0 — 寄存器累积，j 循环内不写 HBM */
    float o_acc[O_CHUNK];
    for (int k = 0; k < O_CHUNK; ++k) {
        o_acc[k] = 0.0f;
    }

    /* 论文：m_i^(0)=-∞, ℓ_i^(0)=0 — 片上 smem，j 循环内不写 HBM */
    __shared__ float m_i[BR];
    __shared__ float l_i[BR];
    __shared__ float row_alpha[BR];

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
    for (int j = 0; j < Tc; ++j) {
        const int kv_start = j * BC;
        const int bc = min(BC, seq - kv_start);

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

        for (int r = tid; r < br; r += blockDim.x) {
            float m_ij = -FLT_MAX;
            for (int c = 0; c < bc; ++c) {
                m_ij = fmaxf(m_ij, S[r * BC + c]);
            }

            const float m_old = m_i[r];
            const float l_old = l_i[r];
            const float m_new = fmaxf(m_old, m_ij);
            const float alpha = expf(m_old - m_new);

            float l_ij = 0.0f;
            for (int c = 0; c < bc; ++c) {
                const float p = expf(S[r * BC + c] - m_new);
                S[r * BC + c] = p;
                l_ij += p;
            }

            row_alpha[r] = alpha;
            m_i[r] = m_new;
            l_i[r] = alpha * l_old + l_ij;
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

            for (int k = 0; k < O_CHUNK; ++k) {
                const int flat = tid + k * blockDim.x;
                if (flat >= br * dim) continue;
                const int d = flat % dim;
                if (d < td || d >= td + td_size) continue;
                const int r = flat / dim;
                const int t = d - td;
                float pv = 0.0f;
                for (int c = 0; c < bc; ++c) {
                    pv += S[r * BC + c] * Vs[c * TD + t];
                }
                o_acc[k] = row_alpha[r] * o_acc[k] + pv;
            }
            __syncthreads();
        }
    }

    /* 论文：最后写 O_i = O/ℓ_i，L_i = m_i + log(ℓ_i)（backward 用） */
    for (int k = 0; k < O_CHUNK; ++k) {
        const int flat = tid + k * blockDim.x;
        if (flat >= br * dim) continue;
        const int r = flat / dim;
        const int d = flat % dim;
        const int row = q_start + r;
        O[(size_t)row * dim + d] = o_acc[k] / l_i[r];
    }
    for (int r = tid; r < br; r += blockDim.x) {
        const int row = q_start + r;
        L[row] = m_i[r] + logf(l_i[r]);
    }
}

bool verify(const float *gpu, const float *cpu, int n, float tol) {
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_err = fmaxf(max_err, fabsf(gpu[i] - cpu[i]));
    }
    printf("  max_err = %.6f %s\n", max_err, max_err < tol ? "(OK)" : "(FAIL)");
    return max_err < tol;
}

float benchmark_v2(const float *d_Q, const float *d_K, const float *d_V,
                   float *d_out, float *d_L, int seq, int dim, size_t smem,
                   int warmup, int repeats) {
    const int grid = (seq + BR - 1) / BR;

    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, (size_t)seq * dim * sizeof(float)));
        flash_attention_v2<<<grid, BLOCK, smem>>>(d_Q, d_K, d_V, d_out, d_L,
                                                  seq, dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, (size_t)seq * dim * sizeof(float)));
        flash_attention_v2<<<grid, BLOCK, smem>>>(d_Q, d_K, d_V, d_out, d_L,
                                                  seq, dim);
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
            fprintf(stderr, "用法: %s [seq]\n", argv[0]);
            return 1;
        }
    }
    if (seq > 4096) skip_cpu = true;

    const int N = seq * DIM;
    const size_t bytes = (size_t)N * sizeof(float);
    const size_t smem = smem_bytes();
    const int Tr = (seq + BR - 1) / BR;
    const int Tc = (seq + BC - 1) / BC;

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("FlashAttention v2 — 论文 FA-2 Algorithm 1 (单头)\n");
    printf("SEQ=%d, DIM=%d, BR=%d, BC=%d\n", seq, DIM, BR, BC);
    printf("循环: 外层 i=1..Tr(%d) [grid], 内层 j=1..Tc(%d) [kernel loop]\n",
           Tr, Tc);
    printf("GPU: %s, smem/block=%.1f KB\n\n", prop.name, smem / 1024.0f);

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

    float *d_Q, *d_K, *d_V, *d_out, *d_L;
    CUDA_CHECK(cudaMalloc((void **)&d_Q, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_K, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_V, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_L, (size_t)seq * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));

    if (!skip_cpu) {
        printf("CPU 参考计算中...\n");
        attention_cpu(h_Q, h_K, h_V, h_ref, seq, DIM);
    }

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    flash_attention_v2<<<Tr, BLOCK, smem>>>(d_Q, d_K, d_V, d_out, d_L, seq,
                                            DIM);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    if (!skip_cpu) {
        printf("FlashAttention v2 验证: %s\n",
               verify(h_out, h_ref, N, 1e-2f) ? "通过" : "失败");
    }

    const float t_v2 =
        benchmark_v2(d_Q, d_K, d_V, d_out, d_L, seq, DIM, smem, 2, 10);
    printf("FlashAttention v2 耗时: %.3f ms\n", t_v2);

    printf("\n── FA-2 论文要点 ──\n");
    printf("  外 i 内 j；O/m/ℓ 在 j 循环内留片上\n");
    printf("  j 结束后写 O_i 与 L_i=m_i+log(ℓ_i) 各一次\n");
    printf("  相对 FA-1: 减少 O_i, ℓ_i 的 HBM 往返\n");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_L));
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_out);
    free(h_ref);
    return 0;
}
