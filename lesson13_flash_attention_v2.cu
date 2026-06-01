/*
 * FlashAttention v2 — FA-2 Algorithm 1
 * nvcc -O3 -arch=sm_86 -o lesson13 lesson13_flash_attention_v2.cu
 * ./lesson13 [seq]
 */

#include <stdio.h>
#include <stdlib.h>
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
#define SMEM_O_TILE_BYTES ((size_t)BR * DIM * sizeof(float))
#define SMEM_EXTENDED_MAX 98304

static size_t smem_bytes_base(void) {
    return (BR * TD + BC * TD + BC * TD + BR * BC) * sizeof(float);
}

static size_t smem_bytes_o_tile(void) {
    return smem_bytes_base() + SMEM_O_TILE_BYTES;
}

static bool can_use_smem_o(const cudaDeviceProp *prop) {
    return prop->major >= 8 && smem_bytes_o_tile() <= SMEM_EXTENDED_MAX;
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

__global__ void flash_attention_v2_smem(const float *Q, const float *K,
                                          const float *V, float *O, float *L,
                                          int seq, int dim) {
    const int q_start = blockIdx.x * BR;
    if (q_start >= seq) return;

    const int br = min(BR, seq - q_start);
    const int tid = threadIdx.x;
    const float scale = rsqrtf((float)dim);

    __shared__ float m_i[BR];
    __shared__ float l_i[BR];
    __shared__ float row_alpha[BR];

    extern __shared__ float smem[];
    float *Qs = smem;
    float *Ks = Qs + BR * TD;
    float *Vs = Ks + BC * TD;
    float *S = Vs + BC * TD;
    float *Os = S + BR * BC;

    for (int idx = tid; idx < br * dim; idx += blockDim.x) {
        Os[idx] = 0.0f;
    }
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

            for (int idx = tid; idx < br * td_size; idx += blockDim.x) {
                const int r = idx / td_size;
                const int t = idx % td_size;
                float pv = 0.0f;
                for (int c = 0; c < bc; ++c) {
                    pv += S[r * BC + c] * Vs[c * TD + t];
                }
                Os[r * dim + td + t] =
                    row_alpha[r] * Os[r * dim + td + t] + pv;
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < br * dim; idx += blockDim.x) {
        const int r = idx / dim;
        const int d = idx % dim;
        const int row = q_start + r;
        O[(size_t)row * dim + d] = Os[idx] / l_i[r];
    }
    for (int r = tid; r < br; r += blockDim.x) {
        const int row = q_start + r;
        L[row] = m_i[r] + logf(l_i[r]);
    }
}

__global__ __launch_bounds__(BLOCK, 2) void flash_attention_v2_reg(
    const float *Q, const float *K, const float *V, float *O, float *L,
    int seq, int dim) {
    const int q_start = blockIdx.x * BR;
    if (q_start >= seq) return;

    const int br = min(BR, seq - q_start);
    const int tid = threadIdx.x;
    const float scale = rsqrtf((float)dim);

    float o_acc[O_CHUNK];
    for (int k = 0; k < O_CHUNK; ++k) {
        o_acc[k] = 0.0f;
    }

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

static void configure_v2_smem_kernel(size_t smem_o) {
    CUDA_CHECK(cudaFuncSetAttribute(
        flash_attention_v2_smem, cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_o));
}

static void launch_v2(const float *d_Q, const float *d_K, const float *d_V,
                      float *d_out, float *d_L, int seq, int dim, bool use_smem) {
    const int grid = (seq + BR - 1) / BR;
    if (use_smem) {
        const size_t smem_o = smem_bytes_o_tile();
        configure_v2_smem_kernel(smem_o);
        flash_attention_v2_smem<<<grid, BLOCK, smem_o>>>(
            d_Q, d_K, d_V, d_out, d_L, seq, dim);
    } else {
        flash_attention_v2_reg<<<grid, BLOCK, smem_bytes_base()>>>(
            d_Q, d_K, d_V, d_out, d_L, seq, dim);
    }
}

static bool verify(const float *gpu, const float *cpu, int n, float tol) {
    for (int i = 0; i < n; ++i) {
        if (fabsf(gpu[i] - cpu[i]) > tol) return false;
    }
    return true;
}

static float benchmark_v2(const float *d_Q, const float *d_K, const float *d_V,
                          float *d_out, float *d_L, int seq, int dim,
                          bool use_smem, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, (size_t)seq * dim * sizeof(float)));
        launch_v2(d_Q, d_K, d_V, d_out, d_L, seq, dim, use_smem);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, (size_t)seq * dim * sizeof(float)));
        launch_v2(d_Q, d_K, d_V, d_out, d_L, seq, dim, use_smem);
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

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const bool use_smem = can_use_smem_o(&prop);

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
        attention_cpu(h_Q, h_K, h_V, h_ref, seq, DIM);
    }

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    launch_v2(d_Q, d_K, d_V, d_out, d_L, seq, DIM, use_smem);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    if (!skip_cpu && !verify(h_out, h_ref, N, 1e-2f)) {
        fprintf(stderr, "验证失败\n");
        return 1;
    }

    const float t_v2 =
        benchmark_v2(d_Q, d_K, d_V, d_out, d_L, seq, DIM, use_smem, 2, 10);
    printf("FlashAttention v2 耗时: %.3f ms\n", t_v2);

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
