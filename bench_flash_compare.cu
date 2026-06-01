/*
 * FlashAttention 性能对比：标准版(baseline) / 优化 v1 / 优化 v2
 *
 * 同一 GPU、同一 Q/K/V 数据、相同 warmup/repeats，公平对比三种 kernel。
 *
 * 编译：nvcc -O3 -o bench_flash bench_flash_compare.cu
 * 运行：./bench_flash
 *       ./bench_flash 512 1024 2048
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

#define DIM 512
#define BR 32
#define BC 32
#define TD 64
#define BLOCK 256
#define O_CHUNK ((BR * DIM + BLOCK - 1) / BLOCK)
#define WARMUP 3
#define REPEATS 20
#define VERIFY_TOL 1e-2f

static size_t smem_bytes(void) {
    return (BR * TD + BC * TD + BC * TD + BR * BC) * sizeof(float);
}

// ═══════════════════ lesson10 baseline: 两遍 Safe Softmax，无 Online ═══════════════════

__device__ void l10_compute_s_tile(const float *Q, const float *K, float *Qs,
                                   float *Ks, float *S, int q_start,
                                   int kv_start, int br, int bc, int dim,
                                   int tid, float scale) {
    for (int idx = tid; idx < br * BC; idx += blockDim.x) S[idx] = 0.0f;
    __syncthreads();
    for (int td = 0; td < dim; td += TD) {
        const int td_size = min(TD, dim - td);
        for (int idx = tid; idx < br * td_size; idx += blockDim.x) {
            const int r = idx / td_size, t = idx % td_size;
            Qs[r * TD + t] = Q[(q_start + r) * dim + td + t];
        }
        for (int idx = tid; idx < bc * td_size; idx += blockDim.x) {
            const int c = idx / td_size, t = idx % td_size;
            Ks[c * TD + t] = K[(kv_start + c) * dim + td + t];
        }
        __syncthreads();
        for (int idx = tid; idx < br * bc; idx += blockDim.x) {
            const int r = idx / bc, c = idx % bc;
            float dot = 0.0f;
            for (int t = 0; t < td_size; ++t)
                dot += Qs[r * TD + t] * Ks[c * TD + t];
            S[r * BC + c] += dot;
        }
        __syncthreads();
    }
    for (int idx = tid; idx < br * bc; idx += blockDim.x) S[idx] *= scale;
    __syncthreads();
}

__global__ void flash_attention_l10(const float *Q, const float *K,
                                    const float *V, float *O, int seq,
                                    int dim) {
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
    for (int j = 0; j < Tc; ++j) {
        const int kv_start = j * BC;
        const int bc = min(BC, seq - kv_start);
        l10_compute_s_tile(Q, K, Qs, Ks, S, q_start, kv_start, br, bc, dim, tid,
                           scale);
        for (int r = tid; r < br; r += blockDim.x) {
            float m_ij = -FLT_MAX;
            for (int c = 0; c < bc; ++c) m_ij = fmaxf(m_ij, S[r * BC + c]);
            m_i[r] = fmaxf(m_i[r], m_ij);
        }
        __syncthreads();
    }

    for (int j = 0; j < Tc; ++j) {
        const int kv_start = j * BC;
        const int bc = min(BC, seq - kv_start);
        l10_compute_s_tile(Q, K, Qs, Ks, S, q_start, kv_start, br, bc, dim, tid,
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
                const int c = idx / td_size, t = idx % td_size;
                Vs[c * TD + t] = V[(kv_start + c) * dim + td + t];
            }
            __syncthreads();
            for (int idx = tid; idx < br * td_size; idx += blockDim.x) {
                const int r = idx / td_size, t = idx % td_size;
                float pv = 0.0f;
                for (int c = 0; c < bc; ++c) pv += S[r * BC + c] * Vs[c * TD + t];
                O[(q_start + r) * dim + td + t] += pv;
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < br * dim; idx += blockDim.x) {
        const int r = idx / dim, d = idx % dim;
        O[(q_start + r) * dim + d] /= l_i[r];
    }
}

// ═══════════════════ lesson12 FA-1：外 j 内 i，HBM 读写 m/ℓ/O ═══════════════════

__device__ void fa_v1_online_softmax_row(float *S, int r, int bc, float m_row,
                                        float *m_new_out, float *l_row_inout,
                                        float *scale_m_out,
                                        float *scale_ij_out) {
    float m_ij = -FLT_MAX;
    for (int c = 0; c < bc; ++c) m_ij = fmaxf(m_ij, S[r * BC + c]);
    float l_ij = 0.0f;
    for (int c = 0; c < bc; ++c) {
        const float p = expf(S[r * BC + c] - m_ij);
        S[r * BC + c] = p;
        l_ij += p;
    }
    const float m_new = fmaxf(m_row, m_ij);
    const float scale_m = expf(m_row - m_new);
    const float scale_ij = expf(m_ij - m_new);
    *m_new_out = m_new;
    *scale_m_out = scale_m;
    *scale_ij_out = scale_ij;
    *l_row_inout = scale_m * (*l_row_inout) + scale_ij * l_ij;
}

__global__ void flash_attention_v1_step(const float *Q, const float *K,
                                        const float *V, float *M, float *L,
                                        float *O, int j, int seq, int dim) {
    const int q_start = blockIdx.x * BR;
    if (q_start >= seq) return;
    const int kv_start = j * BC;
    if (kv_start >= seq) return;

    const int br = min(BR, seq - q_start);
    const int bc = min(BC, seq - kv_start);
    const int tid = threadIdx.x;
    const float scale = rsqrtf((float)dim);

    __shared__ float m_tile[BR];
    __shared__ float l_tile[BR];
    __shared__ float scale_m[BR];
    __shared__ float scale_ij[BR];

    extern __shared__ float smem[];
    float *Qs = smem;
    float *Ks = Qs + BR * TD;
    float *Vs = Ks + BC * TD;
    float *S = Vs + BC * TD;

    for (int r = tid; r < br; r += blockDim.x) {
        const int row = q_start + r;
        m_tile[r] = M[row];
        l_tile[r] = L[row];
    }
    __syncthreads();

    for (int idx = tid; idx < br * BC; idx += blockDim.x) S[idx] = 0.0f;
    __syncthreads();

    for (int td = 0; td < dim; td += TD) {
        const int td_size = min(TD, dim - td);
        for (int idx = tid; idx < br * td_size; idx += blockDim.x) {
            const int r = idx / td_size, t = idx % td_size;
            Qs[r * TD + t] = Q[(q_start + r) * dim + td + t];
        }
        for (int idx = tid; idx < bc * td_size; idx += blockDim.x) {
            const int c = idx / td_size, t = idx % td_size;
            Ks[c * TD + t] = K[(kv_start + c) * dim + td + t];
        }
        __syncthreads();
        for (int idx = tid; idx < br * bc; idx += blockDim.x) {
            const int r = idx / bc, c = idx % bc;
            float dot = 0.0f;
            for (int t = 0; t < td_size; ++t)
                dot += Qs[r * TD + t] * Ks[c * TD + t];
            S[r * BC + c] += dot;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < br * bc; idx += blockDim.x) S[idx] *= scale;
    __syncthreads();

    for (int r = tid; r < br; r += blockDim.x) {
        float m_new, sm, sij;
        fa_v1_online_softmax_row(S, r, bc, m_tile[r], &m_new, &l_tile[r], &sm,
                                 &sij);
        m_tile[r] = m_new;
        scale_m[r] = sm;
        scale_ij[r] = sij;
    }
    __syncthreads();

    for (int td = 0; td < dim; td += TD) {
        const int td_size = min(TD, dim - td);
        for (int idx = tid; idx < bc * td_size; idx += blockDim.x) {
            const int c = idx / td_size, t = idx % td_size;
            Vs[c * TD + t] = V[(kv_start + c) * dim + td + t];
        }
        __syncthreads();
        for (int idx = tid; idx < br * td_size; idx += blockDim.x) {
            const int r = idx / td_size, t = idx % td_size;
            float pv = 0.0f;
            for (int c = 0; c < bc; ++c) pv += S[r * BC + c] * Vs[c * TD + t];
            float *o = O + (q_start + r) * dim + td + t;
            *o = scale_m[r] * (*o) + scale_ij[r] * pv;
        }
        __syncthreads();
    }

    for (int r = tid; r < br; r += blockDim.x) {
        const int row = q_start + r;
        M[row] = m_tile[r];
        L[row] = l_tile[r];
    }
}

__global__ void flash_attention_v1_finalize(float *O, const float *L, int seq,
                                          int dim) {
    const int q_start = blockIdx.x * BR;
    if (q_start >= seq) return;
    const int br = min(BR, seq - q_start);
    const int tid = threadIdx.x;
    for (int idx = tid; idx < br * dim; idx += blockDim.x) {
        const int r = idx / dim, d = idx % dim;
        O[(q_start + r) * dim + d] /= L[q_start + r];
    }
}

static void init_m_l_gpu(float *d_M, float *d_L, int seq) {
    float *h_M = (float *)malloc((size_t)seq * sizeof(float));
    for (int i = 0; i < seq; ++i) h_M[i] = -FLT_MAX;
    CUDA_CHECK(cudaMemcpy(d_M, h_M, (size_t)seq * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_L, 0, (size_t)seq * sizeof(float)));
    free(h_M);
}

static void launch_fa1(const float *d_Q, const float *d_K, const float *d_V,
                       float *d_M, float *d_L, float *d_O, int seq, int dim) {
    const int Tr = (seq + BR - 1) / BR;
    const int Tc = (seq + BC - 1) / BC;
    const size_t smem = smem_bytes();
    for (int j = 0; j < Tc; ++j) {
        flash_attention_v1_step<<<Tr, BLOCK, smem>>>(d_Q, d_K, d_V, d_M, d_L,
                                                      d_O, j, seq, dim);
    }
    flash_attention_v1_finalize<<<Tr, BLOCK>>>(d_O, d_L, seq, dim);
}

// ═══════════════════ lesson13 FA-2：外 i 内 j，O/m/ℓ 片上 ═══════════════════

__global__ void flash_attention_v2(const float *Q, const float *K, const float *V,
                                   float *O, float *L, int seq, int dim) {
    const int q_start = blockIdx.x * BR;
    if (q_start >= seq) return;

    const int br = min(BR, seq - q_start);
    const int tid = threadIdx.x;
    const float scale = rsqrtf((float)dim);

    float o_acc[O_CHUNK];
    for (int k = 0; k < O_CHUNK; ++k) o_acc[k] = 0.0f;

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

        for (int idx = tid; idx < br * BC; idx += blockDim.x) S[idx] = 0.0f;
        __syncthreads();

        for (int td = 0; td < dim; td += TD) {
            const int td_size = min(TD, dim - td);
            for (int idx = tid; idx < br * td_size; idx += blockDim.x) {
                const int r = idx / td_size, t = idx % td_size;
                Qs[r * TD + t] = Q[(q_start + r) * dim + td + t];
            }
            for (int idx = tid; idx < bc * td_size; idx += blockDim.x) {
                const int c = idx / td_size, t = idx % td_size;
                Ks[c * TD + t] = K[(kv_start + c) * dim + td + t];
            }
            __syncthreads();
            for (int idx = tid; idx < br * bc; idx += blockDim.x) {
                const int r = idx / bc, c = idx % bc;
                float dot = 0.0f;
                for (int t = 0; t < td_size; ++t)
                    dot += Qs[r * TD + t] * Ks[c * TD + t];
                S[r * BC + c] += dot;
            }
            __syncthreads();
        }

        for (int idx = tid; idx < br * bc; idx += blockDim.x) S[idx] *= scale;
        __syncthreads();

        for (int r = tid; r < br; r += blockDim.x) {
            float m_ij = -FLT_MAX;
            for (int c = 0; c < bc; ++c) m_ij = fmaxf(m_ij, S[r * BC + c]);
            const float m_old = m_i[r], l_old = l_i[r];
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
                const int c = idx / td_size, t = idx % td_size;
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
                for (int c = 0; c < bc; ++c) pv += S[r * BC + c] * Vs[c * TD + t];
                o_acc[k] = row_alpha[r] * o_acc[k] + pv;
            }
            __syncthreads();
        }
    }

    for (int k = 0; k < O_CHUNK; ++k) {
        const int idx = tid + k * blockDim.x;
        if (idx >= br * dim) continue;
        const int r = idx / dim, d = idx % dim;
        O[(q_start + r) * dim + d] = o_acc[k] / l_i[r];
    }
    for (int r = tid; r < br; r += blockDim.x) {
        L[q_start + r] = m_i[r] + logf(l_i[r]);
    }
}

// ═══════════════════ 工具函数 ═══════════════════

enum KernelId { K_BASE, K_V1, K_V2 };

static float dot_cpu(const float *a, const float *b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) s += a[i] * b[i];
    return s;
}

static void attention_cpu(const float *Q, const float *K, const float *V,
                            float *out, int seq, int dim) {
    float *scores = (float *)malloc(seq * sizeof(float));
    const float scale = rsqrtf((float)dim);
    for (int s = 0; s < seq; ++s) {
        const float *q = Q + s * dim;
        for (int t = 0; t < seq; ++t)
            scores[t] = dot_cpu(q, K + t * dim, dim) * scale;
        float row_max = -FLT_MAX;
        for (int t = 0; t < seq; ++t) row_max = fmaxf(row_max, scores[t]);
        double row_sum = 0.0;
        for (int t = 0; t < seq; ++t) {
            scores[t] = expf(scores[t] - row_max);
            row_sum += scores[t];
        }
        for (int t = 0; t < seq; ++t) scores[t] /= (float)row_sum;
        float *o = out + s * dim;
        for (int d = 0; d < dim; ++d) {
            double sum = 0.0;
            for (int t = 0; t < seq; ++t) sum += (double)scores[t] * V[t * dim + d];
            o[d] = (float)sum;
        }
    }
    free(scores);
}

static void launch_kernel(KernelId kid, const float *d_Q, const float *d_K,
                          const float *d_V, float *d_M, float *d_L,
                          float *d_O, int seq, int dim) {
    const int Tr = (seq + BR - 1) / BR;
    if (kid == K_BASE) {
        flash_attention_l10<<<Tr, BLOCK, smem_bytes()>>>(d_Q, d_K, d_V, d_O, seq,
                                                         dim);
    } else if (kid == K_V1) {
        launch_fa1(d_Q, d_K, d_V, d_M, d_L, d_O, seq, dim);
    } else {
        flash_attention_v2<<<Tr, BLOCK, smem_bytes()>>>(d_Q, d_K, d_V, d_O, d_L,
                                                        seq, dim);
    }
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

static float benchmark_kernel(KernelId kid, const float *d_Q, const float *d_K,
                              const float *d_V, float *d_M, float *d_L,
                              float *d_O, int seq, int dim) {
    const size_t bytes = (size_t)seq * dim * sizeof(float);

    for (int i = 0; i < WARMUP; ++i) {
        CUDA_CHECK(cudaMemset(d_O, 0, bytes));
        if (kid == K_V1) init_m_l_gpu(d_M, d_L, seq);
        launch_kernel(kid, d_Q, d_K, d_V, d_M, d_L, d_O, seq, dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < REPEATS; ++i) {
        CUDA_CHECK(cudaMemset(d_O, 0, bytes));
        if (kid == K_V1) init_m_l_gpu(d_M, d_L, seq);
        launch_kernel(kid, d_Q, d_K, d_V, d_M, d_L, d_O, seq, dim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / REPEATS;
}

static double attention_gflops(int seq, int dim, float ms) {
    const double flops = 4.0 * (double)seq * seq * dim;
    return flops / (ms * 1e6);
}

static void run_one_seq(int seq, bool do_verify) {
    const int N = seq * DIM;
    const size_t bytes = N * sizeof(float);

    float *h_Q = (float *)malloc(bytes);
    float *h_K = (float *)malloc(bytes);
    float *h_V = (float *)malloc(bytes);
    float *h_obase = (float *)malloc(bytes);
    float *h_ov1 = (float *)malloc(bytes);
    float *h_ov2 = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < N; ++i) {
        h_Q[i] = (float)(rand() % 200 - 100) / 50.0f;
        h_K[i] = (float)(rand() % 200 - 100) / 50.0f;
        h_V[i] = (float)(rand() % 200 - 100) / 50.0f;
    }

    float *d_Q, *d_K, *d_V, *d_O, *d_M, *d_L;
    CUDA_CHECK(cudaMalloc((void **)&d_Q, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_K, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_V, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_O, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_M, (size_t)seq * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_L, (size_t)seq * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));

    if (do_verify) {
        attention_cpu(h_Q, h_K, h_V, h_ref, seq, DIM);

        CUDA_CHECK(cudaMemset(d_O, 0, bytes));
        launch_kernel(K_BASE, d_Q, d_K, d_V, d_M, d_L, d_O, seq, DIM);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_obase, d_O, bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(d_O, 0, bytes));
        init_m_l_gpu(d_M, d_L, seq);
        launch_kernel(K_V1, d_Q, d_K, d_V, d_M, d_L, d_O, seq, DIM);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_ov1, d_O, bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(d_O, 0, bytes));
        launch_kernel(K_V2, d_Q, d_K, d_V, d_M, d_L, d_O, seq, DIM);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_ov2, d_O, bytes, cudaMemcpyDeviceToHost));

        const float err_base = max_abs_diff(h_obase, h_ref, N);
        const float errv1 = max_abs_diff(h_ov1, h_ref, N);
        const float errv2 = max_abs_diff(h_ov2, h_ref, N);
        printf("  [verify seq=%d] base=%.4e v1=%.4e v2=%.4e", seq, err_base,
               errv1, errv2);
        printf("  %s\n",
               (err_base < VERIFY_TOL && errv1 < VERIFY_TOL && errv2 < VERIFY_TOL)
                   ? "OK"
                   : "FAIL");
    }

    const float tbase =
        benchmark_kernel(K_BASE, d_Q, d_K, d_V, d_M, d_L, d_O, seq, DIM);
    const float tv1 =
        benchmark_kernel(K_V1, d_Q, d_K, d_V, d_M, d_L, d_O, seq, DIM);
    const float tv2 =
        benchmark_kernel(K_V2, d_Q, d_K, d_V, d_M, d_L, d_O, seq, DIM);

    printf("%6d  %10.3f  %10.3f  %10.3f  %8.2fx  %8.2fx  %8.1f  %8.1f  %8.1f\n",
           seq, tbase, tv1, tv2, tbase / tv1, tbase / tv2,
           attention_gflops(seq, DIM, tbase), attention_gflops(seq, DIM, tv1),
           attention_gflops(seq, DIM, tv2));

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_L));
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_obase);
    free(h_ov1);
    free(h_ov2);
    free(h_ref);
}

int main(int argc, char **argv) {
    int default_seqs[] = {512, 1024, 2048, 4096, 8192};
    int *seqs = default_seqs;
    int nseqs = 5;

    if (argc > 1) {
        seqs = (int *)malloc((size_t)(argc - 1) * sizeof(int));
        nseqs = 0;
        for (int i = 1; i < argc; ++i) {
            const int s = atoi(argv[i]);
            if (s <= 0) {
                fprintf(stderr, "用法: %s [seq1 seq2 ...]\n", argv[0]);
                return 1;
            }
            seqs[nseqs++] = s;
        }
    }

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("FlashAttention 性能对比 (标准版 / v1 / v2-FA2)\n");
    printf("GPU: %s | DIM=%d BR=%d BC=%d TD=%d | warmup=%d repeats=%d\n\n",
           prop.name, DIM, BR, BC, TD, WARMUP, REPEATS);
    printf("   SEQ   baseline_ms       v1_ms       v2_ms  v1/base  v2/base  GFLOPS_b  GFLOPS_v1  GFLOPS_v2\n");
    printf("------  ------------  ----------  ----------  -------  -------  --------  ---------  ---------\n");

    for (int i = 0; i < nseqs; ++i) {
        const bool verify = seqs[i] <= 2048;
        run_one_seq(seqs[i], verify);
    }

    if (argc > 1) free(seqs);
    return 0;
}
