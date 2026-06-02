/*
 * FlashAttention v1 — FA-1 Algorithm 1
 * nvcc -O3 -o lesson12 lesson12_flash_attention_v1.cu
 * ./lesson12 [seq]
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
#define Br 32
#define Bc 32
#define Td 64
#define BLOCK 256

static size_t fa_v1_smem_bytes(void) {
    return (Br * Td + Bc * Td + Bc * Td + Br * Bc) * sizeof(float);
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

__device__ void fa_v1_online_softmax_row(float *S, int r, int bc, float m_row,
                                         float *m_new_out, float *l_row_inout,
                                         float *scale_m_out,
                                         float *scale_ij_out) {
    float m_ij = -FLT_MAX;
    for (int c = 0; c < bc; ++c) {
        m_ij = fmaxf(m_ij, S[r * Bc + c]);
    }

    float l_ij = 0.0f;
    for (int c = 0; c < bc; ++c) {
        const float p = expf(S[r * Bc + c] - m_ij);
        S[r * Bc + c] = p;
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
    const int q_start = blockIdx.x * Br;
    if (q_start >= seq) return;

    const int kv_start = j * Bc;
    if (kv_start >= seq) return;

    const int br = min(Br, seq - q_start);
    const int bc = min(Bc, seq - kv_start);
    const int tid = threadIdx.x;
    const float scale = rsqrtf((float)dim);

    __shared__ float m_tile[Br];
    __shared__ float l_tile[Br];
    __shared__ float scale_m[Br];
    __shared__ float scale_ij[Br];

    extern __shared__ float smem[];
    float *Qs = smem;
    float *Ks = Qs + Br * Td;
    float *Vs = Ks + Bc * Td;
    float *S = Vs + Bc * Td;

    for (int r = tid; r < br; r += blockDim.x) {
        const int row = q_start + r;
        m_tile[r] = M[row];
        l_tile[r] = L[row];
    }
    __syncthreads();

    for (int idx = tid; idx < br * Bc; idx += blockDim.x) {
        S[idx] = 0.0f;
    }
    __syncthreads();

    for (int td = 0; td < dim; td += Td) {
        const int td_size = min(Td, dim - td);

        for (int idx = tid; idx < br * td_size; idx += blockDim.x) {
            const int r = idx / td_size;
            const int t = idx % td_size;
            Qs[r * Td + t] = Q[(q_start + r) * dim + td + t];
        }
        for (int idx = tid; idx < bc * td_size; idx += blockDim.x) {
            const int c = idx / td_size;
            const int t = idx % td_size;
            Ks[c * Td + t] = K[(kv_start + c) * dim + td + t];
        }
        __syncthreads();

        for (int idx = tid; idx < br * bc; idx += blockDim.x) {
            const int r = idx / bc;
            const int c = idx % bc;
            float dot = 0.0f;
            for (int t = 0; t < td_size; ++t) {
                dot += Qs[r * Td + t] * Ks[c * Td + t];
            }
            S[r * Bc + c] += dot;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < br * bc; idx += blockDim.x) {
        S[idx] *= scale;
    }
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

    for (int td = 0; td < dim; td += Td) {
        const int td_size = min(Td, dim - td);

        for (int idx = tid; idx < bc * td_size; idx += blockDim.x) {
            const int c = idx / td_size;
            const int t = idx % td_size;
            Vs[c * Td + t] = V[(kv_start + c) * dim + td + t];
        }
        __syncthreads();

        for (int idx = tid; idx < br * td_size; idx += blockDim.x) {
            const int r = idx / td_size;
            const int t = idx % td_size;
            float pv = 0.0f;
            for (int c = 0; c < bc; ++c) {
                pv += S[r * Bc + c] * Vs[c * Td + t];
            }
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
    const int q_start = blockIdx.x * Br;
    if (q_start >= seq) return;
    const int br = min(Br, seq - q_start);
    const int tid = threadIdx.x;

    for (int idx = tid; idx < br * dim; idx += blockDim.x) {
        const int r = idx / dim;
        const int d = idx % dim;
        const int row = q_start + r;
        O[(size_t)row * dim + d] /= L[row];
    }
}

static void flash_attention_v1_launch(const float *d_Q, const float *d_K,
                                      const float *d_V, float *d_M, float *d_L,
                                      float *d_O, int seq, int dim,
                                      size_t smem) {
    const int Tr = (seq + Br - 1) / Br;
    const int Tc = (seq + Bc - 1) / Bc;

    for (int j = 0; j < Tc; ++j) {
        flash_attention_v1_step<<<Tr, BLOCK, smem>>>(d_Q, d_K, d_V, d_M, d_L,
                                                     d_O, j, seq, dim);
    }
    flash_attention_v1_finalize<<<Tr, BLOCK>>>(d_O, d_L, seq, dim);
}

static bool verify(const float *gpu, const float *cpu, int n, float tol) {
    for (int i = 0; i < n; ++i) {
        if (fabsf(gpu[i] - cpu[i]) > tol) return false;
    }
    return true;
}

static void init_m_l(float *d_M, float *d_L, int seq) {
    float *h_M = (float *)malloc((size_t)seq * sizeof(float));
    for (int i = 0; i < seq; ++i) h_M[i] = -FLT_MAX;
    CUDA_CHECK(cudaMemcpy(d_M, h_M, (size_t)seq * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_L, 0, (size_t)seq * sizeof(float)));
    free(h_M);
}

static float benchmark_v1(const float *d_Q, const float *d_K, const float *d_V,
                          float *d_M, float *d_L, float *d_out, int seq,
                          int dim, size_t smem_bytes, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, (size_t)seq * dim * sizeof(float)));
        init_m_l(d_M, d_L, seq);
        flash_attention_v1_launch(d_Q, d_K, d_V, d_M, d_L, d_out, seq, dim,
                                  smem_bytes);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, (size_t)seq * dim * sizeof(float)));
        init_m_l(d_M, d_L, seq);
        flash_attention_v1_launch(d_Q, d_K, d_V, d_M, d_L, d_out, seq, dim,
                                  smem_bytes);
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
    const size_t smem_bytes = fa_v1_smem_bytes();

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

    if (!skip_cpu) {
        attention_cpu(h_Q, h_K, h_V, h_ref, seq, DIM);
    }

    CUDA_CHECK(cudaMemset(d_O, 0, bytes));
    init_m_l(d_M, d_L, seq);
    flash_attention_v1_launch(d_Q, d_K, d_V, d_M, d_L, d_O, seq, DIM,
                              smem_bytes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_O, bytes, cudaMemcpyDeviceToHost));

    if (!skip_cpu && !verify(h_out, h_ref, N, 1e-2f)) {
        fprintf(stderr, "验证失败\n");
        return 1;
    }

    const float t_v1 = benchmark_v1(d_Q, d_K, d_V, d_M, d_L, d_O, seq, DIM,
                                    smem_bytes, 2, 10);
    printf("FlashAttention v1 耗时: %.3f ms\n", t_v1);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_L));
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_out);
    free(h_ref);
    return 0;
}
