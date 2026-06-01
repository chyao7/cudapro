/*
 * FlashAttention 标准版 — 两遍 Safe Softmax
 * nvcc -O3 -o lesson10 lesson10_flash_attention.cu
 * ./lesson10 [seq] [--gpu-only]
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

static bool verify(const float *gpu, const float *cpu, int n, float tol) {
    for (int i = 0; i < n; ++i) {
        if (fabsf(gpu[i] - cpu[i]) > tol) return false;
    }
    return true;
}

static float benchmark_flash(const float *d_Q, const float *d_K, const float *d_V,
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
        attention_cpu(h_Q, h_K, h_V, h_ref, seq, DIM);
    }

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    flash_attention<<<(seq + BR - 1) / BR, BLOCK, smem_bytes>>>(
        d_Q, d_K, d_V, d_out, seq, DIM);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    if (!skip_cpu && !verify(h_out, h_ref, N, 1e-2f)) {
        fprintf(stderr, "验证失败\n");
        return 1;
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
