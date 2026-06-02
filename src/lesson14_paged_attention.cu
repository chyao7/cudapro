/*
 * PagedAttention — vLLM 风格分页 KV cache + decode 单 token attention
 *
 * 物理 KV: k_cache/v_cache [num_pages, PAGE_SIZE, DIM]  页池，可非连续分配
 * 逻辑映射: block_table[batch, max_blocks]              逻辑块 → 物理页号
 * decode:   Q [batch, DIM]  对 context_lens[i] 长度 cache 做 online softmax
 *
 * nvcc -O3 -arch=sm_86 -o lesson14 lesson14_paged_attention.cu
 * ./lesson14
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

#define DIM 128
#define PAGE_SIZE 16
#define TD 64
#define BLOCK 256
#define BATCH 4
#define MAX_CONTEXT 80
#define MAX_BLOCKS ((MAX_CONTEXT + PAGE_SIZE - 1) / PAGE_SIZE)
#define NUM_PAGES 16

/* 逻辑 token t → 物理地址（k_cache / v_cache 布局 [page, PAGE_SIZE, DIM]） */
__device__ __forceinline__ const float *page_kv_ptr(const float *cache, int page_id,
                                                    int offset_in_page, int dim) {
    return cache + ((size_t)page_id * PAGE_SIZE + offset_in_page) * dim;
}

/*
 * decode：每个 block 处理 batch 中一条序列的一个 Q token
 * 按 block_table 从分页 cache 读 K/V，online softmax 累积 O
 */
__global__ void paged_attention_decode(const float *Q, const float *k_cache,
                                       const float *v_cache,
                                       const int *block_tables,
                                       const int *context_lens, float *O,
                                       int batch, int dim, int max_blocks) {
    const int seq_id = blockIdx.x;
    if (seq_id >= batch) return;

    const int ctx_len = context_lens[seq_id];
    if (ctx_len <= 0) return;

    const int tid = threadIdx.x;
    const float scale = rsqrtf((float)dim);
    const int *table = block_tables + seq_id * max_blocks;
    const float *q = Q + (size_t)seq_id * dim;

    __shared__ float smem_m;
    __shared__ float smem_l;
    __shared__ float smem_alpha;
    __shared__ float o_smem[DIM];
    __shared__ float ks[PAGE_SIZE * TD];
    __shared__ float vs[PAGE_SIZE * TD];
    __shared__ float scores[PAGE_SIZE];

    if (tid == 0) {
        smem_m = -FLT_MAX;
        smem_l = 0.0f;
    }
    for (int d = tid; d < dim; d += blockDim.x) {
        o_smem[d] = 0.0f;
    }
    __syncthreads();

    const int num_blocks = (ctx_len + PAGE_SIZE - 1) / PAGE_SIZE;

    for (int j = 0; j < num_blocks; ++j) {
        const int page_id = table[j];
        const int tok_start = j * PAGE_SIZE;
        const int bc = min(PAGE_SIZE, ctx_len - tok_start);

        /* S[tok] = scale * q · K[tok] */
        for (int idx = tid; idx < bc; idx += blockDim.x) {
            scores[idx] = 0.0f;
        }
        __syncthreads();

        for (int td = 0; td < dim; td += TD) {
            const int td_size = min(TD, dim - td);

            for (int idx = tid; idx < bc * td_size; idx += blockDim.x) {
                const int c = idx / td_size;
                const int t = idx % td_size;
                ks[c * TD + t] =
                    page_kv_ptr(k_cache, page_id, c, dim)[td + t];
            }
            __syncthreads();

            if (tid == 0) {
                for (int c = 0; c < bc; ++c) {
                    float dot = 0.0f;
                    for (int t = 0; t < td_size; ++t) {
                        dot += q[td + t] * ks[c * TD + t];
                    }
                    scores[c] += dot;
                }
            }
            __syncthreads();
        }

        for (int idx = tid; idx < bc; idx += blockDim.x) {
            scores[idx] *= scale;
        }
        __syncthreads();

        /* online softmax over this page */
        float m_page = -FLT_MAX;
        if (tid == 0) {
            for (int c = 0; c < bc; ++c) {
                m_page = fmaxf(m_page, scores[c]);
            }
        }
        __shared__ float smem_m_page;
        if (tid == 0) smem_m_page = m_page;
        __syncthreads();
        m_page = smem_m_page;

        if (tid == 0) {
            const float m_new = fmaxf(smem_m, m_page);
            smem_alpha = expf(smem_m - m_new);
            float l_page = 0.0f;
            for (int c = 0; c < bc; ++c) {
                scores[c] = expf(scores[c] - m_new);
                l_page += scores[c];
            }
            smem_l = smem_alpha * smem_l + l_page;
            smem_m = m_new;
        }
        __syncthreads();

        for (int td = 0; td < dim; td += TD) {
            const int td_size = min(TD, dim - td);

            for (int idx = tid; idx < bc * td_size; idx += blockDim.x) {
                const int c = idx / td_size;
                const int t = idx % td_size;
                vs[c * TD + t] =
                    page_kv_ptr(v_cache, page_id, c, dim)[td + t];
            }
            __syncthreads();

            for (int d = tid; d < dim; d += blockDim.x) {
                if (d < td || d >= td + td_size) continue;
                const int t = d - td;
                float pv = 0.0f;
                for (int c = 0; c < bc; ++c) {
                    pv += scores[c] * vs[c * TD + t];
                }
                o_smem[d] = smem_alpha * o_smem[d] + pv;
            }
            __syncthreads();
        }
    }

    float *out = O + (size_t)seq_id * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        out[d] = o_smem[d] / smem_l;
    }
}

/* 连续 KV baseline（同一 Q，K/V 在 HBM 连续存放） */
__global__ void contiguous_attention_decode(const float *Q, const float *K,
                                            const float *V,
                                            const int *context_lens, float *O,
                                            int batch, int dim) {
    const int seq_id = blockIdx.x;
    if (seq_id >= batch) return;

    const int ctx_len = context_lens[seq_id];
    if (ctx_len <= 0) return;

    const int tid = threadIdx.x;
    const float scale = rsqrtf((float)dim);
    const float *q = Q + (size_t)seq_id * dim;
    const float *k_base = K + (size_t)seq_id * MAX_CONTEXT * dim;
    const float *v_base = V + (size_t)seq_id * MAX_CONTEXT * dim;

    __shared__ float smem_m;
    __shared__ float smem_l;
    __shared__ float smem_alpha;
    __shared__ float o_smem[DIM];
    __shared__ float ks[PAGE_SIZE * TD];
    __shared__ float vs[PAGE_SIZE * TD];
    __shared__ float scores[PAGE_SIZE];

    if (tid == 0) {
        smem_m = -FLT_MAX;
        smem_l = 0.0f;
    }
    for (int d = tid; d < dim; d += blockDim.x) {
        o_smem[d] = 0.0f;
    }
    __syncthreads();

    const int num_blocks = (ctx_len + PAGE_SIZE - 1) / PAGE_SIZE;

    for (int j = 0; j < num_blocks; ++j) {
        const int tok_start = j * PAGE_SIZE;
        const int bc = min(PAGE_SIZE, ctx_len - tok_start);

        for (int idx = tid; idx < bc; idx += blockDim.x) {
            scores[idx] = 0.0f;
        }
        __syncthreads();

        for (int td = 0; td < dim; td += TD) {
            const int td_size = min(TD, dim - td);

            for (int idx = tid; idx < bc * td_size; idx += blockDim.x) {
                const int c = idx / td_size;
                const int t = idx % td_size;
                const int tok = tok_start + c;
                ks[c * TD + t] = k_base[(size_t)tok * dim + td + t];
            }
            __syncthreads();

            if (tid == 0) {
                for (int c = 0; c < bc; ++c) {
                    float dot = 0.0f;
                    for (int t = 0; t < td_size; ++t) {
                        dot += q[td + t] * ks[c * TD + t];
                    }
                    scores[c] += dot;
                }
            }
            __syncthreads();
        }

        for (int idx = tid; idx < bc; idx += blockDim.x) {
            scores[idx] *= scale;
        }
        __syncthreads();

        float m_page = -FLT_MAX;
        if (tid == 0) {
            for (int c = 0; c < bc; ++c) {
                m_page = fmaxf(m_page, scores[c]);
            }
        }
        __shared__ float smem_m_page;
        if (tid == 0) smem_m_page = m_page;
        __syncthreads();
        m_page = smem_m_page;

        if (tid == 0) {
            const float m_new = fmaxf(smem_m, m_page);
            smem_alpha = expf(smem_m - m_new);
            float l_page = 0.0f;
            for (int c = 0; c < bc; ++c) {
                scores[c] = expf(scores[c] - m_new);
                l_page += scores[c];
            }
            smem_l = smem_alpha * smem_l + l_page;
            smem_m = m_new;
        }
        __syncthreads();

        for (int td = 0; td < dim; td += TD) {
            const int td_size = min(TD, dim - td);

            for (int idx = tid; idx < bc * td_size; idx += blockDim.x) {
                const int c = idx / td_size;
                const int t = idx % td_size;
                const int tok = tok_start + c;
                vs[c * TD + t] = v_base[(size_t)tok * dim + td + t];
            }
            __syncthreads();

            for (int d = tid; d < dim; d += blockDim.x) {
                if (d < td || d >= td + td_size) continue;
                const int t = d - td;
                float pv = 0.0f;
                for (int c = 0; c < bc; ++c) {
                    pv += scores[c] * vs[c * TD + t];
                }
                o_smem[d] = smem_alpha * o_smem[d] + pv;
            }
            __syncthreads();
        }
    }

    float *out = O + (size_t)seq_id * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        out[d] = o_smem[d] / smem_l;
    }
}

/* CPU：按 block_table  gather 后算 decode attention（验证用） */
static void paged_attention_cpu(const float *Q, const float *k_cache,
                                const float *v_cache, const int *block_tables,
                                const int *context_lens, float *O, int batch,
                                int dim, int max_blocks) {
    const float scale = rsqrtf((float)dim);

    for (int b = 0; b < batch; ++b) {
        const int ctx = context_lens[b];
        const int *table = block_tables + b * max_blocks;
        const float *q = Q + (size_t)b * dim;
        float *o = O + (size_t)b * dim;

        float m = -FLT_MAX;
        double l = 0.0;
        for (int d = 0; d < dim; ++d) o[d] = 0.0f;

        for (int t = 0; t < ctx; ++t) {
            const int j = t / PAGE_SIZE;
            const int off = t % PAGE_SIZE;
            const int page_id = table[j];
            const float *k =
                k_cache + ((size_t)page_id * PAGE_SIZE + off) * dim;

            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) dot += q[d] * k[d];
            dot *= scale;

            m = fmaxf(m, dot);
        }

        l = 0.0;
        for (int t = 0; t < ctx; ++t) {
            const int j = t / PAGE_SIZE;
            const int off = t % PAGE_SIZE;
            const int page_id = table[j];
            const float *k =
                k_cache + ((size_t)page_id * PAGE_SIZE + off) * dim;
            const float *v =
                v_cache + ((size_t)page_id * PAGE_SIZE + off) * dim;

            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) dot += q[d] * k[d];
            dot *= scale;

            const float p = expf(dot - m);
            l += p;
            for (int d = 0; d < dim; ++d) {
                o[d] += (double)p * v[d];
            }
        }

        for (int d = 0; d < dim; ++d) {
            o[d] = (float)((double)o[d] / l);
        }
    }
}

/*
 * 模拟 vLLM 页池：为每条序列分配非连续物理页，写入 KV
 * k_src/v_src: [batch, MAX_CONTEXT, dim] 逻辑连续 prefill 数据
 */
static void build_paged_cache(const float *k_src, const float *v_src,
                              float *k_cache, float *v_cache,
                              int *block_tables, int *context_lens, int batch,
                              int dim, int max_blocks) {
    int next_page = 0;

    for (int b = 0; b < batch; ++b) {
        const int ctx = context_lens[b];
        const int nb = (ctx + PAGE_SIZE - 1) / PAGE_SIZE;
        int *table = block_tables + b * max_blocks;

        for (int j = 0; j < nb; ++j) {
            table[j] = next_page++;
            const int page_id = table[j];
            const int tok_start = j * PAGE_SIZE;
            const int ntok = min(PAGE_SIZE, ctx - tok_start);

            for (int t = 0; t < ntok; ++t) {
                const float *ks = k_src + ((size_t)b * MAX_CONTEXT + tok_start + t) * dim;
                const float *vs = v_src + ((size_t)b * MAX_CONTEXT + tok_start + t) * dim;
                float *kd =
                    k_cache + ((size_t)page_id * PAGE_SIZE + t) * dim;
                float *vd =
                    v_cache + ((size_t)page_id * PAGE_SIZE + t) * dim;
                memcpy(kd, ks, (size_t)dim * sizeof(float));
                memcpy(vd, vs, (size_t)dim * sizeof(float));
            }
        }
        for (int j = nb; j < max_blocks; ++j) {
            table[j] = -1;
        }
    }
}

static bool verify_output(const float *gpu, const float *ref, int n, float tol) {
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_err = fmaxf(max_err, fabsf(gpu[i] - ref[i]));
    }
    printf("  max_err = %.6f %s\n", max_err, max_err < tol ? "(OK)" : "(FAIL)");
    return max_err < tol;
}

static void print_block_tables(const int *block_tables, const int *context_lens,
                               int batch, int max_blocks) {
    printf("── block_table（逻辑块 j → 物理页 page_id）──\n");
    for (int b = 0; b < batch; ++b) {
        const int ctx = context_lens[b];
        const int nb = (ctx + PAGE_SIZE - 1) / PAGE_SIZE;
        printf("  seq%d ctx=%2d: ", b, ctx);
        for (int j = 0; j < nb; ++j) {
            printf("j%d→p%d ", j, block_tables[b * max_blocks + j]);
        }
        printf("\n");
    }
    printf("  （物理页故意非连续分配，模拟显存碎片）\n\n");
}

int main(void) {
    const int context_lens[BATCH] = {37, 50, 21, 80};
    const size_t q_bytes = (size_t)BATCH * DIM * sizeof(float);
    const size_t kv_logical_bytes =
        (size_t)BATCH * MAX_CONTEXT * DIM * sizeof(float);
    const size_t page_pool_bytes =
        (size_t)NUM_PAGES * PAGE_SIZE * DIM * sizeof(float);
    const size_t table_bytes =
        (size_t)BATCH * MAX_BLOCKS * sizeof(int);
    const size_t ctx_bytes = (size_t)BATCH * sizeof(int);

    printf("PagedAttention decode demo\n");
    printf("BATCH=%d DIM=%d PAGE_SIZE=%d MAX_CONTEXT=%d\n\n",
           BATCH, DIM, PAGE_SIZE, MAX_CONTEXT);

    float *h_Q = (float *)malloc(q_bytes);
    float *h_K = (float *)malloc(kv_logical_bytes);
    float *h_V = (float *)malloc(kv_logical_bytes);
    float *h_k_cache = (float *)calloc(NUM_PAGES * PAGE_SIZE * DIM, sizeof(float));
    float *h_v_cache = (float *)calloc(NUM_PAGES * PAGE_SIZE * DIM, sizeof(float));
    float *h_O_gpu = (float *)malloc(q_bytes);
    float *h_O_ref = (float *)malloc(q_bytes);
    float *h_O_contig = (float *)malloc(q_bytes);
    int *h_block_tables = (int *)malloc(table_bytes);
    int *h_context_lens = (int *)malloc(ctx_bytes);
    memcpy(h_context_lens, context_lens, ctx_bytes);

    srand(42);
    for (size_t i = 0; i < q_bytes / sizeof(float); ++i) {
        h_Q[i] = (float)(rand() % 200 - 100) / 50.0f;
    }
    for (size_t i = 0; i < kv_logical_bytes / sizeof(float); ++i) {
        h_K[i] = (float)(rand() % 200 - 100) / 50.0f;
        h_V[i] = (float)(rand() % 200 - 100) / 50.0f;
    }

    build_paged_cache(h_K, h_V, h_k_cache, h_v_cache, h_block_tables,
                      h_context_lens, BATCH, DIM, MAX_BLOCKS);
    print_block_tables(h_block_tables, h_context_lens, BATCH, MAX_BLOCKS);

    paged_attention_cpu(h_Q, h_k_cache, h_v_cache, h_block_tables,
                        h_context_lens, h_O_ref, BATCH, DIM, MAX_BLOCKS);

    float *d_Q, *d_k_cache, *d_v_cache, *d_O, *d_K, *d_V;
    int *d_block_tables, *d_context_lens;
    CUDA_CHECK(cudaMalloc(&d_Q, q_bytes));
    CUDA_CHECK(cudaMalloc(&d_k_cache, page_pool_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_cache, page_pool_bytes));
    CUDA_CHECK(cudaMalloc(&d_O, q_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, kv_logical_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, kv_logical_bytes));
    CUDA_CHECK(cudaMalloc(&d_block_tables, table_bytes));
    CUDA_CHECK(cudaMalloc(&d_context_lens, ctx_bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, q_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_cache, h_k_cache, page_pool_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_cache, h_v_cache, page_pool_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, kv_logical_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, kv_logical_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_tables, h_block_tables, table_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_context_lens, h_context_lens, ctx_bytes,
                          cudaMemcpyHostToDevice));

    paged_attention_decode<<<BATCH, BLOCK>>>(
        d_Q, d_k_cache, d_v_cache, d_block_tables, d_context_lens, d_O, BATCH,
        DIM, MAX_BLOCKS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O_gpu, d_O, q_bytes, cudaMemcpyDeviceToHost));

    printf("PagedAttention 验证: %s\n",
           verify_output(h_O_gpu, h_O_ref, BATCH * DIM, 1e-2f) ? "通过" : "失败");

    contiguous_attention_decode<<<BATCH, BLOCK>>>(d_Q, d_K, d_V, d_context_lens,
                                                  d_O, BATCH, DIM);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O_contig, d_O, q_bytes, cudaMemcpyDeviceToHost));
    printf("连续 KV baseline 与 paged 一致: %s\n",
           verify_output(h_O_contig, h_O_ref, BATCH * DIM, 1e-2f) ? "是" : "否");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < 10; ++i) {
        paged_attention_decode<<<BATCH, BLOCK>>>(
            d_Q, d_k_cache, d_v_cache, d_block_tables, d_context_lens, d_O,
            BATCH, DIM, MAX_BLOCKS);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; ++i) {
        paged_attention_decode<<<BATCH, BLOCK>>>(
            d_Q, d_k_cache, d_v_cache, d_block_tables, d_context_lens, d_O,
            BATCH, DIM, MAX_BLOCKS);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_paged = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_paged, start, stop));

    for (int i = 0; i < 10; ++i) {
        contiguous_attention_decode<<<BATCH, BLOCK>>>(d_Q, d_K, d_V,
                                                    d_context_lens, d_O, BATCH,
                                                    DIM);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; ++i) {
        contiguous_attention_decode<<<BATCH, BLOCK>>>(d_Q, d_K, d_V,
                                                      d_context_lens, d_O,
                                                      BATCH, DIM);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_contig = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_contig, start, stop));

    printf("PagedAttention decode: %.3f ms\n", ms_paged / 100.0f);
    printf("连续 KV decode:        %.3f ms\n", ms_contig / 100.0f);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_k_cache));
    CUDA_CHECK(cudaFree(d_v_cache));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_block_tables));
    CUDA_CHECK(cudaFree(d_context_lens));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_k_cache);
    free(h_v_cache);
    free(h_O_gpu);
    free(h_O_ref);
    free(h_O_contig);
    free(h_block_tables);
    free(h_context_lens);
    return 0;
}
