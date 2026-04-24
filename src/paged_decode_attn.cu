
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __forceinline__ int page_for_token(const int *page_offsets, int num_pages,
                                              int token_index) {
  for (int page = 0; page < num_pages; ++page) {
    if (token_index < page_offsets[page + 1]) {
      return page;
    }
  }
  return num_pages - 1;
}

__global__ void paged_decode_attn_fp32_kernel(
    const float *Q, const float *K_pages, const float *V_pages,
    const int *page_offsets, const float *mask, float *out, int B, int H,
    int num_pages, int page_size, int total_seq, int D, float inv_sqrt_d) {
  int bh = blockIdx.x;
  int b = bh / H;
  int h = bh % H;
  if (b >= B) {
    return;
  }

  const float *q = Q + (b * H + h) * D;
  const float *msk = mask + (b * H + h) * total_seq;
  float *o = out + (b * H + h) * D;

  extern __shared__ float smem[];
  float *logits = smem;

  for (int j = threadIdx.x; j < total_seq; j += blockDim.x) {
    int page = page_for_token(page_offsets, num_pages, j);
    int offset = j - page_offsets[page];
    size_t base = ((((size_t)page * B + b) * H + h) * page_size + offset) * D;
    float sum = 0.f;
    for (int d = 0; d < D; ++d) {
      sum += q[d] * K_pages[base + d];
    }
    logits[j] = sum * inv_sqrt_d + msk[j];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float mv = logits[0];
    for (int i = 1; i < total_seq; ++i) {
      mv = fmaxf(mv, logits[i]);
    }
    float s = 0.f;
    for (int i = 0; i < total_seq; ++i) {
      logits[i] = expf(logits[i] - mv);
      s += logits[i];
    }
    for (int i = 0; i < total_seq; ++i) {
      logits[i] /= s;
    }
  }
  __syncthreads();

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < total_seq; ++j) {
      int page = page_for_token(page_offsets, num_pages, j);
      int offset = j - page_offsets[page];
      size_t base = ((((size_t)page * B + b) * H + h) * page_size + offset) * D;
      acc += logits[j] * V_pages[base + d];
    }
    o[d] = acc;
  }
}

__global__ void paged_decode_attn_int8_kernel(
    const float *Q, const signed char *K_pages, const signed char *V_pages,
    const int *page_offsets, const float *k_scales, const float *v_scales,
    const float *mask, float *out, int B, int H, int num_pages, int page_size,
    int total_seq, int D, float inv_sqrt_d) {
  int bh = blockIdx.x;
  int b = bh / H;
  int h = bh % H;
  if (b >= B) {
    return;
  }

  const float *q = Q + (b * H + h) * D;
  const float *msk = mask + (b * H + h) * total_seq;
  float *o = out + (b * H + h) * D;

  extern __shared__ float smem[];
  float *logits = smem;

  for (int j = threadIdx.x; j < total_seq; j += blockDim.x) {
    int page = page_for_token(page_offsets, num_pages, j);
    int offset = j - page_offsets[page];
    float k_scale = k_scales[page];
    size_t base = ((((size_t)page * B + b) * H + h) * page_size + offset) * D;
    float sum = 0.f;
    for (int d = 0; d < D; ++d) {
      sum += q[d] * ((float)K_pages[base + d] * k_scale);
    }
    logits[j] = sum * inv_sqrt_d + msk[j];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float mv = logits[0];
    for (int i = 1; i < total_seq; ++i) {
      mv = fmaxf(mv, logits[i]);
    }
    float s = 0.f;
    for (int i = 0; i < total_seq; ++i) {
      logits[i] = expf(logits[i] - mv);
      s += logits[i];
    }
    for (int i = 0; i < total_seq; ++i) {
      logits[i] /= s;
    }
  }
  __syncthreads();

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < total_seq; ++j) {
      int page = page_for_token(page_offsets, num_pages, j);
      int offset = j - page_offsets[page];
      float v_scale = v_scales[page];
      size_t base = ((((size_t)page * B + b) * H + h) * page_size + offset) * D;
      acc += logits[j] * ((float)V_pages[base + d] * v_scale);
    }
    o[d] = acc;
  }
}

static void chk(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

extern "C" {

void paged_decode_attn_host_fp32(
    const float *Q, const float *K_pages, const float *V_pages,
    const int *page_offsets, const float *mask, float *out, int B, int H,
    int num_pages, int page_size, int total_seq, int D, float inv_sqrt_d) {
  size_t q_sz = (size_t)B * H * D * sizeof(float);
  size_t kv_sz = (size_t)num_pages * B * H * page_size * D * sizeof(float);
  size_t offsets_sz = (size_t)(num_pages + 1) * sizeof(int);
  size_t m_sz = (size_t)B * H * total_seq * sizeof(float);
  size_t o_sz = q_sz;

  float *d_Q, *d_K, *d_V, *d_mask, *d_out;
  int *d_offsets;
  chk(cudaMalloc((void **)&d_Q, q_sz), "cudaMalloc paged Q");
  chk(cudaMalloc((void **)&d_K, kv_sz), "cudaMalloc paged K");
  chk(cudaMalloc((void **)&d_V, kv_sz), "cudaMalloc paged V");
  chk(cudaMalloc((void **)&d_offsets, offsets_sz), "cudaMalloc paged offsets");
  chk(cudaMalloc((void **)&d_mask, m_sz), "cudaMalloc paged mask");
  chk(cudaMalloc((void **)&d_out, o_sz), "cudaMalloc paged out");

  chk(cudaMemcpy(d_Q, Q, q_sz, cudaMemcpyHostToDevice), "memcpy paged Q");
  chk(cudaMemcpy(d_K, K_pages, kv_sz, cudaMemcpyHostToDevice), "memcpy paged K");
  chk(cudaMemcpy(d_V, V_pages, kv_sz, cudaMemcpyHostToDevice), "memcpy paged V");
  chk(cudaMemcpy(d_offsets, page_offsets, offsets_sz, cudaMemcpyHostToDevice), "memcpy paged offsets");
  chk(cudaMemcpy(d_mask, mask, m_sz, cudaMemcpyHostToDevice), "memcpy paged mask");

  int blocks = B * H;
  int threads = 256;
  size_t shmem = (size_t)total_seq * sizeof(float);
  paged_decode_attn_fp32_kernel<<<blocks, threads, shmem>>>(
      d_Q, d_K, d_V, d_offsets, d_mask, d_out, B, H, num_pages, page_size,
      total_seq, D, inv_sqrt_d);
  chk(cudaDeviceSynchronize(), "sync paged fp32");

  chk(cudaMemcpy(out, d_out, o_sz, cudaMemcpyDeviceToHost), "memcpy paged out");

  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_offsets);
  cudaFree(d_mask);
  cudaFree(d_out);
}

void paged_decode_attn_host_int8(
    const float *Q, const signed char *K_pages, const signed char *V_pages,
    const int *page_offsets, const float *k_scales, const float *v_scales,
    const float *mask, float *out, int B, int H, int num_pages, int page_size,
    int total_seq, int D, float inv_sqrt_d) {
  size_t q_sz = (size_t)B * H * D * sizeof(float);
  size_t kv_sz = (size_t)num_pages * B * H * page_size * D * sizeof(signed char);
  size_t offsets_sz = (size_t)(num_pages + 1) * sizeof(int);
  size_t scales_sz = (size_t)num_pages * sizeof(float);
  size_t m_sz = (size_t)B * H * total_seq * sizeof(float);
  size_t o_sz = q_sz;

  float *d_Q, *d_mask, *d_out, *d_k_scales, *d_v_scales;
  signed char *d_K, *d_V;
  int *d_offsets;
  chk(cudaMalloc((void **)&d_Q, q_sz), "cudaMalloc paged int8 Q");
  chk(cudaMalloc((void **)&d_K, kv_sz), "cudaMalloc paged int8 K");
  chk(cudaMalloc((void **)&d_V, kv_sz), "cudaMalloc paged int8 V");
  chk(cudaMalloc((void **)&d_offsets, offsets_sz), "cudaMalloc paged int8 offsets");
  chk(cudaMalloc((void **)&d_k_scales, scales_sz), "cudaMalloc paged int8 k_scales");
  chk(cudaMalloc((void **)&d_v_scales, scales_sz), "cudaMalloc paged int8 v_scales");
  chk(cudaMalloc((void **)&d_mask, m_sz), "cudaMalloc paged int8 mask");
  chk(cudaMalloc((void **)&d_out, o_sz), "cudaMalloc paged int8 out");

  chk(cudaMemcpy(d_Q, Q, q_sz, cudaMemcpyHostToDevice), "memcpy paged int8 Q");
  chk(cudaMemcpy(d_K, K_pages, kv_sz, cudaMemcpyHostToDevice), "memcpy paged int8 K");
  chk(cudaMemcpy(d_V, V_pages, kv_sz, cudaMemcpyHostToDevice), "memcpy paged int8 V");
  chk(cudaMemcpy(d_offsets, page_offsets, offsets_sz, cudaMemcpyHostToDevice), "memcpy paged int8 offsets");
  chk(cudaMemcpy(d_k_scales, k_scales, scales_sz, cudaMemcpyHostToDevice), "memcpy paged int8 k_scales");
  chk(cudaMemcpy(d_v_scales, v_scales, scales_sz, cudaMemcpyHostToDevice), "memcpy paged int8 v_scales");
  chk(cudaMemcpy(d_mask, mask, m_sz, cudaMemcpyHostToDevice), "memcpy paged int8 mask");

  int blocks = B * H;
  int threads = 256;
  size_t shmem = (size_t)total_seq * sizeof(float);
  paged_decode_attn_int8_kernel<<<blocks, threads, shmem>>>(
      d_Q, d_K, d_V, d_offsets, d_k_scales, d_v_scales, d_mask, d_out, B, H,
      num_pages, page_size, total_seq, D, inv_sqrt_d);
  chk(cudaDeviceSynchronize(), "sync paged int8");

  chk(cudaMemcpy(out, d_out, o_sz, cudaMemcpyDeviceToHost), "memcpy paged int8 out");

  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_offsets);
  cudaFree(d_k_scales);
  cudaFree(d_v_scales);
  cudaFree(d_mask);
  cudaFree(d_out);
}

}
