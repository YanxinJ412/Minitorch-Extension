/**
 * Fused single-step decoder attention: for query length 1, compute
 * softmax(Q K^T / sqrt(d) + mask) V in one kernel.
 * KV may be float32 or int8 (per-tensor scale), matching LayerKVCache quantization.
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define FUSED_MAX_SEQ 1024

__global__ void fused_decode_attn_fp32_kernel(
    const float *Q, const float *K, const float *V, const float *mask, float *out,
    int B, int H, int L, int D, float inv_sqrt_d) {
  int bh = blockIdx.x;
  int b = bh / H;
  int h = bh % H;
  if (b >= B)
    return;

  const float *q = Q + (b * H + h) * D;
  const float *k_row = K + (b * H + h) * L * D;
  const float *v_row = V + (b * H + h) * L * D;
  const float *msk = mask + (b * H + h) * L;
  float *o = out + (b * H + h) * D;

  extern __shared__ float smem[];
  float *logits = smem;

  for (int j = threadIdx.x; j < L; j += blockDim.x) {
    float sum = 0.f;
    for (int d = 0; d < D; ++d) {
      sum += q[d] * k_row[j * D + d];
    }
    logits[j] = sum * inv_sqrt_d + msk[j];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float mv = logits[0];
    for (int i = 1; i < L; ++i)
      mv = fmaxf(mv, logits[i]);
    float s = 0.f;
    for (int i = 0; i < L; ++i) {
      logits[i] = expf(logits[i] - mv);
      s += logits[i];
    }
    for (int i = 0; i < L; ++i)
      logits[i] /= s;
  }
  __syncthreads();

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < L; ++j) {
      acc += logits[j] * v_row[j * D + d];
    }
    o[d] = acc;
  }
}

__global__ void fused_decode_attn_int8_kv_kernel(
    const float *Q, const signed char *K, const signed char *V, const float *mask,
    float *out, int B, int H, int L, int D, float inv_sqrt_d, float k_scale,
    float v_scale) {
  int bh = blockIdx.x;
  int b = bh / H;
  int h = bh % H;
  if (b >= B)
    return;

  const float *q = Q + (b * H + h) * D;
  const signed char *k_row = K + (b * H + h) * L * D;
  const signed char *v_row = V + (b * H + h) * L * D;
  const float *msk = mask + (b * H + h) * L;
  float *o = out + (b * H + h) * D;

  extern __shared__ float smem[];
  float *logits = smem;

  for (int j = threadIdx.x; j < L; j += blockDim.x) {
    float sum = 0.f;
    for (int d = 0; d < D; ++d) {
      sum += q[d] * (float)k_row[j * D + d] * k_scale;
    }
    logits[j] = sum * inv_sqrt_d + msk[j];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float mv = logits[0];
    for (int i = 1; i < L; ++i)
      mv = fmaxf(mv, logits[i]);
    float s = 0.f;
    for (int i = 0; i < L; ++i) {
      logits[i] = expf(logits[i] - mv);
      s += logits[i];
    }
    for (int i = 0; i < L; ++i)
      logits[i] /= s;
  }
  __syncthreads();

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < L; ++j) {
      acc += logits[j] * ((float)v_row[j * D + d] * v_scale);
    }
    o[d] = acc;
  }
}

__device__ __forceinline__ float load_int4_dequant(const unsigned char *packed,
                                                   int linear_idx,
                                                   float scale) {
  // Packed layout matches kv_cache.py _pack_int4: unsigned nibble (0..15) storing signed int4 (nibble-8).
  int byte_idx = linear_idx >> 1;
  unsigned char byte = packed[byte_idx];
  unsigned char nibble = (linear_idx & 1) ? (byte >> 4) : (byte & 0x0F);
  signed char s = (signed char)((int)nibble - 8);
  return ((float)s) * scale;
}

__global__ void fused_decode_attn_int4_kv_kernel(
    const float *Q, const unsigned char *K, const unsigned char *V, const float *mask,
    float *out, int B, int H, int L, int D, float inv_sqrt_d, float k_scale,
    float v_scale) {
  int bh = blockIdx.x;
  int b = bh / H;
  int h = bh % H;
  if (b >= B)
    return;

  const float *q = Q + (b * H + h) * D;
  const float *msk = mask + (b * H + h) * L;
  float *o = out + (b * H + h) * D;

  extern __shared__ float smem[];
  float *logits = smem;

  // Base linear offset for this (b,h) in logical (B,H,L,D) layout.
  int bh_base = (b * H + h) * L * D;

  for (int j = threadIdx.x; j < L; j += blockDim.x) {
    float sum = 0.f;
    int row_base = bh_base + j * D;
    for (int d = 0; d < D; ++d) {
      sum += q[d] * load_int4_dequant(K, row_base + d, k_scale);
    }
    logits[j] = sum * inv_sqrt_d + msk[j];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float mv = logits[0];
    for (int i = 1; i < L; ++i)
      mv = fmaxf(mv, logits[i]);
    float s = 0.f;
    for (int i = 0; i < L; ++i) {
      logits[i] = expf(logits[i] - mv);
      s += logits[i];
    }
    for (int i = 0; i < L; ++i)
      logits[i] /= s;
  }
  __syncthreads();

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < L; ++j) {
      int lin = bh_base + j * D + d;
      acc += logits[j] * load_int4_dequant(V, lin, v_scale);
    }
    o[d] = acc;
  }
}

extern "C" {

void launch_fused_decode_attn_fp32(const float *Q, const float *K, const float *V,
                                     const float *mask, float *out, int B, int H, int L,
                                     int D, float inv_sqrt_d, cudaStream_t stream) {
  int bh = B * H;
  int threads = 256;
  size_t shmem = (size_t)L * sizeof(float);
  fused_decode_attn_fp32_kernel<<<bh, threads, shmem, stream>>>(
      Q, K, V, mask, out, B, H, L, D, inv_sqrt_d);
}

void launch_fused_decode_attn_int8_kv(const float *Q, const signed char *K,
                                     const signed char *V, const float *mask, float *out,
                                     int B, int H, int L, int D, float inv_sqrt_d,
                                     float k_scale, float v_scale, cudaStream_t stream) {
  int bh = B * H;
  int threads = 256;
  size_t shmem = (size_t)L * sizeof(float);
  fused_decode_attn_int8_kv_kernel<<<bh, threads, shmem, stream>>>(
      Q, K, V, mask, out, B, H, L, D, inv_sqrt_d, k_scale, v_scale);
}

void launch_fused_decode_attn_int4_kv(const float *Q, const unsigned char *K,
                                     const unsigned char *V, const float *mask, float *out,
                                     int B, int H, int L, int D, float inv_sqrt_d,
                                     float k_scale, float v_scale, cudaStream_t stream) {
  int bh = B * H;
  int threads = 256;
  size_t shmem = (size_t)L * sizeof(float);
  fused_decode_attn_int4_kv_kernel<<<bh, threads, shmem, stream>>>(
      Q, K, V, mask, out, B, H, L, D, inv_sqrt_d, k_scale, v_scale);
}

static void chk(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

/** Host wrapper: alloc GPU, H2D, launch, D2H, free — matches softmax_kernel style. */
void fused_decode_attn_host_fp32(const float *Q, const float *K, const float *V,
                                 const float *mask, float *out, int B, int H, int L,
                                 int D, float inv_sqrt_d) {
  size_t q_sz = (size_t)B * H * D * sizeof(float);
  size_t kv_sz = (size_t)B * H * L * D * sizeof(float);
  size_t m_sz = (size_t)B * H * L * sizeof(float);
  size_t o_sz = q_sz;

  float *d_Q, *d_K, *d_V, *d_mask, *d_out;
  chk(cudaMalloc((void **)&d_Q, q_sz), "cudaMalloc Q");
  chk(cudaMalloc((void **)&d_K, kv_sz), "cudaMalloc K");
  chk(cudaMalloc((void **)&d_V, kv_sz), "cudaMalloc V");
  chk(cudaMalloc((void **)&d_mask, m_sz), "cudaMalloc mask");
  chk(cudaMalloc((void **)&d_out, o_sz), "cudaMalloc out");

  chk(cudaMemcpy(d_Q, Q, q_sz, cudaMemcpyHostToDevice), "memcpy Q");
  chk(cudaMemcpy(d_K, K, kv_sz, cudaMemcpyHostToDevice), "memcpy K");
  chk(cudaMemcpy(d_V, V, kv_sz, cudaMemcpyHostToDevice), "memcpy V");
  chk(cudaMemcpy(d_mask, mask, m_sz, cudaMemcpyHostToDevice), "memcpy mask");

  cudaStream_t stream = 0;
  launch_fused_decode_attn_fp32(d_Q, d_K, d_V, d_mask, d_out, B, H, L, D, inv_sqrt_d,
                                stream);
  chk(cudaDeviceSynchronize(), "sync fp32");

  chk(cudaMemcpy(out, d_out, o_sz, cudaMemcpyDeviceToHost), "memcpy out");

  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_mask);
  cudaFree(d_out);
}

void fused_decode_attn_host_int8(const float *Q, const signed char *K,
                                 const signed char *V, const float *mask, float *out,
                                 int B, int H, int L, int D, float inv_sqrt_d,
                                 float k_scale, float v_scale) {
  size_t q_sz = (size_t)B * H * D * sizeof(float);
  size_t kv_sz = (size_t)B * H * L * D * sizeof(signed char);
  size_t m_sz = (size_t)B * H * L * sizeof(float);
  size_t o_sz = q_sz;

  float *d_Q;
  signed char *d_K, *d_V;
  float *d_mask, *d_out;
  chk(cudaMalloc((void **)&d_Q, q_sz), "cudaMalloc Q");
  chk(cudaMalloc((void **)&d_K, kv_sz), "cudaMalloc K int8");
  chk(cudaMalloc((void **)&d_V, kv_sz), "cudaMalloc V int8");
  chk(cudaMalloc((void **)&d_mask, m_sz), "cudaMalloc mask");
  chk(cudaMalloc((void **)&d_out, o_sz), "cudaMalloc out");

  chk(cudaMemcpy(d_Q, Q, q_sz, cudaMemcpyHostToDevice), "memcpy Q");
  chk(cudaMemcpy(d_K, K, kv_sz, cudaMemcpyHostToDevice), "memcpy K");
  chk(cudaMemcpy(d_V, V, kv_sz, cudaMemcpyHostToDevice), "memcpy V");
  chk(cudaMemcpy(d_mask, mask, m_sz, cudaMemcpyHostToDevice), "memcpy mask");

  cudaStream_t stream = 0;
  launch_fused_decode_attn_int8_kv(d_Q, d_K, d_V, d_mask, d_out, B, H, L, D, inv_sqrt_d,
                                   k_scale, v_scale, stream);
  chk(cudaDeviceSynchronize(), "sync int8");

  chk(cudaMemcpy(out, d_out, o_sz, cudaMemcpyDeviceToHost), "memcpy out");

  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_mask);
  cudaFree(d_out);
}

void fused_decode_attn_host_int4(const float *Q, const unsigned char *K,
                                 const unsigned char *V, const float *mask, float *out,
                                 int B, int H, int L, int D, float inv_sqrt_d,
                                 float k_scale, float v_scale) {
  size_t q_sz = (size_t)B * H * D * sizeof(float);
  size_t total_elems = (size_t)B * H * L * D;
  size_t kv_sz = ((total_elems + 1) / 2) * sizeof(unsigned char);
  size_t m_sz = (size_t)B * H * L * sizeof(float);
  size_t o_sz = q_sz;

  float *d_Q;
  unsigned char *d_K, *d_V;
  float *d_mask, *d_out;
  chk(cudaMalloc((void **)&d_Q, q_sz), "cudaMalloc Q");
  chk(cudaMalloc((void **)&d_K, kv_sz), "cudaMalloc K int4");
  chk(cudaMalloc((void **)&d_V, kv_sz), "cudaMalloc V int4");
  chk(cudaMalloc((void **)&d_mask, m_sz), "cudaMalloc mask");
  chk(cudaMalloc((void **)&d_out, o_sz), "cudaMalloc out");

  chk(cudaMemcpy(d_Q, Q, q_sz, cudaMemcpyHostToDevice), "memcpy Q");
  chk(cudaMemcpy(d_K, K, kv_sz, cudaMemcpyHostToDevice), "memcpy K");
  chk(cudaMemcpy(d_V, V, kv_sz, cudaMemcpyHostToDevice), "memcpy V");
  chk(cudaMemcpy(d_mask, mask, m_sz, cudaMemcpyHostToDevice), "memcpy mask");

  cudaStream_t stream = 0;
  launch_fused_decode_attn_int4_kv(d_Q, d_K, d_V, d_mask, d_out, B, H, L, D, inv_sqrt_d,
                                  k_scale, v_scale, stream);
  chk(cudaDeviceSynchronize(), "sync int4");

  chk(cudaMemcpy(out, d_out, o_sz, cudaMemcpyDeviceToHost), "memcpy out");

  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_mask);
  cudaFree(d_out);
}
}
