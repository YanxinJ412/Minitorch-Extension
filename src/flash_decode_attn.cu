/**
 * Flash-style single-step decoder attention over a contiguous FP32 KV cache.
 *
 * This kernel targets the old nonpaged KV cache layout:
 *   (batch, heads, seq_len, d_head)
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void flash_decode_attn_fp32_kernel(
    const float *Q, const float *K, const float *V, const float *mask, float *out,
    int B, int H, int L, int D, float inv_sqrt_d) {
  int bh = blockIdx.x;
  int b = bh / H;
  int h = bh % H;
  if (b >= B) {
    return;
  }

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
    for (int i = 1; i < L; ++i) {
      mv = fmaxf(mv, logits[i]);
    }
    float s = 0.f;
    for (int i = 0; i < L; ++i) {
      logits[i] = expf(logits[i] - mv);
      s += logits[i];
    }
    for (int i = 0; i < L; ++i) {
      logits[i] /= s;
    }
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

__global__ void flash_attn_fp32_kernel(
    const float *Q, const float *K, const float *V, const float *mask, float *out,
    int B, int H, int QL, int KL, int D, float inv_sqrt_d) {
  int bhq = blockIdx.x;
  int q_idx = bhq % QL;
  int bh = bhq / QL;
  int b = bh / H;
  int h = bh % H;
  if (b >= B) {
    return;
  }

  const float *q = Q + (((size_t)b * H + h) * QL + q_idx) * D;
  const float *k_row = K + ((size_t)b * H + h) * KL * D;
  const float *v_row = V + ((size_t)b * H + h) * KL * D;
  const float *msk = mask + (((size_t)b * H + h) * QL + q_idx) * KL;
  float *o = out + (((size_t)b * H + h) * QL + q_idx) * D;

  extern __shared__ float smem[];
  float *logits = smem;

  for (int j = threadIdx.x; j < KL; j += blockDim.x) {
    float sum = 0.f;
    for (int d = 0; d < D; ++d) {
      sum += q[d] * k_row[j * D + d];
    }
    logits[j] = sum * inv_sqrt_d + msk[j];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float mv = logits[0];
    for (int i = 1; i < KL; ++i) {
      mv = fmaxf(mv, logits[i]);
    }
    float s = 0.f;
    for (int i = 0; i < KL; ++i) {
      logits[i] = expf(logits[i] - mv);
      s += logits[i];
    }
    for (int i = 0; i < KL; ++i) {
      logits[i] /= s;
    }
  }
  __syncthreads();

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < KL; ++j) {
      acc += logits[j] * v_row[j * D + d];
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

void flash_decode_attn_host_fp32(
    const float *Q, const float *K, const float *V, const float *mask, float *out,
    int B, int H, int L, int D, float inv_sqrt_d) {
  size_t q_sz = (size_t)B * H * D * sizeof(float);
  size_t kv_sz = (size_t)B * H * L * D * sizeof(float);
  size_t m_sz = (size_t)B * H * L * sizeof(float);
  size_t o_sz = q_sz;

  float *d_Q, *d_K, *d_V, *d_mask, *d_out;
  chk(cudaMalloc((void **)&d_Q, q_sz), "cudaMalloc flash Q");
  chk(cudaMalloc((void **)&d_K, kv_sz), "cudaMalloc flash K");
  chk(cudaMalloc((void **)&d_V, kv_sz), "cudaMalloc flash V");
  chk(cudaMalloc((void **)&d_mask, m_sz), "cudaMalloc flash mask");
  chk(cudaMalloc((void **)&d_out, o_sz), "cudaMalloc flash out");

  chk(cudaMemcpy(d_Q, Q, q_sz, cudaMemcpyHostToDevice), "memcpy flash Q");
  chk(cudaMemcpy(d_K, K, kv_sz, cudaMemcpyHostToDevice), "memcpy flash K");
  chk(cudaMemcpy(d_V, V, kv_sz, cudaMemcpyHostToDevice), "memcpy flash V");
  chk(cudaMemcpy(d_mask, mask, m_sz, cudaMemcpyHostToDevice), "memcpy flash mask");

  int blocks = B * H;
  int threads = 256;
  size_t shmem = (size_t)L * sizeof(float);
  flash_decode_attn_fp32_kernel<<<blocks, threads, shmem>>>(
      d_Q, d_K, d_V, d_mask, d_out, B, H, L, D, inv_sqrt_d);
  chk(cudaDeviceSynchronize(), "sync flash fp32");

  chk(cudaMemcpy(out, d_out, o_sz, cudaMemcpyDeviceToHost), "memcpy flash out");

  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_mask);
  cudaFree(d_out);
}

void flash_attn_host_fp32(
    const float *Q, const float *K, const float *V, const float *mask, float *out,
    int B, int H, int QL, int KL, int D, float inv_sqrt_d) {
  size_t q_sz = (size_t)B * H * QL * D * sizeof(float);
  size_t kv_sz = (size_t)B * H * KL * D * sizeof(float);
  size_t m_sz = (size_t)B * H * QL * KL * sizeof(float);
  size_t o_sz = q_sz;

  float *d_Q, *d_K, *d_V, *d_mask, *d_out;
  chk(cudaMalloc((void **)&d_Q, q_sz), "cudaMalloc flash full Q");
  chk(cudaMalloc((void **)&d_K, kv_sz), "cudaMalloc flash full K");
  chk(cudaMalloc((void **)&d_V, kv_sz), "cudaMalloc flash full V");
  chk(cudaMalloc((void **)&d_mask, m_sz), "cudaMalloc flash full mask");
  chk(cudaMalloc((void **)&d_out, o_sz), "cudaMalloc flash full out");

  chk(cudaMemcpy(d_Q, Q, q_sz, cudaMemcpyHostToDevice), "memcpy flash full Q");
  chk(cudaMemcpy(d_K, K, kv_sz, cudaMemcpyHostToDevice), "memcpy flash full K");
  chk(cudaMemcpy(d_V, V, kv_sz, cudaMemcpyHostToDevice), "memcpy flash full V");
  chk(cudaMemcpy(d_mask, mask, m_sz, cudaMemcpyHostToDevice), "memcpy flash full mask");

  int blocks = B * H * QL;
  int threads = 256;
  size_t shmem = (size_t)KL * sizeof(float);
  flash_attn_fp32_kernel<<<blocks, threads, shmem>>>(
      d_Q, d_K, d_V, d_mask, d_out, B, H, QL, KL, D, inv_sqrt_d);
  chk(cudaDeviceSynchronize(), "sync flash full fp32");

  chk(cudaMemcpy(out, d_out, o_sz, cudaMemcpyDeviceToHost), "memcpy flash full out");

  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_mask);
  cudaFree(d_out);
}

}
