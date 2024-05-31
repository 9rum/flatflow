#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.cuh"

namespace flatflow {
namespace kernel {

// flatflow::kernel::ceil_div
//
// utility function to calculate grid size with given thread block size
template <class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

// flatflow::kernel::permute_kernel
//
// Current permute function is implemented to adhere huggingface qkv shape.
// query, key, value shape needs to be (batch, num_heads, sequence_length, dim)
// pre_split_input is concatenation of these three. (batch, sequence_length, 3
// ,num_heads , dim)
__global__ void permute_kernel(float *query, float *key, float *value,
                               const float *pre_split_input, const int batch,
                               const int sequence_length, const int num_heads,
                               const int dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch * sequence_length * num_heads * dim) {
    const int new_batch = idx / (num_heads * sequence_length * dim);
    int rest = idx % (num_heads * sequence_length * dim);
    const int new_num_heads = rest / (sequence_length * dim);
    rest = rest % (sequence_length * dim);
    const int new_sequence_length = rest / dim;
    const int new_dim = rest % dim;

    const int pre_idx = new_batch * num_heads * 3 * sequence_length * dim +
                        new_sequence_length * 3 * num_heads * dim +
                        new_num_heads * dim + new_dim;
    query[idx] = pre_split_input[pre_idx];
    key[idx] = pre_split_input[pre_idx + num_heads * dim];
    value[idx] = pre_split_input[pre_idx + 2 * num_heads * dim];
  }
}

// flatflow::kernel::unpermute_kernel
//
//  Current unpermute function is implemented to adhere huggingface qkv shape.
//  After attention output_ has shape (batch, num_heads, sequence_length, dim)
//  Unpermute it to (batch, sequence_length, num_heads, dim)
__global__ void unpermute_kernel(float *input_, float *output_, const int batch,
                                 const int sequence_length, const int num_heads,
                                 const int dim) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < batch * sequence_length * num_heads * dim) {
    const int new_batch = idx / (num_heads * sequence_length * dim);
    int rest = idx % (num_heads * sequence_length * dim);
    const int new_num_heads = rest / (sequence_length * dim);
    rest = rest % (sequence_length * dim);
    const int new_sequence_length = rest / dim;
    const int new_dim = rest % dim;

    const int dest = new_batch * num_heads * sequence_length * dim +
                     new_sequence_length * num_heads * dim +
                     new_num_heads * dim + new_dim;
    output_[dest] = input_[idx];
  }
}

__global__ void scale_kernel(float *pre_softmax, const float scale,
                             const int batch, const int num_heads,
                             const int sequence_length) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < batch * num_heads * sequence_length * sequence_length) {
    int rest = index % (num_heads * sequence_length * sequence_length);
    rest = rest % (sequence_length * sequence_length);
    int row = rest / sequence_length;
    int col = rest % sequence_length;
    if (col > row) {
      pre_softmax[index] = -INFINITY;
    } else {
      pre_softmax[index] *= scale;
    }
  }
}

// flatflow::kernel::attention_forward
//
// Attention forward function that computes MultiHead Attention.
// pre_split_input shape is (batch, sequence_length, 3*dim) since Linear.
// Operation is processed before this forward.
void attention_forward(float *output, float *post_attention,
                       float *pre_split_input, float *pre_attention,
                       float *attention, const float *input_, const int batch,
                       const int sequence_length, const int dim,
                       const int num_heads, const int block_size) {
  cublasHandle_t cublas_handle;
  const int head_size = dim / num_heads;
  float *query, *key, *value;
  query = pre_split_input + 0 * batch * sequence_length * dim;
  key = pre_split_input + 1 * batch * sequence_length * dim;
  value = pre_split_input + 2 * batch * sequence_length * dim;

  int total_thread = batch * sequence_length * dim;
  int num_blocks = ceil_div(total_thread, block_size);
  permute_kernel<<<num_blocks, block_size>>>(
      query, key, value, input_, batch, sequence_length, num_heads, head_size);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasCheck(cublasSgemmStridedBatched(
      cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, sequence_length, sequence_length,
      head_size, &alpha, key, head_size, sequence_length * head_size, query,
      head_size, sequence_length * head_size, &beta, pre_attention,
      sequence_length, sequence_length * sequence_length, batch * num_heads));

  const float scale = 1.0 / sqrtf(head_size);
  total_thread = batch * num_heads * sequence_length * sequence_length;
  num_blocks = ceil_div(total_thread, block_size);
  scale_kernel<<<num_blocks, block_size>>>(pre_attention, scale, batch,
                                           num_heads, sequence_length);

  const int softmax_block_size = 256;
  const int grid_size = batch * num_heads * sequence_length;
  const size_t shared_memory = 2 * softmax_block_size / 32 * sizeof(float);
  softmax_forward_kernel<<<grid_size, softmax_block_size, shared_memory>>>(
      attention, pre_attention, batch * num_heads * sequence_length,
      sequence_length);

  cublasCheck(cublasSgemmStridedBatched(
      cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, head_size, sequence_length,
      sequence_length, &alpha, value, head_size, sequence_length * head_size,
      attention, sequence_length, sequence_length * sequence_length, &beta,
      post_attention, head_size, sequence_length * head_size,
      batch * num_heads));

  num_blocks = ceil_div(batch * sequence_length * dim, block_size);
  unpermute_kernel<<<num_blocks, block_size>>>(
      post_attention, output, batch, sequence_length, num_heads, head_size);
}

}  // namespace kernel
}  // namespace flatflow