#include "flatflow/aten/native/attention.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <thread>
#include <vector>

#include "torch/extension.h"
#include "torch/torch.h"
#include "torch/types.h"

#define WARP_SIZE 32

namespace flatflow {
namespace aten {
namespace native {

// flatflow::aten::native::ceil_div
//
// utility function to calculate grid size with given thread block size
template <class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

// flatflow::aten::native::warp_reduce_max
//
// return max value in warp-level reduction
__device__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

// flatflow::aten::native::warp_reduce_sum
//
// return sum in warp-level reduction
__device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

// flatflow::aten::native::cuda_check
//
// utility function to check CUDA errors
inline void cuda_check(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line << " : "
              << cudaGetErrorString(error) << std::endl;
    exit(EXIT_FAILURE);
  }
};

// flatflow::aten::native::cublas_check
//
// utility function to check cuBLAS status errors
inline void cublas_check(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error at " << file << ":" << line << " : " << status
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

// flatflow::aten::native::permute_kernel
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

// flatflow::aten::native::unpermute_kernel
//
// Current unpermute function is implemented to adhere huggingface qkv shape.
// After attention output_ has shape (batch, num_heads, sequence_length, dim)
// Unpermute it to (batch, sequence_length, num_heads, dim)
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

// flatflow::aten::native::scale_kernel
//
// scale_kernel scales pre_softmax with scale factor.
// For lookahead mask, it sets upper triangular part to -INFINITY.
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

// flatflow::aten::native::softmax_forward_kernel
//
// softmax_forward_kernel computes softmax of input.
// By using shared memory, softmax calculation can share results between warps.
__global__ void softmax_forward_kernel(float *post_softmax,
                                       const float *pre_softmax, int num_blocks,
                                       int sequence_length) {
  extern __shared__ float shared[];
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int warps_per_block = blockDim.x / WARP_SIZE;

  float *max_vals = shared;
  float *sum_vals = &shared[warps_per_block];
  const float *x = pre_softmax + block_id * sequence_length;
  float max_val = -INFINITY;
  for (std::size_t i = thread_id; i < sequence_length; i += blockDim.x) {
    max_val = fmaxf(max_val, x[i]);
  }
  max_val = warp_reduce_max(max_val);

  if (lane_id == 0) {
    max_vals[warp_id] = max_val;
  }
  __syncthreads();
  if (thread_id == 0) {
    float val = max_vals[thread_id];
    for (std::size_t i = 1; i < warps_per_block; ++i) {
      val = fmaxf(val, max_vals[i]);
    }
    max_vals[0] = val;
  }
  __syncthreads();

  const float offset = max_vals[0];
  for (std::size_t i = thread_id; i < sequence_length; i += blockDim.x) {
    post_softmax[block_id * sequence_length + i] = expf(x[i] - offset);
  }

  x = post_softmax + block_id * sequence_length;
  float sum_val = 0.0f;
  for (std::size_t i = thread_id; i < sequence_length; i += blockDim.x) {
    sum_val += x[i];
  }

  sum_val = warp_reduce_sum(sum_val);

  if (lane_id == 0) {
    sum_vals[warp_id] = sum_val;
  }
  __syncthreads();

  if (thread_id == 0) {
    float val = sum_vals[thread_id];
    for (std::size_t i = 1; i < warps_per_block; ++i) {
      val += sum_vals[i];
    }
    sum_vals[0] = val;
  }
  __syncthreads();

  const float sum = sum_vals[0];
  for (std::size_t i = thread_id; i < sequence_length; i += blockDim.x) {
    post_softmax[block_id * sequence_length + i] = x[i] / sum;
  }
}

// flatflow::aten::native::attention_forward
//
// Attention forward function that computes MultiHead Attention.
// pre_split_input shape is (batch, sequence_length, 3*dim) since Linear.
// Operation is processed before this forward.
void attention_forward(cublasHandle_t cublas_handle, float *output,
                       float *post_attention, float *pre_split_input,
                       float *pre_attention, float *attention,
                       const float *input_, const int batch,
                       const int sequence_length, const int dim,
                       const int num_heads, const int block_size,
                       cudaStream_t stream) {
  const int head_size = dim / num_heads;
  float *query, *key, *value;
  query = pre_split_input + 0 * batch * sequence_length * dim;
  key = pre_split_input + 1 * batch * sequence_length * dim;
  value = pre_split_input + 2 * batch * sequence_length * dim;

  int total_thread = batch * sequence_length * dim;
  int num_blocks = ceil_div(total_thread, block_size);
  permute_kernel<<<num_blocks, block_size, 0, stream>>>(
      query, key, value, input_, batch, sequence_length, num_heads, head_size);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublas_check(
      cublasSgemmStridedBatched(
          cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, sequence_length,
          sequence_length, head_size, &alpha, key, head_size,
          sequence_length * head_size, query, head_size,
          sequence_length * head_size, &beta, pre_attention, sequence_length,
          sequence_length * sequence_length, batch * num_heads),
      __FILE__, __LINE__);

  const float scale = 1.0 / sqrtf(head_size);
  total_thread = batch * num_heads * sequence_length * sequence_length;
  num_blocks = ceil_div(total_thread, block_size);
  scale_kernel<<<num_blocks, block_size, 0, stream>>>(
      pre_attention, scale, batch, num_heads, sequence_length);

  const int softmax_block_size = 256;
  const int grid_size = batch * num_heads * sequence_length;
  const size_t shared_memory =
      2 * softmax_block_size / WARP_SIZE * sizeof(float);
  softmax_forward_kernel<<<grid_size, softmax_block_size, shared_memory,
                           stream>>>(attention, pre_attention,
                                     batch * num_heads * sequence_length,
                                     sequence_length);

  cublas_check(cublasSgemmStridedBatched(
                   cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, head_size,
                   sequence_length, sequence_length, &alpha, value, head_size,
                   sequence_length * head_size, attention, sequence_length,
                   sequence_length * sequence_length, &beta, post_attention,
                   head_size, sequence_length * head_size, batch * num_heads),
               __FILE__, __LINE__);

  num_blocks = ceil_div(batch * sequence_length * dim, block_size);
  unpermute_kernel<<<num_blocks, block_size, stream>>>(
      post_attention, output, batch, sequence_length, num_heads, head_size);
}

// flatflow::aten::native::split_attention
//
// Divide batched QKV from Merged QKV linear with given index to split tensor.
// Intialize cublas handle and cuda stream for each splited tensor in order to
// parallelize multiple attentions.
std::vector<torch::Tensor> split_attention(
    torch::Tensor &QKV, const std::vector<std::size_t> &split_indices) {
  TORCH_INTERNAL_ASSERT(QKV.device().type() == torch::DeviceType::CUDA);

  torch::Tensor QKV_contiguous = QKV.contiguous();
  std::vector<torch::Tensor> split_tensors;
  std::vector<std::thread> attention_thread;
  std::vector<cudaStream_t> streams(split_tensors.size());
  std::vector<cublasHandle_t> handles(split_tensors.size());
  std::vector<torch::Tensor> outputs(split_tensors.size());
  int start = 0;
  const int block_size = 256;
  const int num_heads = 12;

  std::vector<float *> post_attentions;
  std::vector<float *> pre_split_inputs;
  std::vector<float *> pre_attentions;
  std::vector<float *> attentions;
  std::vector<float *> inputs;

  for (int end : split_indices) {
    auto sliced_tensor = QKV_contiguous.narrow(0, start, end - start);
    split_tensors.push_back(sliced_tensor);
    outputs.push_back(
        torch::empty(sliced_tensor.sizes(), sliced_tensor.options));
    start = end;
    cudaStreamCreate(&streams[i]);
    cublasCreate(&handles[i]);

    auto sequence_length = sliced_tensor.sizes()[0];
    auto hidden_dim = sliced_tensor.sizes()[1] / 3;

    cuda_check(cudaMalloc(&post_attentions[i],
                          sequence_length * hidden_dim * sizeof(float)),
               __FILE__, __LINE__);
    cuda_check(cudaMalloc(&pre_split_inputs[i],
                          sequence_length * 3 * hidden_dim * sizeof(float)),
               __FILE__, __LINE__);
    cuda_check(
        cudaMalloc(&pre_attentions[i], num_heads * sequence_length *
                                           sequence_length * sizeof(float)),
        __FILE__, __LINE__);
    cuda_check(cudaMalloc(&attentions[i], num_heads * sequence_length *
                                              sequence_length * sizeof(float)),
               __FILE__, __LINE__);
    cuda_check(cudaMalloc(&inputs[i],
                          sequence_length * 3 * hidden_dim * sizeof(float)),
               __FILE__, __LINE__);
    cuda_check(cudaMemcpy(inputs[i], split_tensors,
                          sequence_length * 3 * hidden_dim * sizeof(float),
                          cudaMemcpyHostToDevice),
               __FILE__, __LINE__);
  }

  for (std::size_t i = 0; i < split_tensors.size(); ++i) {
    attention_thread.push_back(
        std::thread(attention_forward, handles[i], outputs[i].data_ptr<float>(),
                    post_attentions[i], pre_split_inputs[i], pre_attentions[i],
                    attentions[i], inputs[i], 1, sequence_length, hidden_dim,
                    num_heads, block_size, streams[i]));
  }
  for (std::size_t i = 0; i < attention_forward.size(); ++i) {
    attention_forward[i].join();
  }

  for (auto &stream : streams) {
    cudaStreamSynchronize(stream);
  }
  for (auto &handle : handles) {
    cublasDestroy(handle);
  }
  for (auto &stream : streams) {
    cudaStreamDestroy(stream);
  }

  for (std::size_t i = 0; i < inputs.size(); ++i) {
    cuda_check(cudaFree(post_attentions[i]), __FILE__, __LINE__);
    cuda_check(cudaFree(pre_split_inputs[i]), __FILE__, __LINE__);
    cuda_check(cudaFree(pre_attentions[i]), __FILE__, __LINE__);
    cuda_check(cudaFree(attentions[i]), __FILE__, __LINE__);
    cuda_check(cudaFree(inputs[i]), __FILE__, __LINE__);
  }

  return outputs;
}

}  // namespace native
}  // namespace aten
}  // namespace flatflow

#undef WARP_SIZE
