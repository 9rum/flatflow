#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

namespace flatflow {
namespace kernel {

// flatflow::kernel::ceil_div
//
// utility function to calculate grid size with given thread block size
template <class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

// flatflow::kernel::cuda_check
//
// utility function to check CUDA errors
void cuda_check(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

// flatflow::kernel::cublas_check
//
// utility function to check cuBLAS status errors
void cublas_check(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
    exit(EXIT_FAILURE);
  }
}

#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))
#define cublasCheck(status) \
  { cublas_check((status), __FILE__, __LINE__); }

// flatflow::kernel::permute_kernel
//
// Current permute function is implemented to adhere huggingface qkv shape
// query, key, value shape needs to be (batch, num_heads, sequence_length, dim)
// pre_split_input is concatenation of these three. (batch, sequence_length, 3 ,num_heads , dim)
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

//flatflow::kernel::unpermute_kernel
//
// Current unpermute function is implemented to adhere huggingface qkv shape after attention
// output_ has shape (batch, num_heads, sequence_length, dim) but we need to unpermute it to (batch, sequence_length, num_heads, dim)
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

}  // namespace kernel
}  // namespace flatflow