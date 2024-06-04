#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

namespace flatflow {
namespace torch {
namespace nn {
namespace modules {

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

std::vector<torch::Tensor> split_attention(
    torch::Tensor &QKV, const std::vector<int> &split_indices);

}  // namespace modules
}  // namespace nn
}  // namespace torch
}  // namespace flatflow
