#include <cublas_v2.h>
#inlcude <iostream>

namespace flatflow {
namespace torch {
namespace nn {
namespace modules {

// flatflow::kernel::cuda_check
//
// utility function to check CUDA errors
inline void cuda_check(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    std::cout << "CUDA error at " << file << ":" << line << " : "
              << cudaGetErrorString(error) << std::endl;
    exit(EXIT_FAILURE);
  }
};

// flatflow::kernel::cublas_check
//
// utility function to check cuBLAS status errors
inline void cublas_check(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "cuBLAS error at " << file << ":" << line << " : "
              << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

// flatflow::kernel::split_attention
//
// Split input tensor into multiple tensors based on the split indices
std::vector<torch::Tensor> split_attention(
    torch::Tensor &QKV, const std::vector<int> &split_indices);

}  // namespace modules
}  // namespace nn
}  // namespace torch
}  // namespace flatflow
