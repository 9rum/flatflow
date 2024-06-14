#ifndef FLATFLOW_ATEN_NATIVE_ATTENTION_H_
#define FLATFLOW_ATEN_NATIVE_ATTENTION_H_

#include <cublas_v2.h>

#include <iostream>

namespace flatflow {
namespace aten {
namespace native {

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

// flatflow::aten::native::split_attention
//
// Split input tensor into multiple tensors based on the split indices
std::vector<torch::Tensor> split_attention(
    torch::Tensor &QKV, const std::vector<int> &split_indices);

}  // namespace native
}  // namespace aten
}  // namespace flatflow

#endif  // FLATFLOW_ATEN_NATIVE_ATTENTION_H_
