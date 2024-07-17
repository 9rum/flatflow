#ifndef FLATFLOW_ATEN_NATIVE_ATTENTION_H_
#define FLATFLOW_ATEN_NATIVE_ATTENTION_H_

#include <cublas_v2.h>

#include <iostream>

namespace flatflow {
namespace aten {
namespace native {
  
// flatflow::aten::native::split_attention
//
// Split input tensor into multiple tensors based on the split indices
std::vector<torch::Tensor> split_attention(
    torch::Tensor &QKV, const std::vector<int> &split_indices);

}  // namespace native
}  // namespace aten
}  // namespace flatflow

#endif  // FLATFLOW_ATEN_NATIVE_ATTENTION_H_
