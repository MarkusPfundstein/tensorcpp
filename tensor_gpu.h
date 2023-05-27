#ifndef TENSOR_GPU_H
#define TENSOR_GPU_H

#include "tensor.h"

float gpu_dot(const Tensor &a, const Tensor &b);
Tensor gpu_tensor_add(const Tensor& a, const Tensor &b);

float *alloc_gpu(int nelems);
void free_gpu(float *mem);

Tensor gpu_tensor_mul_mat2d_mat2d(const Tensor &a, const Tensor &b);
Tensor gpu_tensor_mul_mat2d_vec(const Tensor &mat, const Tensor &vec);
Tensor gpu_tensor_mul(const Tensor &a, float scalar);

#endif