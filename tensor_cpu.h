#ifndef TENSOR_CPU_H
#define TENSOR_CPU_H

#include "tensor.h"

float cpu_dot(const Tensor &a, const Tensor &b);

Tensor cpu_tensor_add(const Tensor& a, const Tensor &b);
Tensor cpu_tensor_mul_mat2d_mat2d(const Tensor& a, const Tensor &b);
Tensor cpu_tensor_mul_mat2d_vec(const Tensor &mat, const Tensor &vec);
Tensor cpu_tensor_mul(const Tensor& a, float scalar);
Tensor cpu_tensor_pow(const Tensor& a, float power);
Tensor cpu_pointwise_mul(const Tensor& a, const Tensor &b);
Tensor cpu_tensor_tanh(const Tensor &a);
Tensor cpu_tensor_relu(const Tensor &a);
Tensor cpu_tensor_sin(const Tensor &a);
Tensor cpu_tensor_cos(const Tensor &a);
Tensor cpu_tensor_outer(const Tensor &a, const Tensor &b);
void cpu_mat2d_transpose(Tensor &a);

float *alloc_ram(int n_elems);
void free_ram(float *ptr);

#endif