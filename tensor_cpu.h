#ifndef TENSOR_CPU_H
#define TENSOR_CPU_H

#include "tensor.h"

float cpu_dot(const Tensor &a, const Tensor &b);

Tensor cpu_tensor_add(const Tensor& a, const Tensor &b);
Tensor cpu_tensor_mul_2d(const Tensor& a, const Tensor &b);
Tensor cpu_tensor_mul(const Tensor& a, float scalar);

float *alloc_ram(int n_elems);
void free_ram(float *ptr);

#endif