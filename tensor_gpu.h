#pragma once

#include "tensor.h"

void matrix_mul(float *m1, float *m2, float *r, int m1w, int m2w, int rw, int rh);

Tensor gpu_tensor_add(const Tensor& a, const Tensor &b);