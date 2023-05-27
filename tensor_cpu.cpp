#include <stdexcept>
#include <stdlib.h>
#include "tensor_cpu.h"

float cpu_dot(const Tensor &a, const Tensor &b)
{
    if (a.dimensions() != 1 || b.dimensions() != 1) {
        throw std::runtime_error("dot product defined only for tensors of dim 1");
    }
    if (a.shape[0] != b.shape[0]) {
        throw std::runtime_error("tensors must have same number of elements");
    }

    float sum = 0.0;
    for (int ai = 0; ai < a.shape[0]; ++ai) {
        sum += a.memory[ai] * b.memory[ai];
    }
    return sum;
}

Tensor cpu_tensor_add(const Tensor &a, const Tensor &b)
{
    Tensor t(a.shape);
    for (unsigned long int i = 0; i < a.nelems; ++i) {
        t.memory[i] = a.memory[i] + b.memory[i];
    }
    return t;
}

Tensor cpu_tensor_mul_mat2d_mat2d(const Tensor& a, const Tensor &b)
{
    if (a.shape[1] != b.shape[0]) {
        throw std::runtime_error("Invalid shapes of Tensor a & b. Cannot multiply");
    }

    Tensor out({a.shape[0], b.shape[1]});

    for (int i = 0; i < out.shape[0]; ++i) {
        for (int j = 0; j < out.shape[1]; ++j) {
            float sum = 0.0;
            for (int ai = 0; ai < a.shape[1]; ++ai) {
                float aval = a.memory[i * a.shape[1] + ai];
                float bval = b.memory[ai * b.shape[1] + j];
                sum += aval * bval;
            }
            out.memory[i * out.shape[1] + j] = sum;
        }
    }

    return out;
}

Tensor cpu_tensor_mul_mat2d_vec(const Tensor &a, const Tensor &b)
{
    if (a.shape[1] != b.shape[0]) {
        throw std::runtime_error("Invalid shapes of Tensor a & b. Cannot multiply");
    }

    Tensor out({a.shape[0]});

    for (int i = 0; i < out.shape[0]; ++i) {
            float sum = 0.0;
            for (int ai = 0; ai < a.shape[1]; ++ai) {
                float aval = a.memory[i * a.shape[1] + ai];
                float bval = b.memory[ai];
                sum += aval * bval;
            }
            out.memory[i] = sum;
    }

    return out;
}

Tensor cpu_tensor_mul(const Tensor& a, float scalar)
{
    Tensor t(a.shape);
    for (unsigned long int i = 0; i < a.nelems; ++i) {
        t.memory[i] = a.memory[i] * scalar;
    }
    return t;
}

float *alloc_ram(int n_elems)
{
    return (float*)malloc(sizeof(float) * n_elems);
}

void free_ram(float *ptr)
{
    free(ptr);
}
