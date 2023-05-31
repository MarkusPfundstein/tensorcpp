#include <stdexcept>
#include <stdlib.h>
#include <math.h>
#include <sstream>
#include <cassert>
#include <bits/stdc++.h>
#define HASH_SIZE 128
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
        std::stringstream ss;
        ss << "cpu_tensor_mul_mat2d_mat2d: invalid shapes ";
        ss << "[" << a.shape[0] << "," << a.shape[1] << "] and [" << b.shape[0] << "," << b.shape[1] << "]";
        throw std::runtime_error(ss.str());
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

Tensor cpu_tensor_mul_mat2d_vec(const Tensor &mat, const Tensor &vec)
{
    if (mat.shape[0] != vec.shape[0]) {
        std::stringstream ss;
        ss << "cpu_tensor_mul_mat2d_vec: invalid shapes ";
        ss << "[" << mat.shape[0] << "," << mat.shape[1] << "] and [" << vec.shape[0] << "]";
        throw std::runtime_error(ss.str());
    }

    Tensor out({mat.shape[1]});

    for (int i = 0; i < out.shape[0]; ++i) {
            float sum = 0.0;
            for (int ai = 0; ai < mat.shape[0]; ++ai) {
                float aval = mat.memory[ai * mat.shape[1] + i];
                float bval = vec.memory[ai];
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

Tensor cpu_tensor_pow(const Tensor& a, float power)
{
    Tensor t(a.shape);
    for (unsigned long int i = 0; i < a.nelems; ++i) {
        t.memory[i] = std::pow(a.memory[i], power);
    }
    return t;
}

Tensor cpu_tensor_tanh(const Tensor& a)
{
    Tensor t(a.shape);
    for (unsigned long int i = 0; i < a.nelems; ++i) {
        t.memory[i] = std::tanh(a.memory[i]);
    }
    return t;
}

Tensor cpu_pointwise_mul(const Tensor& a, const Tensor &b)
{
    if (a.shape != b.shape) {
        throw std::runtime_error("cpu_pointwise_mul error: a.shape != b.shape");
    }
    Tensor t(a.shape);

    for (unsigned long i = 0; i < a.nelems; ++i) {
        t.memory[i] = a.memory[i] * b.memory[i];
    }

    return t;
}

Tensor cpu_tensor_relu(const Tensor &a)
{
    Tensor t(a.shape);
    for (unsigned long int i = 0; i < a.nelems; ++i) {
        t.memory[i] = a.memory[i] > 0.0 ? a.memory[i] : 0.0;
    }
    return t;
}

Tensor cpu_tensor_sin(const Tensor &a)
{
    Tensor t(a.shape);
    for (unsigned long int i = 0; i < a.nelems; ++i) {
        t.memory[i] = std::sin(a.memory[i]);
    }
    return t;
}

Tensor cpu_tensor_cos(const Tensor &a)
{
    Tensor t(a.shape);
    for (unsigned long int i = 0; i < a.nelems; ++i) {
        t.memory[i] = std::cos(a.memory[i]);
    }
    return t;
}

Tensor cpu_tensor_outer(const Tensor &a, const Tensor &b)
{
    Tensor t({a.shape[0], b.shape[0]});

    for (int m = 0; m < a.shape[0]; ++m) {
        for (int n = 0; n < b.shape[0]; ++n) {
            t.memory[m * t.shape[1] + n] = a.memory[m] * b.memory[n];
        }
    }

    return t;
}

void cpu_mat2d_transpose(Tensor &a)
{
    assert(a.is_mat2d());

    int r = a.shape[0];
    int c = a.shape[1];

    int size = r*c - 1;
    float t; // holds element to be replaced, eventually becomes next element to move
    int next; // location of 't' to be moved
    int cycleBegin; // holds start of cycle
    int i; // iterator
    std::bitset<HASH_SIZE> b; // hash to mark moved elements
    
    b.reset();
    b[0] = b[size] = 1;
    i = 1; // Note that A[0] and A[size-1] won't move
    while( i < size ) {
        cycleBegin = i;
        t = a.memory[i];
        do {
            // Input matrix [r x c]
            // Output matrix
            // i_new = (i*r)%size
            next = (i*r)%size;
            std::swap(a.memory[next], t);
            b[i] = 1;
            i = next;
        } while( i != cycleBegin );
    
        // Get Next Move (what about querying random location?)
        for(i = 1; i < size && b[i]; i++)
            ;
    }

    int tmp = a.shape[0];
    a.shape[0] = a.shape[1];
    a.shape[1] = tmp;
}

float *alloc_ram(int n_elems)
{
    return (float*)malloc(sizeof(float) * n_elems);
}

void free_ram(float *ptr)
{
    free(ptr);
}
