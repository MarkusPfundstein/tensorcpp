#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <type_traits>
#include <utility>
#include <string>
#include <string.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include "tensor_gpu.h"
#include "tensor_cpu.h"
#include "tensor.h"

Tensor::Tensor()
    : shape(), nelems(0), memory(nullptr), is_on_gpu(false)
{
    //printf("default\n");
}

Tensor::Tensor(std::vector<int> shape, bool _on_gpu)
    : shape(shape), nelems(0), memory(nullptr), is_on_gpu(_on_gpu)
{
    int elems = 1;
    for (auto it = shape.cbegin(); it != shape.cend(); ++it) {
        elems *= *it;
    }
    //printf("alloc %d elems\n", elems);
    nelems = elems;
    if (_on_gpu) {
        memory = alloc_gpu(nelems);
    } else {
        memory = alloc_ram(nelems);
    }
}

Tensor::Tensor(const Tensor &other)
    : shape(other.shape),
      nelems(other.nelems),
      memory(other.memory),
      is_on_gpu(other.is_on_gpu)
{
    //printf("copy\n");
    memory = alloc_ram(nelems);
    memcpy(memory, other.memory, nelems * sizeof(float));
}

Tensor::Tensor(Tensor &&other) noexcept
    : shape(std::move(other.shape)),
      nelems(std::exchange(other.nelems, 0)),
      memory(std::exchange(other.memory, nullptr)),
      is_on_gpu(other.is_on_gpu)
{
    //printf("move\n");
}

Tensor& Tensor::operator=(const Tensor& other)
{
    printf("operator=&\n");
    nelems = other.nelems;
    is_on_gpu = other.is_on_gpu;

    if (is_on_gpu) {
        memory = alloc_gpu(nelems);
        throw std::runtime_error("gpu copy not yet implemented");
    } else {
        memory = alloc_ram(nelems);
        memcpy(memory, other.memory, nelems * sizeof(float));
    }
    
    shape = other.shape;
    
    return *this;
}
    
Tensor& Tensor::operator=(Tensor&& other)
{
    //printf("operator=&&\n");
    memory = std::exchange(other.memory, nullptr);
    nelems = std::exchange(other.nelems, 0);
    shape = std::move(other.shape);
    is_on_gpu = std::exchange(other.is_on_gpu, false);
    return *this;
}

Tensor::~Tensor()
{
    if (memory) {
        if (is_on_gpu) {
            free_gpu(memory);
        } else {
            free_ram(memory);
        }
        memory = nullptr;
    }
}

unsigned int Tensor::dimensions() const
{
    return shape.size();
}

int Tensor::calc_mem_idx(const std::vector<int> &indices) noexcept
{
    int mem_idx = 0;

    for (std::vector<int>::size_type i = 0; i < indices.size() - 1; ++i) {
        const int idx = indices[i];

        int mul = 1;
        for (std::vector<int>::size_type k = i + 1; k < shape.size(); ++k) {
            mul *= shape[k];
        }
        mem_idx += idx * mul;
    }

    mem_idx += indices[indices.size() - 1];
    //printf("  mem_idx: %d\n", mem_idx);
    return mem_idx;
}

Tensor Tensor::operator+(const Tensor &other)
{
    return add(other);
}

Tensor Tensor::add(const Tensor &other)
{
    if (other.shape.size() != shape.size()) {
        throw std::invalid_argument("invalid shapes");
    }

    if (is_on_gpu) {
        return std::forward<Tensor>(gpu_tensor_add(*this, other));
    } else {
        return std::forward<Tensor>(cpu_tensor_add(*this, other));
    }
}

Tensor Tensor::mul(float scalar)
{
    if (is_on_gpu) {
        throw std::runtime_error("mul not yet implemented on gpu");
    } else {
        return std::forward<Tensor>(cpu_tensor_mul(*this, scalar));
    }
}

Tensor Tensor::mul(const Tensor &b)
{
    if (is_on_gpu) {
        return std::forward<Tensor>(gpu_tensor_mul_2d(*this, b));
    } else {
        return std::forward<Tensor>(cpu_tensor_mul_2d(*this, b));
    }
}


Tensor Tensor::operator*(float scalar)
{
    return std::forward<Tensor>(mul(scalar));
}

Tensor Tensor::operator*(const Tensor &b)
{
    return std::forward<Tensor>(mul(b));
}

void Tensor::set(const std::vector<int> &indices, float val)
{
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("invalid indices shape");
    }
    if (is_on_gpu)
    {
        throw std::runtime_error("Error. Tensor is on GPU. _memory cant be accessed");
    }
    const int mem_idx = calc_mem_idx(indices);
    memory[mem_idx] = val;
}

float Tensor::get(const std::vector<int> &indices)
{
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("invalid indices shape");
    }
    if (is_on_gpu) {
        throw std::runtime_error("Error. Tensor is on GPU. _memory cant be accessed");
    }
    const int mem_idx = calc_mem_idx(indices);
    return memory[mem_idx];
}

void Tensor::move_to_gpu()
{
    if (is_on_gpu) {
        throw std::runtime_error("Tensor already on gpu");
    }

    float *tmp = memory;

    memory = alloc_gpu(nelems);

    cudaError_t err = cudaMemcpy(memory, tmp, sizeof(float) * nelems, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("error copying Tensor to device: ") + cudaGetErrorString(err));
    }

    free(tmp);
    is_on_gpu = true;
}

void Tensor::move_to_ram()
{
    if (!is_on_gpu) {
        throw std::runtime_error("Tensor not on gpu");
    }

    float *tmp = memory;

    memory = alloc_ram(nelems);

    cudaError_t err = cudaMemcpy(memory, tmp, sizeof(float) * nelems, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("error copying Tensor from device: ") + cudaGetErrorString(err));
    }

    free_gpu(tmp);

    is_on_gpu = false;
}

float dot(const Tensor &a, const Tensor &b)
{
    if (a.is_on_gpu != b.is_on_gpu) {
        throw std::runtime_error("Both tensors need to be on gpu or not");
    }

    if (a.is_on_gpu) {
        return 0.0;
    } else {
        return cpu_dot(a, b);
    }
}