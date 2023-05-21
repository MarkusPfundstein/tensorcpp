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
#include "tensor.h"

template <typename T>
T *alloc_gpu(int nelems)
{
    T *memory = nullptr;
    cudaError_t err = cudaMalloc((void**)&memory, sizeof(T) * nelems);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("error allocating memory on device: ") + cudaGetErrorString(err));
    }
    return memory;
}

template <typename T>
void free_gpu(T *mem)
{
    cudaFree(mem);
}

Tensor::Tensor()
    : _shape(), _nelems(0), _memory(nullptr), _on_gpu(false)
{
    //printf("default\n");
}

Tensor::Tensor(std::vector<int> shape, bool _on_gpu)
    : _shape(shape), _nelems(0), _memory(nullptr), _on_gpu(_on_gpu)
{
    int elems = 1;
    for (auto it = _shape.cbegin(); it != _shape.cend(); ++it) {
        elems *= *it;
    }
    //printf("alloc %d elems\n", elems);
    _nelems = elems;
    if (_on_gpu) {
        _memory = alloc_gpu<float>(_nelems);
    } else {
        _memory = alloc_ram<float>(_nelems);
    }
}

Tensor::Tensor(const Tensor &other)
    : _shape(other._shape),
      _nelems(other._nelems),
      _memory(other._memory),
      _on_gpu(other._on_gpu)
{
    //printf("copy\n");
    _memory = alloc_ram<float>(_nelems);
    memcpy(_memory, other._memory, _nelems * sizeof(float));
}

Tensor::Tensor(Tensor &&other) noexcept
    : _shape(std::move(other._shape)),
      _nelems(std::exchange(other._nelems, 0)),
      _memory(std::exchange(other._memory, nullptr)),
      _on_gpu(other._on_gpu)
{
    //printf("move\n");
}

Tensor& Tensor::operator=(const Tensor& other)
{
    //printf("operator=&\n");
    _nelems = other._nelems;
    _on_gpu = other._on_gpu;

    if (_on_gpu) {
        _memory = alloc_gpu<float>(_nelems);
        throw std::runtime_error("gpu copy not yet implemented");
    } else {
        _memory = alloc_ram<float>(_nelems);
        memcpy(_memory, other._memory, _nelems * sizeof(float));
    }
    
    _shape = other._shape;
    
    return *this;
}
    
Tensor& Tensor::operator=(Tensor&& other)
{
    //printf("operator=&&\n");
    _memory = std::exchange(other._memory, nullptr);
    _nelems = std::exchange(other._nelems, 0);
    _shape = std::move(other._shape);
    _on_gpu = std::exchange(other._on_gpu, false);
    return *this;
}

Tensor::~Tensor()
{
    if (_memory) {
        if (_on_gpu) {
            free_gpu(_memory);
        } else {
            free_ram(_memory);
        }
        _memory = nullptr;
    }
}

int Tensor::calc_mem_idx(const std::vector<int> &indices) noexcept
{
    int mem_idx = 0;

    for (std::vector<int>::size_type i = 0; i < indices.size() - 1; ++i) {
        const int idx = indices[i];

        int mul = 1;
        for (std::vector<int>::size_type k = i + 1; k < _shape.size(); ++k) {
            mul *= _shape[k];
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
    if (other._shape.size() != _shape.size()) {
        throw std::invalid_argument("invalid shapes");
    }

    if (_on_gpu) {
        return gpu_tensor_add(*this, other);
    } else {
        Tensor t(_shape);
        for (unsigned long int i = 0; i < _nelems; ++i) {
            t._memory[i] = _memory[i] + other._memory[i];
        }
        return t;
    }
}

void Tensor::set(const std::vector<int> &indices, float val)
{
    if (indices.size() != _shape.size()) {
        throw std::invalid_argument("invalid indices shape");
    }
    if (_on_gpu)
    {
        throw std::runtime_error("Error. Tensor is on GPU. _memory cant be accessed");
    }
    const int mem_idx = calc_mem_idx(indices);
    _memory[mem_idx] = val;
}

float Tensor::get(const std::vector<int> &indices)
{
    if (indices.size() != _shape.size()) {
        throw std::invalid_argument("invalid indices shape");
    }
    if (_on_gpu) {
        throw std::runtime_error("Error. Tensor is on GPU. _memory cant be accessed");
    }
    const int mem_idx = calc_mem_idx(indices);
    return _memory[mem_idx];
}

void Tensor::move_to_gpu()
{
    if (_on_gpu) {
        throw std::runtime_error("Tensor already on gpu");
    }

    float *tmp = _memory;

    _memory = alloc_gpu<float>(_nelems);

    cudaError_t err = cudaMemcpy(_memory, tmp, sizeof(float) * _nelems, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("error copying Tensor to device: ") + cudaGetErrorString(err));
    }

    free(tmp);
    _on_gpu = true;
}

void Tensor::move_to_ram()
{
    if (!_on_gpu) {
        throw std::runtime_error("Tensor not on gpu");
    }

    float *tmp = _memory;

    _memory = alloc_ram<float>(_nelems);

    cudaError_t err = cudaMemcpy(_memory, tmp, sizeof(float) * _nelems, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("error copying Tensor from device: ") + cudaGetErrorString(err));
    }

    free_gpu(tmp);

    _on_gpu = false;
}
