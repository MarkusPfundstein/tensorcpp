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
#include <sstream>

Tensor::Tensor()
    : shape(), nelems(0), memory(nullptr), is_on_gpu(false), gradient(nullptr)
{
}

Tensor::Tensor(std::vector<int> shape, bool _on_gpu)
    : shape(shape), nelems(0), memory(nullptr), is_on_gpu(_on_gpu), gradient(nullptr)
{
    int elems = 1;
    for (auto it = shape.cbegin(); it != shape.cend(); ++it) {
        elems *= *it;
    }
    //printf("tensor shape cons\n");
    nelems = elems;
    if (_on_gpu) {
        memory = alloc_gpu(nelems);
    } else {
        memory = alloc_ram(nelems);
    }
}

Tensor::Tensor(std::vector<int> shape, std::vector<float> data, bool _on_gpu)
    : Tensor(shape, false)
{
    //printf("tensor shape & data cons\n");
    set_data(data);

    if (_on_gpu) {
        move_to_gpu();
    }
}

Tensor::Tensor(const Tensor &other)
    : shape(other.shape),
      nelems(other.nelems),
      memory(other.memory),
      is_on_gpu(other.is_on_gpu),
      gradient(other.gradient)
{
    printf("!!!!!COPY Tensor!!!!!!\n");
    memory = alloc_ram(nelems);
    memcpy(memory, other.memory, nelems * sizeof(float));
}

Tensor::Tensor(Tensor &&other) noexcept
    : shape(std::move(other.shape)),
      nelems(std::exchange(other.nelems, 0)),
      memory(std::exchange(other.memory, nullptr)),
      is_on_gpu(other.is_on_gpu),
      gradient(std::exchange(other.gradient, nullptr))
{
}

Tensor& Tensor::operator=(const Tensor& other)
{
    printf("COPY Tensor::operator=&\n");
    throw std::runtime_error("NO TENSOR COPYING FOR NOW");
    nelems = other.nelems;
    is_on_gpu = other.is_on_gpu;
    gradient = other.gradient;

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
    //printf("MOVE OPErATOR\n");
    memory = std::exchange(other.memory, nullptr);
    nelems = std::exchange(other.nelems, 0);
    shape = std::move(other.shape);
    is_on_gpu = std::exchange(other.is_on_gpu, false);
    gradient = std::exchange(other.gradient, nullptr);
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


Tensor Tensor::operator+(const Tensor &other) const
{
    return add(other);
}

Tensor Tensor::add(const Tensor &other) const
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

Tensor Tensor::mul(float scalar) const
{
    if (is_on_gpu) {
        return std::forward<Tensor>(gpu_tensor_mul(*this, scalar));
    } else {
        return std::forward<Tensor>(cpu_tensor_mul(*this, scalar));
    }
}


Tensor Tensor::operator&(const Tensor &b) const
{
    if (shape.size() == 1 && b.shape.size() == 1 && shape[0] == b.shape[0]) {
        float res = Tensor::dot(*this, b);
        return std::move(Tensor({1}, {res}, is_on_gpu));
    }
    else if (shape.size() == 2 && b.shape.size() == 2) {
        if (is_on_gpu) {
            return std::forward<Tensor>(gpu_tensor_mul_mat2d_mat2d(*this, b));
        } else {
            return std::forward<Tensor>(cpu_tensor_mul_mat2d_mat2d(*this, b));
        }
    }
    // scalar * matrix or matrix * scalar
    // or vector * matrix or matrix * vector
    else if ((shape.size() == 1 && b.shape.size() >= 1) || (shape.size() >= 1 && b.shape.size() == 1)) {
        if (shape.size() == 1 && b.shape.size() == 2) {
            if (is_on_gpu) {
                return std::forward<Tensor>(gpu_tensor_mul_mat2d_vec(b, *this));
            } else {
                return std::forward<Tensor>(cpu_tensor_mul_mat2d_vec(b, *this));
            }
        }
    }
    printf("Unhandled shape sizes in Tensor::operator&. [%ld, %ld]\n", shape.size(), b.shape.size());
    throw std::runtime_error("Tensor::operator&. invalid matmul for given shapes");
}

Tensor Tensor::mul(const Tensor &b) const
{
    //printf("%ld, %ld\n", shape.size(), b.shape.size());
    // dot product or scalar mul

    
    // pointwise mul
    if (shape.size() == b.shape.size() && shape == b.shape) {
        if (is_on_gpu) {
            return std::forward<Tensor>(gpu_pointwise_mul(*this, b));
        } else {
            return std::forward<Tensor>(cpu_pointwise_mul(*this, b));
        }
    }
    // scalar * matrix or matrix * scalar
    else if ((shape.size() == 1 && b.shape.size() >= 1) || (shape.size() >= 1 && b.shape.size() == 1)) {
        // scalar * matrix
        if (shape[0] == 1) {
            float s = 0.0;
            if (is_on_gpu) {
                cudaMemcpy(&s, memory, sizeof(float), cudaMemcpyDeviceToHost);
            } else {
                s = memory[0];
            }
            return std::forward<Tensor>(b.mul(s));
        // matrix * scalar
        } else if (b.shape[0] == 1) {
            float s = 0.0;
            if (is_on_gpu) {
                cudaMemcpy(&s, b.memory, sizeof(float), cudaMemcpyDeviceToHost);
            } else {
                s = b.memory[0];
            }
            return std::forward<Tensor>(mul(s));
        }
    }
    printf("Unhandled shape sizes in Tensor::mul. [%ld, %ld]\n", shape.size(), b.shape.size());
    throw std::runtime_error("mul for called shapes not implemented");
}

Tensor Tensor::operator-() const
{
    return *this * (-1);
}

Tensor Tensor::operator-(const Tensor &other) const
{
    return *this + (-other);
}

Tensor Tensor::operator*(float scalar) const
{
    return std::forward<Tensor>(mul(scalar));
}

Tensor Tensor::operator*(const Tensor &b) const
{
    return std::forward<Tensor>(mul(b));
}

Tensor Tensor::operator/(float scalar)
{
    Tensor other({1}, {scalar}, is_on_gpu);

    return *this * other.pow(-1);
}
    
Tensor Tensor::operator/(const Tensor &other)
{
    return *this * other.pow(-1);
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

float Tensor::get(const std::vector<int> &indices) const
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

void Tensor::set_data(const std::vector<float> &data)
{
    if (data.size() != nelems) {
        throw std::runtime_error("Data has different size than sensor");
    }
    if (is_on_gpu) {
        throw std::runtime_error("Tensor is on GPU. Cant manipulate data");
    }

    memcpy(memory, data.data(), nelems * sizeof(float));
}

std::string Tensor::str() const
{
    std::ostringstream ss;

    ss << "Tensor(shape={";
    for (std::vector<int>::size_type i = 0; i < shape.size(); ++i) {
        ss << shape[i];
        if (i < shape.size() - 1) {
            ss << ",";
        }
    }
    ss << "}, mem=[";
    for (std::vector<int>::size_type i = 0; i < nelems; ++i) {
        if (is_on_gpu) {
            float tmp;
            cudaMemcpy(&tmp, &memory[i], sizeof(float), cudaMemcpyDeviceToHost);
            ss << tmp;
        } else {
            ss << memory[i];
        }
        if (i < nelems - 1) {
            ss << ",";
        }
    }
    ss << "], gpu=" << is_on_gpu << ")";

    return ss.str();
}

int Tensor::calc_mem_idx(const std::vector<int> &indices) const noexcept
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
    return mem_idx;
}

float Tensor::dot(const Tensor &a, const Tensor &b)
{
    if (a.is_on_gpu != b.is_on_gpu) {
        throw std::runtime_error("Both tensors need to be on gpu or not");
    }

    if (a.is_on_gpu) {
        return gpu_dot(a, b);
    } else {
        return cpu_dot(a, b);
    }
}

Tensor Tensor::pow(float power) const
{
    if (is_on_gpu) {
        return std::forward<Tensor>(gpu_tensor_pow(*this, power));
    } else {
        return std::forward<Tensor>(cpu_tensor_pow(*this, power));
    }
}

Tensor Tensor::pow(const Tensor& power) const
{
    if (is_on_gpu) {
        float tmp;
        cudaMemcpy(&tmp, &power.memory[0], sizeof(float), cudaMemcpyDeviceToHost);
        return pow(tmp);
    } else {
        return pow(power.memory[0]);
    }
}

Tensor Tensor::tanh() const
{
    if (is_on_gpu) {
        return std::forward<Tensor>(gpu_tensor_tanh(*this));
    } else {
        return std::forward<Tensor>(cpu_tensor_tanh(*this));
    }
}

Tensor Tensor::relu() const
{
    if (is_on_gpu) {
        return std::forward<Tensor>(gpu_tensor_relu(*this));
    } else {
        return std::forward<Tensor>(cpu_tensor_relu(*this));
    }
}

Tensor Tensor::sin() const
{
    if (is_on_gpu) {
        return std::forward<Tensor>(gpu_tensor_sin(*this));
    } else {
        return std::forward<Tensor>(cpu_tensor_sin(*this));
    }
}

Tensor Tensor::sin_backwards() const
{
    if (is_on_gpu) {
        return std::forward<Tensor>(gpu_tensor_cos(*this));
    } else {
        return std::forward<Tensor>(cpu_tensor_cos(*this));
    }
}

Tensor Tensor::cos() const
{
    if (is_on_gpu) {
        return std::forward<Tensor>(gpu_tensor_cos(*this));
    } else {
        return std::forward<Tensor>(cpu_tensor_cos(*this));
    }
}

Tensor Tensor::mul_backwards(const Tensor &other) const
{
    return other;
}

Tensor Tensor::add_backwards(const Tensor &other) const
{
    return Tensor({1}, {1}, other.is_on_gpu);
}