#include <stdio.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
#include "tensor_gpu.h"

float *alloc_gpu(int nelems)
{
    float *memory = nullptr;
    cudaError_t err = cudaMalloc((void**)&memory, sizeof(float) * nelems);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("error allocating memory on device: ") + cudaGetErrorString(err));
    }
    return memory;
}

void free_gpu(float *mem)
{
    cudaFree(mem);
}

#if 0
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
    if (err == cudaSuccess)
        return;
    fprintf(stderr, "%s returned %s at %s : %d", statement, cudaGetErrorString(err), file, line);

}
#define SAFE_CALL(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#endif


__global__ void mul_mat2d_mat2d_kernel(float *m1, float *m2, float *r, int m1w, int m2w, int rw, int rh)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < rh) && (col < rw)) {
        // dot product
        float accum = 0.0f;
        for (int c = 0; c < m1w; c++)
        {
            float v1 = m1[row * m1w + c];
            float v2 = m2[c * m2w + col];
            accum += v1 *  v2;
        }

        r[row * rw + col] = accum;
    }
}


Tensor gpu_tensor_mul_mat2d_mat2d(const Tensor &a, const Tensor &b)
{
    if (!a.is_on_gpu || !b.is_on_gpu) {
        throw std::runtime_error("One of tensors not on gpu");
    }

    Tensor out({a.shape[0], b.shape[1]}, true);

    int aw = out.shape[1];
    int ah = out.shape[0];

    static const int blockWidth = 16;
    static const int blockHeight = blockWidth;
    int numBlocksW = aw / blockWidth;
    int numBlocksH = ah / blockHeight;
    if (aw % blockWidth) {
        numBlocksW++;
    }
    if (ah % blockHeight) {
        numBlocksH++;
    }

    dim3 dimGrid(numBlocksW, numBlocksH);
    dim3 dimBlock(blockWidth, blockHeight);

    mul_mat2d_mat2d_kernel<<<dimGrid, dimBlock>>>(
        a.memory,
        b.memory,
        out.memory,
        a.shape[1],		// aw
        b.shape[1],		// bw
        out.shape[1],	// out w
        out.shape[0]	// out h
    );	

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("error matrix2d_mul_kernel: ") + cudaGetErrorString(err));
    }

    return out;
}

__global__ void mul_mat2d_vec_kernel(float *mat2d, float *vec, float *r, int matw, int vecw, int rw)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if ((row < 1) && (col < rw)) {
        float accum = 0.0f;
        for (int c = 0; c < vecw; c++)
        {
            float v1 = mat2d[c * matw + col];
            float v2 = vec[c];
            accum += v1 *  v2;
        }

        r[col] = accum;
    }
}

Tensor gpu_tensor_mul_mat2d_vec(const Tensor &mat, const Tensor &vec)
{
    if (mat.shape[0] != vec.shape[0]) {
        std::stringstream ss;
        ss << "cpu_tensor_mul_mat2d_mat2d: invalid shapes ";
        ss << "[" << mat.shape[0] << "," << mat.shape[1] << "] and [" << vec.shape[0] << "]";
        throw std::runtime_error(ss.str());
    }

    Tensor out({mat.shape[1]}, true);

    int ah = out.shape[0];

    static const int blockWidth = 16;
    static const int blockHeight = blockWidth;
    int numBlocksH = ah / blockHeight;
    if (ah % blockHeight) {
        numBlocksH++;
    }

    dim3 dimGrid(1, numBlocksH);
    dim3 dimBlock(blockWidth, blockHeight);

    mul_mat2d_vec_kernel<<<dimGrid, dimBlock>>>(
        mat.memory,
        vec.memory,
        out.memory,
        mat.shape[1],       // aw
        vec.shape[0],       // bw
        out.shape[0]        // out w
    );	

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("error matrix2d_mul_kernel: ") + cudaGetErrorString(err));
    }

    return out;
}

__global__ void tensor_add(float *a, float *b, float *out, int nelems)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < nelems) {
        out[id] = a[id] + b[id];
    }
}

Tensor gpu_tensor_add(const Tensor& a, const Tensor &b)
{
    if (!a.is_on_gpu || !b.is_on_gpu) {
        throw std::runtime_error("One of tensors not on gpu");
    }

    Tensor out(a.shape, true);

    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(a.nelems) / thr_per_blk );

    tensor_add<<<blk_in_grid, thr_per_blk>>>(a.memory, b.memory, out.memory, a.nelems);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("error tensor_add: ") + cudaGetErrorString(err));
    }

    return out;
}

__global__ void tensor_mul(float *a, float scalar, float *out, int nelems)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < nelems) {
        out[id] = a[id] * scalar;
    }
}

Tensor gpu_tensor_mul(const Tensor &a, float scalar)
{
    if (!a.is_on_gpu) {
        throw std::runtime_error("Tensor not on gpu");
    }

    Tensor out(a.shape, true);

    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(a.nelems) / thr_per_blk );

    tensor_mul<<<blk_in_grid, thr_per_blk>>>(a.memory, scalar, out.memory, a.nelems);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("error tensor_add: ") + cudaGetErrorString(err));
    }

    return out;
}

__global__ void tensor_dot(float *a, float *b, float *c, int nelems)
{
    __shared__ float cache[256];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    cache[threadIdx.x] = a[index] * b[index];

    __syncthreads();

    if (threadIdx.x == 0)
    {
        float sum = 0;
        for (int i = 0; i < nelems; i++)
        {
            sum += cache[i];
        }
        atomicAdd(c, sum);
    }

    cache[threadIdx.x] = 0.0;
}

float gpu_dot(const Tensor &a, const Tensor &b)
{
    cudaError_t err;
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(a.nelems) / thr_per_blk );

    float *out;
    
    err = cudaMalloc((void**)&out, 1 * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc: ") + cudaGetErrorString(err));
    }

    float zero = 0;
    cudaMemcpy(out, &zero, sizeof(float), cudaMemcpyHostToDevice);

    tensor_dot<<<blk_in_grid, thr_per_blk>>>(a.memory, b.memory, out, a.nelems);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(out);
        throw std::runtime_error(std::string("error tensor_dot: ") + cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    float result;
    cudaMemcpy(&result, out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(out);

    return result;
}

__global__ void tensor_pow(float *a, float power, float *out, int nelems)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < nelems) {
        out[id] = pow(a[id], power);
    }
}

Tensor gpu_tensor_pow(const Tensor &a, float power)
{
    if (!a.is_on_gpu) {
        throw std::runtime_error("Tensor not on gpu");
    }

    Tensor out(a.shape, true);

    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(a.nelems) / thr_per_blk );

    tensor_pow<<<blk_in_grid, thr_per_blk>>>(a.memory, power, out.memory, a.nelems);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("error tensor_add: ") + cudaGetErrorString(err));
    }

    return out;
}

__global__ void tensor_pointwise_mul(float *a, float *b, float *out, int nelems)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < nelems) {
        out[id] = a[id] * b[id];
    }
}

Tensor gpu_pointwise_mul(const Tensor& a, const Tensor &b)
{
    if (!a.is_on_gpu || !b.is_on_gpu) {
        throw std::runtime_error("One of tensors not on gpu");
    }

    Tensor out(a.shape, true);

    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(a.nelems) / thr_per_blk );

    tensor_pointwise_mul<<<blk_in_grid, thr_per_blk>>>(a.memory, b.memory, out.memory, a.nelems);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("error tensor_pointwise_mul: ") + cudaGetErrorString(err));
    }

    return out;
}

__global__ void tensor_tanh(float *a, float *out, int nelems)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < nelems) {
        double expx = exp(a[id]);
        double expminx = exp(-a[id]);
        out[id] = (expx - expminx) / (expx + expminx);
    }
}

Tensor gpu_tensor_tanh(const Tensor &a)
{
    if (!a.is_on_gpu) {
        throw std::runtime_error("Tensor not on gpu");
    }

    Tensor out(a.shape, true);

    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(a.nelems) / thr_per_blk );

    tensor_tanh<<<blk_in_grid, thr_per_blk>>>(a.memory, out.memory, a.nelems);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("error gpu_tensor_tanh: ") + cudaGetErrorString(err));
    }

    return out;
}

__global__ void tensor_relu(float *a, float *out, int nelems)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < nelems) {
        out[id] = a[id] > 0.0 ? a[id] : 0.0;
    }
}

Tensor gpu_tensor_relu(const Tensor &a)
{
    if (!a.is_on_gpu) {
        throw std::runtime_error("Tensor not on gpu");
    }

    Tensor out(a.shape, true);

    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(a.nelems) / thr_per_blk );

    tensor_relu<<<blk_in_grid, thr_per_blk>>>(a.memory, out.memory, a.nelems);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("error tensor_relu: ") + cudaGetErrorString(err));
    }

    return out;
}

Tensor gpu_tensor_sin(const Tensor &a)
{
    throw std::runtime_error("gpu_tensor_sin not implemented");
}

Tensor gpu_tensor_cos(const Tensor &a)
{
    throw std::runtime_error("gpu_tensor_cos not implemented");
}

void gpu_reset()
{
    cudaDeviceReset();
}