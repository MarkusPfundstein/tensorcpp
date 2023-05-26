#include <stdio.h>
#include <cuda_runtime.h>
#include <stdexcept>
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


__global__ void matrix2d_mul_kernel(float *m1, float *m2, float *r, int m1w, int m2w, int rw, int rh)
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

Tensor gpu_tensor_mul_2d(const Tensor &a, const Tensor &b)
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

	matrix2d_mul_kernel<<<dimGrid, dimBlock>>>(
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