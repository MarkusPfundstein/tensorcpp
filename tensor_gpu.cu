#include <stdio.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include "tensor_gpu.h"

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
			accum += (v1 *  v2);
		}

		r[row * rw + col] = accum;
	}
}

void matrix_mul(float *m1, float *m2, float *r, int m1w, int m2w, int rw, int rh)
{
	static const int blockWidth = 16;
	static const int blockHeight = blockWidth;
	int numBlocksW = 1/ blockWidth;
	int numBlocksH = 1/ blockHeight;
	if (1 % blockWidth) numBlocksW++;
	if (1 % blockHeight) numBlocksH++;

	dim3 dimGrid(numBlocksW, numBlocksH);
	dim3 dimBlock(blockWidth, blockHeight);

	matrix2d_mul_kernel<<<dimGrid, dimBlock>>>(m1, m2, r, m1w, m2w, 1, 1);
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
	if (!a._on_gpu || !b._on_gpu) {
		throw std::runtime_error("One of tensors not on gpu");
	}

	Tensor out(a._shape, true);

    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(a._nelems) / thr_per_blk );
	printf("blks_in_grid: %d, thr_per_blk: %d\n", blk_in_grid, thr_per_blk);

	tensor_add<<<blk_in_grid, thr_per_blk>>>(a._memory, b._memory, out._memory, a._nelems);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("error tensor_add: ") + cudaGetErrorString(err));
	}

	return out;
}