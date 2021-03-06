#include <stdio.h>
// Based on NVIDIA docs for reference

// Size of arrays
#define N 4096
#define THREADS_PER_BLOCK 512

// Kernel
__global__ void dot_prod(int *a, int *b, int *res)
{
	__shared__ int products[THREADS_PER_BLOCK];
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	products[threadIdx.x] = a[id]*b[id];
	__syncthreads();

	if(threadIdx.x == 0) {
		int sum = 0;
		for(int i=0; i<THREADS_PER_BLOCK; i++) {
			sum = sum + products[i];
		}
		atomicAdd(res, sum);
	}
}

// Main program
int main()
{
	// Number of bytes to allocate for N integers
	size_t bytes = N*sizeof(int);

	// Allocate memory for arrays A, B, and result on host
	int *A = (int*)malloc(bytes);
	int *B = (int*)malloc(bytes);
	int *result = (int*)malloc(sizeof(int));

	// Allocate memory for arrays d_A, d_B, and d_result on device
	int *d_A, *d_B, *d_result;
	cudaErrorCheck( cudaMalloc(&d_A, bytes) );
	cudaErrorCheck( cudaMalloc(&d_B, bytes) );
	cudaErrorCheck( cudaMalloc(&d_result, sizeof(int)) );

	// Fill host arrays A and B
  for(int i=0; i<N; i++)
  {
    A[i] = 1;
    B[i] = 2;
  }

  // Copy data from host arrays A and B to device arrays d_A and d_B
  cudaErrorCheck( cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) );
  cudaErrorCheck( cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice) );

  // Set execution configuration parameters
  //    thr_per_blk: number of CUDA threads per grid block
  //    blk_in_grid: number of blocks in grid
  int thr_per_blk = THREADS_PER_BLOCK;
  int blk_in_grid = ceil( float(N) / thr_per_blk );

	// Launch kernel
	dot_prod<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_result);

	  // Check for errors in kernel launch (e.g. invalid execution configuration paramters)
  cudaError_t cuErrSync  = cudaGetLastError();

  // Check for errors on the GPU after control is returned to CPU
  cudaError_t cuErrAsync = cudaDeviceSynchronize();

  if (cuErrSync != cudaSuccess) { printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErrSync)); exit(0); }
  if (cuErrAsync != cudaSuccess) { printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErrAsync)); exit(0); }

	// Copy result from device to host
	cudaErrorCheck( cudaMemcpy(result, d_result, sizeof(int), cudaMemcpyDeviceToHost) );

	// Verify results
	if(*result != 2*N) { printf("Error: result is %d instead of %d\n", *result, 2*N); exit(0); }

	// Free CPU memory
	free(A);
	free(B);

	// Free GPU memory
	cudaErrorCheck( cudaFree(d_A) );
	cudaErrorCheck( cudaFree(d_B) );
	cudaErrorCheck( cudaFree(d_result) );

  printf("\n---------------------------\n");
  printf("__SUCCESS__\n");
  printf("---------------------------\n");
  printf("N                 = %d\n", N);
  printf("Threads Per Block = %d\n", thr_per_blk);
  printf("Blocks In Grid    = %d\n", blk_in_grid);
  printf("---------------------------\n\n");
	
	return 0;
}
