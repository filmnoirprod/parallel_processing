/*
 *  dmm_gpu.cu -- Template for DMM GPU kernels
 *
 *  Copyright (C) 2019, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2019, Athena Elafrou
 */

#include "dmm.h"
#include <stdio.h>
#include <cublas_v2.h>

/*
 *  Utility function to get the thread ID within the
 *  global working space.
 */
__device__ int get_global_tid() {
  return (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * blockDim.y +
         blockDim.x * threadIdx.y + threadIdx.x;
}

/*
 *  Utility function to get the thread ID within the
 *  local/block working space.
 */
__device__ int get_local_tid() {
  return blockDim.x * threadIdx.y + threadIdx.x;
}

/*
 *  Naive kernel
 */
__global__ void dmm_gpu_naive(const value_t *A, const value_t *B, value_t *C,
                              const size_t M, const size_t N, const size_t K) {
  /*
   * FILLME: fill the code for the naive kernel.
   */

   value_t sum = 0;
   int row = blockIdx.y * blockDim.y +threadIdx.y;
   int column = blockIdx.x * blockDim.x + threadIdx.x;

   if(column < K && row < M) {
	for(int i=0 ; i<N ; i++) {
		sum += A[(row*N)+i]*B[(i*K)+column];
	}
        C[(row*K)+column] = sum ;
   }
}

/*
 *  Coalesced memory acceses
 */
__global__ void dmm_gpu_coalesced(const value_t *A, const value_t *B,
                                  value_t *C, const size_t M, const size_t N,
                                  const size_t K) {
  /*
   * FILLME: fill the code for the coalesced kernel.
   */

   value_t sum = 0;
   int row = blockIdx.y * blockDim.y +threadIdx.y;
   int column = blockIdx.x * blockDim.x + threadIdx.x;

   if(column < K && row < M) {
        for(int i=0 ; i<N ; i++) {
                sum += A[(i*N)+row]*B[(i*K)+column];
        }
        C[(row*K)+column] = sum ;
   }


}

/*
 *  Use of shared memory
 */
__global__ void dmm_gpu_shmem(const value_t *A, const value_t *B, value_t *C,
                              const size_t M, const size_t N, const size_t K) {
  /*
   * FILLME: fill the code for the shared memory kernel.
   */
}

/*
 Cublas matrix multiplication
 CublasSqemm performs C=aop(A)op(B) + bC
*/

__global__ void dmm_gpu_cublas(cublasHandle_t &handle, const value_t *A, const value_t *B,value_t *C,
			       const size_t M, const size_t N, const size_t K) {
  int m,n,k,lda,ldb,ldc;
  const float alf = 1.0 , bet = 0.0 ;
  const float *alpha = &alf , *beta = &bet ; 
  m = M ; n = N ; k = K ; 
  lda = m , ldb = k , ldc = m ;

  cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}
