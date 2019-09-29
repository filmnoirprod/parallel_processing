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

   value_t sum = 0;
   int row = blockIdx.y * blockDim.y +threadIdx.y;
   int column = blockIdx.x * blockDim.x + threadIdx.x;

   if(column < N && row < M) {
	for(int i=0 ; i<K ; i++) 
		sum += A[(row*K)+i]*B[(i*N)+column];

        C[(row*N)+column] = sum ;
   }
}

/*
 *  Coalesced memory acceses
 */

__global__ void dmm_gpu_coalesced(const value_t *A, const value_t *B, value_t *C, 
				 const size_t M, const size_t N, const size_t K) {

   value_t sum = 0;
   int row = blockIdx.y * blockDim.y +threadIdx.y;
   int column = blockIdx.x * blockDim.x + threadIdx.x;

   if(column < N && row < M) {
        for(int i=0 ; i<K ; i++) {
                sum += A[(i*M)+row]*B[(i*N)+column];
        }
        C[(row*N)+column] = sum ;
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
   
   extern __shared__ value_t localA[];
   //__shared__ value_t localA[1024];

   int bx,by,tx,ty,row,column;
   int Tile;
   value_t sum = 0 ;

   Tile = blockDim.x ;
   bx = blockIdx.x , by = blockIdx.y ;
   tx = threadIdx.x , ty = threadIdx.y ;
   row = by*Tile + ty , column = bx*Tile + tx ;
   localA[ty*Tile+tx] = 0.0 , localA[Tile*Tile+ty*Tile+tx] = 0.0 ;   

   for (int i = 0 ; i<(((K-1)/(Tile))+1) ; ++i ) {

	if( row < M && (i*Tile+tx) < K ) 
	   localA[ty*Tile+tx] = A[(i*Tile+tx)+row*K];
	else localA[ty*Tile+tx] = 0.0 ;

	if( column < N && (i*Tile+ty) < K )
	   localA[Tile*Tile+ty*Tile+tx] = B[(i*Tile+ty)*N+column];
	else localA[Tile*Tile+ty*Tile+tx] = 0.0 ;

	__syncthreads();

	for(int j = 0 ; j < Tile ; ++j) 
	   sum +=localA[ty*Tile+j]*localA[Tile*Tile+j*Tile+tx];

	__syncthreads();
   }
   
   if(row < M && column < N)
	C[row*N+column] = sum ;
	
}

/*
 Cublas matrix multiplication
 CublasSqemm performs C=aop(A)op(B) + bC
*/

__global__ void dmm_gpu_cublas(const value_t *A, const value_t *B, value_t *C,
                              const size_t M, const size_t N, const size_t K) {
  /*
   * Dummy function just to declare cublas .
   */
}


/*
 Cublas matrix multiplication
 CublasSqemm performs C=aop(A)op(B) + bC
*/
/*
__global__ void dmm_gpu_cublas(cublasHandle_t &handle, const value_t *A, const value_t *B,value_t *C,
			       const size_t M, const size_t N, const size_t K) {
  int m,n,k,lda,ldb,ldc;
  const float alf = 1.0 , bet = 0.0 ;
  const float *alpha = &alf , *beta = &bet ; 
  m = M ; n = N ; k = K ; 
  lda = m , ldb = k , ldc = m ;

  cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}
*/
