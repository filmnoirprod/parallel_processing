/*
 *  dmm_main.cu -- DMM front-end program.
 *
 *  Copyright (C) 2019, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2019, Athena Elafrou
 */

#include "alloc.h"
#include "dmm.h"
#include "error.h"
#include "gpu_util.h"
#include "mat_util.h"
#include "timer.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

#ifndef VALUES_MAX
#define VALUES_MAX MAKE_VALUE_CONSTANT(1.)
#endif

#ifndef EPS
#define EPS MAKE_VALUE_CONSTANT(1.e-6)
#endif

#ifndef NR_ITER
#define NR_ITER 100
#endif


static void check_result(const value_t *const *test, const value_t *const *orig,
                         const size_t M, const size_t N) {
  printf("Checking ... ");
  bool ret = mat_equals(test, orig, M, N, EPS);
  if (ret) {
    printf("PASSED!\n");
  } else {
    printf("FAILED!\n");
  }
}

static void report_results(xtimer_t *timer, const size_t M, const size_t N,
                           const size_t K) {
  double elapsed_time = timer_elapsed_time(timer);
  size_t flops = 2 * M * N * K * NR_ITER;

  printf(" Elapsed time: %lf s", elapsed_time);
  printf(" Performance:  %lf Gflop/s\n", flops * 1.e-9 / elapsed_time);
}

static void print_usage() {
  printf("Usage: [GPU_KERNEL=<kernel_no>] [GPU_BLOCK_SIZE=<size>] "
         "%s <M> <N> <K>\n",
         program_name);
  printf("GPU_KERNEL defaults to 0\n");
  printf("GPU_BLOCK_SIZE defaults to 256\n");
  printf("Available kernels [id:name]:\n");
  size_t i;
  for (i = 0; i < GPU_KERNEL_END; ++i) {
    printf("\t%zd:%s\n", i, gpu_kernels[i].name);
  }
}

int main(int argc, char **argv) {
  set_program_name(argv[0]);
  if (argc < 4) {
    warning(0, "too few arguments");
    print_usage();
    exit(EXIT_FAILURE);
  }

  size_t M = atoi(argv[1]);
  if (!M)
    error(0, "invalid argument: %s", argv[1]);
  size_t N = atoi(argv[2]);
  if (!N)
    error(0, "invalid argument: %s", argv[2]);
  size_t K = atoi(argv[3]);
  if (!K)
    error(0, "invalid argument: %s", argv[3]);

  /* Read block size and kernel to launch from the environment */
  const char *env_gpu_kernel = getenv("GPU_KERNEL");
  const char *env_gpu_block_sizex = getenv("GPU_BLOCK_SIZEX");
  const char *env_gpu_block_sizey = getenv("GPU_BLOCK_SIZEY");
  int kernel = (env_gpu_kernel) ? atoi(env_gpu_kernel) : GPU_NAIVE;
  int block_sizex = (env_gpu_block_sizex) ? atoi(env_gpu_block_sizex) : 32;
  int block_sizey = (env_gpu_block_sizey) ? atoi(env_gpu_block_sizey) : 32;
  size_t orig_M = M;
  size_t orig_N = N;
  size_t orig_K = K;

  if((block_sizey*block_sizex) > 1024) block_sizey = 1024/block_sizex;

  int grid_sizex = ceil((N*1.0)/block_sizex); // FILLME: compute the grid size
  int grid_sizey = ceil((M*1.0)/block_sizey);
  size_t shmem_size = 0; 

  // create the cublas handle
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0;
  float beta = 0.0;
  cublasStatus_t t;

  /*
   *  FILLME: you can optionally adjust appropriately (increase
   *          only) the matrix size here if that helps you with your
   *          kernel code, e.g., to avoid divergent warps.
   */


  printf("M: %zd", orig_M);
 // printf("Adjusted dimension M: %zd\n", M);
  printf(" N: %zd", orig_N);
 // printf("Adjusted dimension N: %zd\n", N);
  printf(" K: %zd", orig_K);
 // printf("Adjusted dimension K: %zd\n", K);

  /*
   * Allocate the structures.
   *
   * Initialization to zero is crucial if you adjusted the matrix
   * size.
   */

  value_t **A = (value_t **)calloc_2d(M, K, sizeof(**A));
  if (!A)
    error(1, "alloc_2d failed");

  value_t **A_t = (value_t **)calloc_2d(K, M, sizeof(**A_t));
  if(!A_t)
    error(1,"alloc_2d_failed");

  value_t **B = (value_t **)calloc_2d(K, N, sizeof(**B));
  if (!B)
    error(1, "alloc_2d failed");

  value_t **C = (value_t **)calloc_2d(M, N, sizeof(**C));
  if (!C)
    error(1, "alloc_2d failed");

#ifdef _CHECK_
  value_t **C_serial = (value_t **)calloc_2d(M, N, sizeof(**C_serial));
  if (!C_serial)
    error(1, "alloc_2d failed");
#endif

  /* Initialize */
  srand48(0);
  mat_init_rand(A, orig_M, orig_K, VALUES_MAX);
  mat_init_rand(B, orig_K, orig_N, VALUES_MAX);

  /* Setup timers */
  xtimer_t timer;

  /*
   *  FILLME: Set up the blocks, grid and shared memory depending on
   *          the kernel. Make any transformations to the input
   *          matrix here.
   */

  /*
	In order to achieve memory coalesce we will use
	the transposed array of A.
	The above is meant only for kernels 1,2,3 !
	For kernel 0 (NAIVE) we will use the initial array A.
	We will deep copy array A into A_t if needed in order 
	to maintain initial A for results verification .

  */

  /* GPU allocate A */
  value_t *gpu_A = (value_t *)gpu_alloc(M * K * sizeof(*gpu_A));
  if (!gpu_A)
    error(0, "gpu_alloc failed: %s", gpu_get_last_errmsg());

  if (kernel == 1 ) {
	//Coalesced

	//Deep copy of array A into A_t (also transpose it)
	for(int i=0 ; i<K; i++)
	   for(int j=0; j<M; j++)
	      A_t[i][j] = A[j][i];

  	// Copy A to GPU
  	if (copy_to_gpu(A_t[0], gpu_A, M * K * sizeof(*gpu_A)) < 0)
    		error(0, "copy_to_gpu failed: %s", gpu_get_last_errmsg());

  }

  else {
	//Naive or CUBLAS or tiled

  	// Copy A to GPU
  	if (copy_to_gpu(A[0], gpu_A, M * K * sizeof(*gpu_A)) < 0)
    		error(0, "copy_to_gpu failed: %s", gpu_get_last_errmsg());

  }

  if( kernel == 2 ) {

	// tiled matrix multiplication 
	
	shmem_size = 2*block_sizey*block_sizex*sizeof(value_t);
  }


  dim3 gpu_block(block_sizex,block_sizey);  // FILLME: set up the block dimensions
  dim3 gpu_grid(grid_sizex,grid_sizey);   // FILLME: set up the grid dimensions

  //printf(">>>> Begin of record <<<<\n");
  printf(" Block size: %dx%d", gpu_block.x, gpu_block.y);
  printf(" Grid size : %dx%d", gpu_grid.x, gpu_grid.y);
  printf(" Shared memory size: %ld bytes", shmem_size);

  /* GPU allocations */
  value_t *gpu_B = (value_t *)gpu_alloc(K * N * sizeof(*gpu_B));
  if (!gpu_B)
    error(0, "gpu_alloc failed: %s", gpu_get_last_errmsg());

  value_t *gpu_C = (value_t *)gpu_alloc(M * N * sizeof(*gpu_C));
  if (!gpu_C)
    error(0, "gpu_alloc failed: %s", gpu_get_last_errmsg());

  /* Copy data to GPU */
  if (copy_to_gpu(B[0], gpu_B, K * N * sizeof(*gpu_B)) < 0)
    error(0, "copy_to_gpu failed: %s", gpu_get_last_errmsg());

  /* Reset C and copy it to GPU */
  mat_init(C, M, N, MAKE_VALUE_CONSTANT(0.0));
  if (copy_to_gpu(C[0], gpu_C, M * N * sizeof(*gpu_C)) < 0)
    error(0, "copy_to_gpu failed: %s", gpu_get_last_errmsg());

  if (kernel >= GPU_KERNEL_END)
    error(0, "the requested kernel does not exist");

  //printf("GPU kernel version: %s\n", gpu_kernels[kernel].name);

  /* Execute and time the kernel */
  timer_clear(&timer);
  timer_start(&timer);
  for (size_t i = 0; i < NR_ITER; ++i) {
    if(kernel==3) 
	//t = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,M,N,K,&alpha,gpu_A,M,gpu_B,K,&beta,gpu_C,M);
        t = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,gpu_B,N,gpu_A,K,&beta,gpu_C,N);
    else
	gpu_kernels[kernel].fn<<<gpu_grid, gpu_block, shmem_size>>>(gpu_A, gpu_B,
                                                                   gpu_C, M, N, K);
#ifdef _DEBUG_
    cudaError_t err;
    if ((err = cudaGetLastError()) != cudaSuccess)
      error(0, "gpu kernel failed to launch: %s", gpu_get_errmsg(err));
    if(kernel==3 && t!= CUBLAS_STATUS_SUCCESS) {
           if (t == CUBLAS_STATUS_INVALID_VALUE)
   		printf("CUBLAS invalid params\n");
   	   else if (t == CUBLAS_STATUS_EXECUTION_FAILED)
   		printf("CUBLAS execution failed\n");
    }
#endif
    cudaThreadSynchronize();
  }
  timer_stop(&timer);

  /* Copy result back to host and check */
  if (copy_from_gpu(C[0], gpu_C, M * N * sizeof(*gpu_C)) < 0)
    error(0, "copy_from_gpu failed: %s", gpu_get_last_errmsg());

#ifdef _CHECK_
  /* Compute serial */
  dmm_serial(A, B, C_serial, orig_M, orig_N, orig_K);
  check_result(C, C_serial, orig_M, orig_N);
#endif

  report_results(&timer, orig_M, orig_N, orig_K);
  //printf(">>>> End of record <<<<\n");

  /* Free resources on host */
  free_2d((void **)A);
  free_2d((void **)A_t);
  free_2d((void **)B);
  free_2d((void **)C);
#ifdef _CHECK_
  free_2d((void **)C_serial);
#endif

  /* Free resources on GPU */
  gpu_free(gpu_A);
  gpu_free(gpu_B);
  gpu_free(gpu_C);

  // Destroy the cublas handle
  cublasDestroy(handle);

  return EXIT_SUCCESS;
}
