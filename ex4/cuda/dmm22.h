/*
 *  dmm.h -- Declarations and definitions related to the DMM
 *           multiplication kernels.
 *
 *  Copyright (C) 2019, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2019, Athena Elafrou
 */

#ifndef DMM_H
#define DMM_H

#include "common.h"
#include <stdbool.h>
#include <stddef.h>
#include <cublas_v2.h>

__BEGIN_C_DECLS

void dmm_serial(const value_t *const *A, const value_t *const *B, value_t **C,
                const size_t M, const size_t N, const size_t K);

__END_C_DECLS

#ifdef __CUDACC__
#define __MAKE_KERNEL_NAME(name) dmm_gpu##name
#define MAKE_KERNEL_NAME(name) __MAKE_KERNEL_NAME(name)
#define MAKE_KERNEL2_NAME(name) __MAKE_KERNEL_NAME(name)

#define DECLARE_GPU_KERNEL(name)                                               \
  __global__ void MAKE_KERNEL_NAME(name)(const value_t *A, const value_t *B,   \
                                         value_t *C, const size_t M,           \
                                         const size_t N, const size_t K)

#define DECLARE_GPU2_KERNEL(name)                                                                      \
 __global__ void MAKE_KERNEL2_NAME(name)(cublasHandle_t &handle , const value_t *A, const value_t *B , \
					value_t *C , const size_t M, const size_t N, const size_t K)  
#define SHMEM_PER_BLOCK 8 * 1024

typedef void (*dmm_kernel_t)(const value_t *A, const value_t *B, value_t *C,
                             const size_t M, const size_t N, const size_t K);

typedef struct {
  const char *name;
  dmm_kernel_t fn;
} gpu_kernel_t;

enum { GPU_NAIVE = 0, GPU_COALESCED, GPU_SHMEM, GPU_CUBLAS, GPU_KERNEL_END };

DECLARE_GPU_KERNEL(_naive);
DECLARE_GPU_KERNEL(_coalesced);
DECLARE_GPU_KERNEL(_shmem);
DECLARE_GPU2_KERNEL(_cublas);

static gpu_kernel_t gpu_kernels[] = {
    {
        .name = "naive", .fn = MAKE_KERNEL_NAME(_naive),
    },

    {
        .name = "coalesced", .fn = MAKE_KERNEL_NAME(_coalesced),
    },

    {
        .name = "shmem", .fn = MAKE_KERNEL_NAME(_shmem),
    },

    {
	.name = "cublas", .fn =MAKE_KERNEL2_NAME(_cublas),
    },
};

#endif /* __CUDACC__ */
#endif /* DMM_H */
