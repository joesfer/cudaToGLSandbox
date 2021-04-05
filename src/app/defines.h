#pragma once

#include <cstdlib>

#ifdef __CUDACC__

static void *unifiedMalloc(size_t size) {
  void *ret;
  cudaMallocManaged(&ret, size);

  return ret;
}

#define HOST_DEVICE __host__ __device__
#define ALLOCATE(SIZE) unifiedMalloc(SIZE)
#define FREE(PTR) cudaFree(PTR)

#else

#define HOST_DEVICE
#define ALLOCATE(SIZE) malloc(SIZE)
#define FREE(PTR) free(PTR)

#endif
