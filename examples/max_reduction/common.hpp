#pragma once

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef __CUDACC__

// Block size for CUDA kernel launches
#ifndef BLOCKSIZE
#define BLOCKSIZE 1024
#endif

#endif