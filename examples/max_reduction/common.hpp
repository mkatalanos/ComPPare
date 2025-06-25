#pragma once

#ifdef DOUBLE_PRECISION
#define float_t double
#else
#define float_t float
#endif

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef __CUDACC__
// CUDA device code for max reduction of either float (fp32) or double (fp64)
#ifdef DOUBLE_PRECISION
#define float_t_max fmax
#else
#define float_t_max fmaxf
#endif

// Block size for CUDA kernel launches
#ifndef BLOCKSIZE
#define BLOCKSIZE 1024
#endif

#endif