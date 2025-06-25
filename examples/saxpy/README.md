# SAXPY Benchmark with ComPPare Framework

## SAXPY

**SAXPY** stands for **Single-Precision A·X Plus Y**.  
It computes the vector operation:

$$
y_{\text{out}}[i] = a \cdot x[i] + y[i]
$$

where:
- `x`, `y` are input vectors (length N)
- `a` is a scalar constant
- `y_out` is the result vector

---

## Build Instructions

### Dependencies

* CMake
* C++20 compatible compiler
* For GPU kernel build:
  * NVIDIA nvcc compiler wrapper
* For C++ stdpar on GPU builds:
  * NVIDIA HPC SDK with `nvc++`

---

### GPU Build (with nvc++, stdpar on gpu, CUDA kernel)

```bash
mkdir build && cd build

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=nvc++ \
    -DCMAKE_CUDA_HOST_COMPILER=nvc++ \
    -DCUDA=ON \
    -DCUDA_ARCH=70 \
    ..

make
````

### GPU Build (with g++, CUDA kernel)

```bash
mkdir build && cd build

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_HOST_COMPILER=g++ \
    -DCUDA=ON \
    -DCUDA_ARCH=70 \
    ..

make
````

### CPU Build (with nvc++, stdpar on GPU)
```bash
mkdir build && cd build

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=nvc++ \
    -DCUDA=OFF \
    ..

make
````

Notes:

* Requires `nvc++` compiler from NVIDIA HPC SDK for stdpar to run on GPU

### CPU only Build (default)

```bash
mkdir build && cd build
cmake ..
make
```

Or explicitly:

```bash
mkdir build && cd build

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCUDA=OFF \
    ..

make
```

Only CPU backends (scalar and parallel) will be built.

---

## Run the Benchmark

The executable takes the following command-line arguments:

### Arguments

| Flag      | Description                        | Default         |
| --------- | ---------------------------------- | --------------- |
| `--size`      | log2 of vector size (`N = 2^n`)    | 26              |
| `--scalar` | scalar multiplier `a`              | random \[-5, 5] |
| `--iter`   | number of benchmark iterations     | 10              |
| `--tol`    | numerical tolerance for validation | 1e-6            |

### Example

```bash
./saxpy --size 27 --scalar 10.98 --iter 100 --tol 1e-8
```


### Output Example

Output will print:

* Function execution time
* Core compute time (isolated)
* Overhead (e.g., memory copies)
* Validation metrics: max, mean, total error
* if max error > tolerance --> index of the max error (where)

---

```bash
=== SAXPY Benchmark Parameters ===
Vector size (N)     : 134217728
Scalar a            : 10.98
Iterations          : 100
Error tolerance     : 1e-08
===================================

Name                        Func µs          Core µs          Ovhd µs       Max|err|[0]      (MaxErr-idx)      Mean|err|[0]     Total|err|[0]
cpu serial                 134034.89         132343.70           1691.18          0.00e+00               —          0.00e+00          0.00e+00
cpu parallel                29604.00          27894.86           1709.14          0.00e+00               —          0.00e+00          0.00e+00
gpu std par                  4489.56           2774.86           1714.70          0.00e+00               —          0.00e+00          0.00e+00
gpu kernel                   6943.10           1936.11           5006.99          1.10e+13           5254268          2.74e+12          3.68e+20  <-- FAIL
```

> Note: The gpu kernel example in saxpy_gpu.cu deliberately contains a bug that causes the validation to fail (<-- FAIL). You’ll need to debug and correct the GPU implementation in `saxpy_gpu.cu`