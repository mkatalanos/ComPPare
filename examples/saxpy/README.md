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

---

### GPU Build


```bash
mkdir build && cd build

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_HOST_COMPILER=g++ \
    -DUSE_CUDA=ON \
    ..

make
```

#### Optional GPU build option
```bash
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_HOST_COMPILER=g++ \
    -DUSE_CUDA=ON \
    -DCUDA_ARCH=90 \ #Optional: Specify CUDA Architecture 
    ..
```

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
    -DUSE_CUDA=OFF \
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
| `--size`      | log2 of vector size (`N = 2^n`)    | 10              |
| `--scalar` | scalar multiplier `a`              | random \[-5, 5] |

### Example

```bash
./saxpy --size 27 --scalar 10.98 --iter 100
```


### Output Example

Output will print:

* Function execution time
* Core compute time (isolated)
* Overhead (e.g., memory copies)
* Correctness metrics: max, mean, total error

---

```bash
=== SAXPY Benchmark Parameters ===
Vector size (N)     : 1024
Scalar a            : 4.30614
===================================

*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
============ ComPPare Framework ============
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

Number of implementations:             3
Warmup iterations:                   100
Benchmark iterations:                100
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

Implementation                  Func µs             ROI µs            Ovhd µs         Max|err|[0]        Mean|err|[0]       Total|err|[0]
cpu serial                         38.01               15.97               22.05            0.00e+00            0.00e+00            0.00e+00
cpu parallel                    48098.13            17477.30            30620.83            0.00e+00            0.00e+00            0.00e+00
gpu kernel                      73151.41                1.07            73150.34            3.30e+06            1.66e+06            1.70e+09  <-- FAIL
```

> Note: The gpu kernel example in saxpy_gpu.cu deliberately contains a bug that causes the validation to fail (<-- FAIL). You’ll need to debug and correct the GPU implementation in `saxpy_gpu.cu`