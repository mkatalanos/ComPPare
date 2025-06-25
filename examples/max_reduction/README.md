# Max Reduction Benchmark with ComPPare Framework

## Max

Finds the maximum value within an array/vector.

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
    -DCUDA=ON \
    -DCUDA_ARCH=70 \
    ..

make
````


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
    -DCUDA=OFF \
    ..

make
```

Only CPU backends (serial and parallel) will be built.

---

## Run the Benchmark

The executable takes the following command-line arguments:

### Arguments

| Flag      | Description                        | Default         |
| --------- | ---------------------------------- | --------------- |
| `--size`      | log2 of vector size (`N = 2^n`)    | 26              |
| `--iter`   | number of benchmark iterations     | 10              |
| `--tol`    | numerical tolerance for validation | 1e-6            |
| `--warmup/--no-warmup` | to do warmup run or not | warmup |

### Example

```bash
./max_reduction --size 28 --iter 20 --tol 1e-7
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
=== Max Reduction Benchmark Parameters ===
Vector size (N)     : 268435456
Iterations          : 20
Error tolerance     : 1e-07
===================================

Name                        Func µs           ROI µs          Ovhd µs       Max|err|[0]      (MaxErr-idx)      Mean|err|[0]     Total|err|[0]
cpu serial               15708841.11       15708840.00              1.11          0.00e+00               —          0.00e+00          0.00e+00
cpu omp                    250068.62         250067.00              1.62          0.00e+00               —          0.00e+00          0.00e+00
cpu thread                 937883.36         937882.00              1.36          0.00e+00               —          0.00e+00          0.00e+00
gpu kernel                1830507.95          66976.77        1763531.18          0.00e+00               —          0.00e+00          0.00e+00
gpu kernel opt            1792535.98          33857.54        1758678.45          0.00e+00               —          0.00e+00          0.00e+00
```

