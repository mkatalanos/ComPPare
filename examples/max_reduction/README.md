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
    -DUSE_CUDA=ON \
    ..

make
````

#### Optional GPU build option
```bash
cmake \
    -DCMAKE_BUILD_TYPE=Release \
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
    -DUSE_CUDA=OFF \
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


### Example

```bash
./max_reduction --size 18
```


### Output Example

Output will print:

* Function execution time
* Core compute time (isolated)
* Overhead (e.g., memory copies)
* Error of Max Val against reference

---

```bash
=== Max Reduction Benchmark Parameters ===
Vector size (N)     : 262144
===================================

*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
============ ComPPare Framework ============
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

Number of implementations:             5
Warmup iterations:                   100
Benchmark iterations:                100
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

Implementation                  Func µs             ROI µs            Ovhd µs       Total|err|[0]
cpu serial                      36303.95            17437.95            18866.00            0.00e+00
cpu omp                        107208.48            62084.54            45123.94            0.00e+00
cpu thread                      44459.22            21151.77            23307.45            0.00e+00
gpu kernel                     133597.43            32933.28           100664.15            0.00e+00
gpu kernel opt                 109846.87            20369.54            89477.33            0.00e+00
```


## Implementation of ComPPare in this example

### Basic Usage of `HOTLOOP` macro
Adding `HOTLOOPSTART` and `HOTLOOPEND` macros to the region you want to benchmark.
It will run the region with certain iterations of warmup (default=100) before running another certain number of iterations for benchmark (default=100).

```c
// max_cpu.cpp
void cpu_max_serial(std::span<const float> in, float &out)
{
    HOTLOOPSTART;   // Start of Region you want to benchmark
    out = *std::max_element(in.data(), in.data() + in.size());
    HOTLOOPEND;     // End of Region you want to benchmark
}
```

### More advanced usage -- Manual Timing
In some cases, there are certain regions we want to NOT benchmark, for instance memory transfer overhead etc. 
Macros like `GPU_START_MANUAL_TIMER` and `GPU_STOP_MANUAL_TIMER` allows users to only target certain region within the whole loop.


```c
// max_gpu.cu
template <void (*KERNEL)(const int, float *__restrict__, const float *__restrict__)>
void gpu_max(std::span<const float> in, float &out)
{
    ...

    GPU_HOTLOOPSTART;

    /* Region you want to ignore but in loop -- say memory transfer */

    GPU_START_MANUAL_TIMER;
    /* Region you want to time */
    GPU_STOP_MANUAL_TIMER;

    /* Region you want to ignore but in loop */

    GPU_HOTLOOPEND;

    ...
}
```

### ComPPare in `main()`

First create the the instance of comppare by providing the input type(s) in InputContext (`std::span<const float>` in this case) and the output type(s) in OutputContext (`float` in this case).
Then initialize the instance with input data. 
```c
comppare::
        InputContext<std::span<const float>>::
            OutputContext<float>
                compare(/*type: std::span<const float>*/input_data);
```

Set Reference function and add other functions for comparison
```c
    // Set reference implementation
    compare.set_reference("cpu serial", cpu_max_serial);
    // Add implementations to compare
    compare.add("cpu omp", cpu_max_omp);
```