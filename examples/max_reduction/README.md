# Max Reduction Benchmark with ComPPare Framework <!-- omit from toc -->
Finds the maximum value within an array/vector.

- [1. Build Instructions](#1-build-instructions)
  - [1.1. Dependencies](#11-dependencies)
- [2. Run the Benchmark](#2-run-the-benchmark)
- [3. Implementation of ComPPare in this example](#3-implementation-of-comppare-in-this-example)
  - [3.1. Basic Usage of `HOTLOOP` macro](#31-basic-usage-of-hotloop-macro)
  - [3.2. Manual Timing](#32-manual-timing)
  - [3.3. Use of Pointer with `std::span`](#33-use-of-pointer-with-stdspan)
  - [3.4. ComPPare in `main()`](#34-comppare-in-main)


## 1. Build Instructions

### 1.1. Dependencies

* CMake
* C++20 compatible compiler
* For GPU kernel build:
  * NVIDIA nvcc compiler wrapper

---

### 1.1.2. GPU Build <!-- omit from toc -->

```bash
mkdir build && cd build

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    ..

make
```

#### 1.1.2.1. Optional GPU build option <!-- omit from toc -->
```bash
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    -DCUDA_ARCH=90 \ #Optional: Specify CUDA Architecture 
    ..
```


### 1.1.3. CPU only Build (default) <!-- omit from toc -->

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

## 2. Run the Benchmark

The executable takes the following command-line arguments:

### 1.2.1. Arguments <!-- omit from toc -->

| Flag      | Description                        | Default         |
| --------- | ---------------------------------- | --------------- |
| `--size`      | log2 of vector size (`N = 2^n`)    | 18              |


### 1.2.2. Example <!-- omit from toc -->

```bash
./max_reduction --size 26
```


### 1.2.3. Output Example <!-- omit from toc -->

Output will print:

* Function execution time
* Core compute time (isolated)
* Overhead (e.g., memory copies)
* Error of Max Val against reference

---

```bash
=== Max Reduction Benchmark Parameters ===
Vector size (N)     : 67108864
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


## 3. Implementation of ComPPare in this example

### 3.1. Basic Usage of `HOTLOOP` macro
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

### 3.2. Manual Timing

#### Reduction on GPUs <!-- omit from toc -->
In GPU, operations are executed in units of thread-blocks. Each block contains a number of threads, and a grid (a number of) of blocks are launched. 

In reduction, each block is reponsible for a portion of the whole array. After each block reduces, it returns the local maximum. As there are multiple blocks, it forms an output array of local maxima, or a partially reduced output. Therefore, we take this partial reduced output and do reduction again until there is only 1 value, which is our global maximum.

Below is a graphical explanation of reduction on GPU:
<style>
  .diagram { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; line-height: 1.35; }
  .h      { font-weight: 800; letter-spacing: .02em; }
  .note   { color: #444; }
  .swapptr{ color: #6a1b9a; font-weight: 700; }
  .b0 { color:#e53935; }  /* * */
  .b1 { color:#222222; }  /* • */
  .b2 { color:#1e88e5; }  /* o */
  .b3 { color:#43a047; }  /* x */
  .b4 { color:#fb8c00; }  /* + */
  .dim { color:#777; }
  .hr  { margin: 8px 0; border-bottom: 1px dashed #bbb; }
  .pad { margin: 6px 0; }
</style>

<div class="diagram">
  <div class="pad"><span class="h">Legend:</span>
    [B0] <span class="b0">*</span> &nbsp; [B1] <span class="b1">•</span> &nbsp; [B2] <span class="b2">o</span> &nbsp; [B3] <span class="b3">x</span> &nbsp; [B4] <span class="b4">+</span> &nbsp; &nbsp; [B#] denotes a thread block.
  </div>

  <div class="hr"></div>

  <div class="pad h">— ONE REDUCTION ITERATION —</div>

  <pre>
<strong>INPUT</strong>  (<code>d_input</code>, N elements; grouped by blocks of size <code>BLOCKSIZE</code>)
[B0]  <span class="b0">*</span> <span class="b0">*</span> <span class="b0">*</span> <span class="b0">*</span> <span class="b0">*</span> <span class="b0">*</span> <span class="b0">*</span> <span class="b0">*</span>  | [B1]  <span class="b1">•</span> <span class="b1">•</span> <span class="b1">•</span> <span class="b1">•</span> <span class="b1">•</span> <span class="b1">•</span> <span class="b1">•</span> <span class="b1">•</span> |  [B2]  <span class="b2">o</span> <span class="b2">o</span> <span class="b2">o</span> <span class="b2">o</span> <span class="b2">o</span> <span class="b2">o</span> <span class="b2">o</span> <span class="b2">o</span> |  [B3]  <span class="b3">x</span> <span class="b3">x</span> <span class="b3">x</span> <span class="b3">x</span> <span class="b3">x</span> <span class="b3">x</span> <span class="b3">x</span> <span class="b3">x</span> |   …

      │                        │                        │                        │
      v                        v                        v                        v

<strong>OUTPUT</strong>  (<code>d_output</code>, one per block = N/BLOCKSIZE)
      <span class="b0">*</span>                        <span class="b1">•</span>                        <span class="b2">o</span>                        <span class="b3">x</span>                   …
  </pre>

  <div class="note">Each block reduces its chunk → <strong>one block-maximum</strong> goes to <code>d_output</code>.</div>

  <div class="pad"><span class="swapptr">SWAP POINTERS:</span> <code>d_input</code> ↔ <code>d_output</code> &nbsp; <span class="dim">Now the block maxima are the next iteration’s input, so the partial result can be further reduced.</span></div>

  <div class="hr"></div>

  <div class="pad h">— NEXT ITERATION —</div>

  <pre>
<strong>INPUT</strong>  (<code>d_input</code>, N/BLOCKSIZE; previous iteration's partially reduced output)
[G0]  <span class="b0">*</span> <span class="b1">•</span> <span class="b2">o</span> <span class="b3">x</span> <span class="b4">+</span> <span class="b0">*</span> <span class="b1">•</span> <span class="b2">o</span> | [G1]  <span class="b3">x</span> <span class="b4">+</span> <span class="b0">*</span> <span class="b1">•</span> <span class="b2">o</span> <span class="b3">x</span> <span class="b4">+</span> <span class="b0">*</span> |   …

      │                       │                   │
      v                       v                   v

<strong>OUTPUT</strong>  (<code>d_output</code>, one per block = ⌈N/BLOCKSIZE²⌉)
      <span class="b0">*</span>                       <span class="b3">x</span>                   …
  </pre>

  <div class="pad"><strong>Repeat until only one value remains -- global maximum</strong>.</div>
  <br><br>
</div>

#### Benchmarking GPU Reduction <!-- omit from toc -->

As shown, input and output arrays are used interchangably in the reduction operation for read and writes. Hence, the original array is no longer avaliable in device memory. Therefore, we would need to copy the original array back into device memory before each iteration. This is memory transfer overhead we would avoid benchmarking, as we want to know the performance of the kernel itself.


To limit to certain regions to benchmark, there are macros provided `START_MANUAL_TIMER` and `STOP_MANUAL_TIMER` to specify the actual region of interest, and ignoring any overhead within the loop.

As for GPU code, macros `GPU_START_MANUAL_TIMER` and `GPU_STOP_MANUAL_TIMER` are used instead to time the device but not host cpu.

Hence, our GPU host code looks like:
```cpp
// max_gpu.cu
template <void (*KERNEL)(const int, float *__restrict__, const float *__restrict__)>
void gpu_max(std::span<const float> in, float &out)
{
    ...

    GPU_HOTLOOPSTART;

    /* Mem copy of Original Array to Device -- Required in each loop */

    GPU_START_MANUAL_TIMER;
    /* Parallel Reduction on GPU */
    GPU_STOP_MANUAL_TIMER;

    GPU_HOTLOOPEND;

    ...
}
```

### 3.3. Use of Pointer with `std::span`
In this code, the data is essentially a pointer to a raw array. However, the framework would not be able to compare raw arrays as the size is unknown. How can we solve this?

#### `std::span` <!-- omit from toc -->
Since C++20, `std::span` is avaliable which acts as a wrapper over raw pointers.

It stores the pointer to the array and the size of the array. Therefore, size would be known and the framework will be able to automatically compare the arrays.


**Essentially, std::span can be defined minimally:**
```cpp
template <typename T>
struct span
{
    T* pointer_to_array;
    std::size_t size_of_array;
}
```
**How to use std::span:**
```cpp
std::size_t array_size = 10;
int* array = new int[array_size];
std::span<int> wrapped_array = std::span<int>(array, array_size);
```


### 3.4. ComPPare in `main()`

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