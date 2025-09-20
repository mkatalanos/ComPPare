# ComPPare

**Header‑only benchmarking and validation for C++20 code.**

ComPPare lets you time and cross‑check any number of host‑side implementations—CPU, OpenMP, CUDA, SYCL, TBB, etc.—that share the same inputs and outputs. It is intended for developers who are porting functions into new framework or hardware, allowing for standalone development and testing.

---

## Purpose

* **Performance comparison**: Measure the total call time, an inner region‑of‑interest (ROI) that you define, and the residual overhead (setup, transfers).
* **Validate results**: Report maximum, mean and total absolute error versus a designated reference implementation and flag discrepancies.
* **Streamline porting**: Run the same data set through multiple versions of a function.

---

## Key capabilities

| Capability              | Description                                                       |
| ----------------------- | ----------------------------------------------------------------- |
| Header‑only             | Copy headers, or include it, and compile with C++20 or newer. |
| Any host backend | Accepts any function pointer that runs on the host.               |
| Detailed timing         | Separates overall call time, your ROI time, and setup/transfer overhead.                   |
| Built-in error comparison | For common data types, automatically choose the correct method and compares against reference function    |

## Getting Started
See [User Guide](docs/user_guide.md) for more detailed user guide and [Examples](examples/README.md) to see real life examples.

Contributions are welcome! Please see [Code Documentation](https://funglf.github.io/ComPPare/) if interested in contributing to this repo.

## Install
### 1. Clone repository
```bash
git clone https://github.com/funglf/ComPPare.git --recursive
```
#### If submodules like google benchmark/ nvbench is not needed:
```bash
git clone https://github.com/funglf/ComPPare.git
```

### 2. (Optional) Build Google Benchmark and nvbench
See [Google Benchmark Instructions](https://github.com/google/benchmark/blob/b20cea674170b2ba45da0dfaf03953cdea473d0d/README.md) 

See [nvbench Intructions](https://github.com/NVIDIA/nvbench/blob/b88a45f4170af4e907e69af22a55af67859d3b49/README.md)

### 3. Include ComPPare
In your C++ code, simply include the comppare header file by:
```c
#include <comppare/comppare.hpp>
```

---

## Quick Start

### 1. Adopt the required function signature
*Function output must be `void`*
and consists of the input, then output types
```cpp
void impl(const Inputs&... in,     // read‑only inputs
        Outputs&...      out);     // outputs compared to reference
```

In order to benchmark specific regions of code, following Macros `HOTLOOPSTART`, `HOTLOOPEND` are needed:
```cpp
void impl(const Inputs&... in,
        Outputs&...      out);
{
    /* 
    setup or overhead you DO NOT want to benchmark 
    -- memory allocation, data transfer, etc.
    */

    HOTLOOPSTART; // Macro of start of Benchmarking Region of Interest

    // ... perform core computation here ...

    HOTLOOPEND;   // Macro of end of Benchmarking Region of Interest
}
```


#### SAXPY function example signatures
```cpp
void saxpy_cpu(/*Input types*/
               float a,
               const std::vector<float> &x,
               const std::vector<float> &y_in,
               /*Output types*/
               std::vector<float> &y_out)

// Comparing with another function with the exact same signature
void saxpy_gpu(/*Input types*/
               float a,
               const std::vector<float> &x,
               const std::vector<float> &y_in,
               /*Output types*/
               std::vector<float> &y_out)
```


### 2. Create a comparison object

1. **Describe the output types** as template argument
2. **Pass the input data** — constructs framework object with input data that will be reused for every implementation


```cpp
auto Cmp = comppare::make_comppare</*Output Types*/std::vector<float>>(a, x, y); // a: float, x: input vector x, y: input vector y
```
> Note: you can use move semantics here. All inputs are perfectly forwarded. eg. `Cmp cmp(a, std::move(x), std::move(y));`


### 3. Register implementations

```cpp
cmp.set_reference("saxpy reference", saxpy_cpu);  // setting reference
cmp.add("saxpy gpu",  saxpy_gpu);           // any number of additional functions
```

### 4. Run and inspect

```cpp
cmp.run(argc, argv);
```

#### Sample report:

```bash
*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
============ ComPPare Framework ============
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

Number of implementations:             4
Warmup iterations:                   100
Benchmark iterations:                100
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

Implementation              ROI µs/Iter            Func µs            Ovhd µs         Max|err|[0]        Mean|err|[0]       Total|err|[0]
saxpy reference                     0.28               33.67                5.63            0.00e+00            0.00e+00            0.00e+00                                   
saxpy gpu                          10.89           137828.11           136739.02            5.75e+06            2.85e+06            2.92e+09            <-- FAIL         
```
### Complete example with SAXPY
[(See SAXPY Full Example)](examples/saxpy/README.md)

```cpp
#include <vector>

#include <comppare/comppare.hpp>

// Serial reference
void saxpy_cpu(/*Input pack*/
               float a,
               const std::vector<float> &x,
               const std::vector<float> &y_in,
               /*Output pack*/
               std::vector<float> &y_out)
{
    size_t N = x.size();
    y_out.resize(N);

    HOTLOOPSTART;
    for (size_t i = 0; i < N; ++i)
    {
        y_out[i] = a * x[i] + y_in[i];
    }
    HOTLOOPEND;
}

// OpenMP variant
void saxpy_gpu(/* same signature */){...};


int main(int argc, char **argv)
{
    float a = 1.1f;
    std::vector<float> x(1000, 2.2); // Vector of size 1000 filled with 2.2
    std::vector<float> y(1000, 3.3); // Vector of size 1000 filled with 3.3

    /*
    Create Instance of the comparison framework with input data
    a -- float
    x -- std::vector<float>
    y -- std::vector<float>
    */
    auto Cmp = comppare::make_comppare</*Output Types*/std::vector<float>>(a, x, y);

    // Set reference implementation
    cmp.set_reference("saxpy reference", /*Function*/ saxpy_cpu);
    // Add implementations to compare
    cmp.add("saxpy gpu", /*Function*/ saxpy_gpu);

    // Run the comparison with specified iterations and tolerance
    cmp.run(argc, argv);

    return 0;
}
```

---




