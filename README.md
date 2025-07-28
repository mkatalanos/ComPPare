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

## Install
### 1. Clone repository
```bash
git clone git@github.com:funglf/ComPPare.git --recursive
```
#### If submodules like google benchmark/ nvbench is not needed:
```bash
git clone git@github.com:funglf/ComPPare.git
```

### 2. (Optional) Build Google Benchmark
See [Google Benchmark Instructions](benchmark/README.md) 

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
void cf(/*Input types*/
            float a,
            const std::vector<float> &x,
            const std::vector<float> &y_in,
            /*Output types*/
            std::vector<float> &y_out)
```


### 2. Create a comparison object

1. **Describe the types** — list the *input* types first, then the *output* types:

```cpp
using Cmp = 
    comppare::
        /*Define Input Pack Types same as the function*/
        InputContext<
            float,  /*float a*/
            std::vector<float>, /*std::vector<float> x*/
            std::vector<float>  /*std::vector<float> y*/
            >::
                /*Define Output Pack types same as the function*/
                OutputContext<
                std::vector<float>  /*std::vector<float> y_out*/
                >;
```

2. **Pass the input data** — constructs framework object with input data that will be reused for every implementation:

```cpp
Cmp cmp(a, x, y);   // a: float, x: input vector x, y: input vector y
```
> Note: you can use move semantics here. All inputs are perfectly forwarded. eg. `Cmp cmp(a, std::move(x), std::move(y));`


### 3. Register implementations

```cpp
cmp.set_reference("cpu serial", saxpy_cpu);  // setting reference
cmp.add("gpu kernel",  saxpy_gpu);           // any number of additional functions
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

Number of implementations:             3
Warmup iterations:                   100
Benchmark iterations:                100
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

Implementation                  Func µs             ROI µs            Ovhd µs         Max|err|[0]        Mean|err|[0]       Total|err|[0]
cpu serial                         38.01               15.97               22.05            0.00e+00            0.00e+00            0.00e+00
gpu kernel                      73151.41                1.07            73150.34            3.30e+06            1.66e+06            1.70e+09  <-- FAIL
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

    using Cmp = 
    comppare::
        /*Define Input Pack Types same as the function*/
            InputContext<float, 
                std::vector<float>, 
                std::vector<float>>::
                    /*Define Output Pack types same as the function*/
                    OutputContext<std::vector<float>>;
    /*
    Create Instance of the comparison framework with input data
    a -- float
    x -- std::vector<float>
    y -- std::vector<float>
    */
    Cmp cmp(a, x, y); // cmp is the instance

    // Set reference implementation
    cmp.set_reference("cpu serial", /*Function*/ saxpy_cpu);
    // Add implementations to compare
    cmp.add("gpu kernel", /*Function*/ saxpy_gpu);

    // Run the comparison with specified iterations and tolerance
    cmp.run(argc, argv);

    return 0;
}
```

---




