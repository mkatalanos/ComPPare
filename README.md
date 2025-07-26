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
| Built-in error checks | Reports maximum, mean, and total absolute errors.    |


---

## Quick Start

### 1. Adopt the required function signature

```cpp
void impl(const Inputs&... in,     // read‑only inputs
        Outputs&...      out);     // outputs compared to reference
{
    // setup (memory allocation, data transfer, etc.)

    HOTLOOPSTART;
    // ... perform core computation here ...
    HOTLOOPEND;

}
```


#### SAXPY function example signatures
```cpp
void saxpy_cpu(/*Input pack*/
            float a,
            const std::vector<float> &x,
            const std::vector<float> &y_in,
            /*Output pack*/
            std::vector<float> &y_out)

// Comparing with another function with the exact same signature
void saxpy_gpu(/*Input pack*/
            float a,
            const std::vector<float> &x,
            const std::vector<float> &y_in,
            /*Output pack*/
            std::vector<float> &y_out)
```


### 2. Create a comparison object

1. **Describe the types** — list the *input* types first, then the *output* types:

```cpp
using Cmp = 
    comppare::
        /*Define Input Pack Types same as the function*/
        InputContext<
            float, 
            std::vector<float>, 
            std::vector<float>
            >::
                /*Define Output Pack types same as the function*/
                OutputContext<
                std::vector<float>
                >;
```

2. **Pass the input data** — constructs framework object with input data that will be reused for every implementation:

```cpp
Cmp cmp(a, x, y);   // a: float, x: input vector x, y: input vector y
```
> Note: you can use move semantics here. All inputs are perfectly forwarded. eg. `Cmp cmp(a, std::move(x), std::move(y));`


### 3. Register implementations

```cpp
cmp.set_reference("CPU serial", cpu_std);  // setting reference
cmp.add("CPU OpenMP",  cpu_omp);           // any number of additional functions
```

### 4. Run and inspect

```cpp
cmp.run(argc, argv, tol);
```

#### Sample report:

```
Name           Func µs   ROI µs   Ovhd µs  Max|err|[0]
CPU serial     1.34e+5   …        …        0.00e+00
CPU OpenMP     2.96e+4   …        …        0.00e+00 
```

### Complete example with SAXPY
[(See SAXPY Full Example)](examples/saxpy/README.md)

```cpp
#include <vector>
#include <omp.h>

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
void saxpy_cpu_omp(/* same signature */){...};


int main(int argc, char **argv)
{
    // Initialize SAXPY configuration
    SaxpyConfig cfg = init_saxpy(argc, argv);

    // Define the input and output types for the comparison framework instance
    using ScalarType = float;
    using VectorType = std::vector<float>;

    using cmp = 
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
    cmp compare(cfg.a, cfg.x, cfg.y);

    // Set reference implementation
    compare.set_reference("cpu serial", /*Function*/ saxpy_cpu);
    // Add implementations to compare
    compare.add("cpu parallel", /*Function*/ saxpy_cpu_omp);

    // Run the comparison with specified iterations and tolerance
    compare.run(argc, argv, cfg.tol);

    return 0;
}
```

---




