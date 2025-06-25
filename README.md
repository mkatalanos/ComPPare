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
| Header‑only             | Copy `ComPPare.hpp`, include it, and compile with C++20 or newer. |
| Any host backend | Accepts any function pointer that runs on the host.               |
| Detailed timing         | Separates overall call time, your ROI time, and setup/transfer overhead.                   |
| Built-in error checks | Reports maximum, mean, and total absolute errors.    |


---

## How to use ComPPare

### 1. Adopt the required function signature

```cpp
void impl(const Inputs&... in,      // read‑only inputs
        Outputs&...      out,     // outputs compared to reference
        size_t           iters,   // loop iterations (benchmark repeats)
        double&          roi_us); // your measured region‑of‑interest in micro‑seconds
{
    // setup (memory allocation, data transfer, etc.)

    auto t_start = now();    // start ROI timer
    for (size_t i = 0; i < iters; ++i) {
        // ... perform core computation here ...
    }
    auto t_end = now();      // end ROI timer
    roi_us = duration_us(t_start, t_end); // write back roi time
}
```

* The final two parameters are supplied by ComPPare.
* `iters` lets you execute your region of interest many times for stable timing.
* `roi_us` is the user defined inner‑loop time -- recommended to repeat calculation for more accurate results. ComPPare records the full call time separately.


#### SAXPY function example signature:
```cpp
void saxpy_cpu(/*Input pack*/
            float a,
            const std::vector<float> &x,
            const std::vector<float> &y_in,
            /*Output pack*/
            std::vector<float> &y_out,
            /*Iterations*/
            size_t iters,
            /*Region of Interest timing (us)*/
            double &roi_us)
```


### 2. Create a comparison object

1. **Describe the types** — list the *input* types first, then the *output* types:

```cpp
using Cmp = 
    ComPPare::
        /*Define Input Pack Types same as the function*/
        InputContext<float, 
            std::vector<float>, 
            std::vector<float>>::
                /*Define Output Pack types same as the function*/
                OutputContext<std::vector<float>>;
```

2. **Pass the input data** — constructs framework object with input data that will be reused for every implementation:

```cpp
Cmp cmp(a, x, y);   // a: float, x: input vector, y: initial output
```

### 3. Register implementations

```cpp
cmp.set_reference("CPU serial", cpu_std);  // reference must be set first
cmp.add("CPU OpenMP",  cpu_omp);           // any number of additional back‑ends
```

### 4. Run and inspect

```cpp
cmp.run(/*iterations*/50, 
        /*tolerance*/1e-6, 
        /*warmup run before benchmark*/true);
```

#### Sample report:

```
Name           Func µs   ROI µs   Ovhd µs  Max|err|[0]  …
CPU serial     1.34e+5   …        …        0.00e+00     OK
CPU OpenMP     2.96e+4   …        …        0.00e+00     OK
```

### Complete example with SAXPY

```cpp
#include "ComPPare.hpp"
#include <vector>

// Serial reference
void saxpy_cpu(/*Input pack*/
               float a,
               const std::vector<float> &x,
               const std::vector<float> &y_in,
               /*Output pack*/
               std::vector<float> &y_out,
               /*Iterations*/
               size_t iters,
               /*Region of Interest timing (us)*/
               double &roi_us);

// OpenMP variant
void saxpy_cpu_omp(/* same signature */);

int main() {
    constexpr size_t N = 1 << 24;
    const float a = 5.55f;
    std::vector<float> x(N), y(N); 

    using Cmp = 
        ComPPare::
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
    Cmp cmp(a, x, y);

    // set reference to compare against
    cmp.set_reference(/*name in report*/"CPU serial", /*function*/saxpy_cpu);
    // add implementation
    cmp.add("CPU OpenMP", saxpy_cpu_omp);

    cmp.run(/*iterations*/50, 
            /*tolerance*/1e-6, 
            /*warmup run before benchmark*/true);
}
```

---




