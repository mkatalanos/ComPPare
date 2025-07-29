# User Guide ComPPare -- Validation & Benchmarking Framework <!-- omit from toc -->
- [1. Install](#1-install)
  - [1.1. Clone repository](#11-clone-repository)
  - [1.2. (Optional) Build Google Benchmark](#12-optional-build-google-benchmark)
  - [1.3. Include ComPPare](#13-include-comppare)
- [2. Basic Usage](#2-basic-usage)
  - [2.1. Adopt the required function signature](#21-adopt-the-required-function-signature)
  - [2.2. Add `HOTLOOP` Macros](#22-add-hotloop-macros)
  - [2.3. Setting up Comparison in `main()`](#23-setting-up-comparison-in-main)
  - [2.4. Command Line Options -- Iterations](#24-command-line-options----iterations)
- [3. Basic Working Principle](#3-basic-working-principle)
  - [3.1. `HOTLOOP` Macros](#31-hotloop-macros)
  - [3.2. Output Comparison](#32-output-comparison)
  
## 1. Install
### 1.1. Clone repository
```bash
git clone git@github.com:funglf/ComPPare.git --recursive
```
#### If submodules like google benchmark/ nvbench is not needed: <!-- omit from toc -->
```bash
git clone git@github.com:funglf/ComPPare.git
```

### 1.2. (Optional) Build Google Benchmark
See [Google Benchmark Instructions](https://github.com/google/benchmark/blob/b20cea674170b2ba45da0dfaf03953cdea473d0d/README.md) 

### 1.3. Include ComPPare 
In your C++ code, simply include the comppare header file by:
```c
#include <comppare/comppare.hpp>
```

## 2. Basic Usage
There are a few rules in order to use comppare.


### 2.1. Adopt the required function signature
*Function output must be `void`*
and consists of the input, then output types
```cpp
void impl(const Inputs&... in,     // read‑only inputs
        Outputs&...      out);     // outputs compared to reference
```

### 2.2. Add `HOTLOOP` Macros
In order to benchmark specific regions of code, following Macros `HOTLOOPSTART`, `HOTLOOPEND` are needed. The region in between will be ran multiple times in order to get an accurate timing. The region first runs certain iterations of warmup before actually running the benchmark iterations.

#### How to use `HOTLOOPSTART`/`HOTLOOPEND` Macros <!-- omit from toc -->
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

#### Alternate Macro <!-- omit from toc -->
Alternatively, the macro `HOTLOOP()` can be used to wrap the whole region.
```cpp
void impl(const Inputs&... in,
        Outputs&...      out);
{
    HOTLOOP(
    // ... perform core computation here ...
    );
}
```

### 2.3. Setting up Comparison in `main()`
In `main()`, you can setup the comparison, such as defining the reference function, initializing input data, naming of benchmark etc.

> ### The SAXPY example will be used throughout this section to demonstrate on usage <!-- omit from toc -->
>**SAXPY** stands for **Single-Precision A·X Plus Y**.  [(Also see examples/saxpy)](../examples/saxpy/README.md)
> 
> It stands for the following operation:
> 
>$$
>y_{\text{out}}[i] = a \cdot x[i] + y[i]
>$$


Based on [Section [Basic Usage]](#2-basic-usage), we can define a `saxpy` function as:
```cpp
void saxpy_cpu(/*Input types*/
               float a,
               const std::vector<float> &x,
               const std::vector<float> &y_in,
               /*Output types*/
               std::vector<float> &y_out)
{
    HOTLOOPSTART;
    for (size_t i = 0; i < x.size(); ++i)
        y_out[i] = a * x[i] + y_in[i];
    HOTLOOPEND;
}
```
---
#### Step 1: Initialize Input data <!-- omit from toc -->
```cpp
/* Intialize Input data */
float a = 1.1f
std::vector x(1000, 1.0); 
std::vector y_in(1000, 2.0);
```
---
#### Step 2: Create a Comparison Object <!-- omit from toc -->
This step defines the input and output types of the functions you are about to test.

For `saxpy_cpu`, the **input types** are:
```cpp
/* INPUTS: */
float,
std::vector<float>,
std::vector<float>
```

Let's define an alias here based on the Input Types:
```cpp
using CMP_INPUT = comppare::
    /*Define Input Pack Types same as the function*/
    InputContext<
                    float,              /*float a*/
                    std::vector<float>, /*std::vector<float> x*/
                    std::vector<float>  /*std::vector<float> y*/
                >::
```

---

As for the **output types**:
```cpp
/* OUTPUTS: */
std::vector<float>
```
Let's define another alias based on the Output Types:
```cpp
using CMP = CMP_INPUT::
    /*Define Output Pack types same as the function*/
    OutputContext<
                    std::vector<float>  /*std::vector<float> y_out*/
                 >;
```

---

Finally, we can create our comparison object with the alias `CMP` and initialize with the input data:
```cpp
CMP cmp(a,x,y_in); // Input data initialized above
```

--- 
**Or, define all at once without alias:**

Therefore, we can define the comparison object as:
```cpp
comppare:: /*namespace comppare*/
    /*Define Input Pack Types same as the function*/
    InputContext<
                    float,              /*float a*/
                    std::vector<float>, /*std::vector<float> x*/
                    std::vector<float>  /*std::vector<float> y*/
                >::
    /*Define Output Pack types same as the function*/
    OutputContext<
                    std::vector<float>  /*std::vector<float> y_out*/
                 >;
    
    cmp(a,x,y_in);   //cmp object -- a,x,y data
```

#### Step 3: Register/Add Functions into framework <!-- omit from toc -->
After creating the `cmp` object, we can add functions into it.

To Define the reference function:
```cpp
cmp.set_reference(/*Displayed Name After Benchmark*/"saxpy reference", /*Function*/saxpy_cpu);
```

To Add more fucntions:
```cpp
cmp.add("saxpy gpu", saxpy_gpu);
``` 

#### Step 4: Run the Benchmarks <!-- omit from toc -->
Command line arguments are the parameters for run():
```cpp
cmp.run(argc, argv);
```

#### Step 5: Results <!-- omit from toc -->
Example Output:
```bash
*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
============ ComPPare Framework ============
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

Number of implementations:             3
Warmup iterations:                   100
Benchmark iterations:                100
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

Implementation                  Func µs             ROI µs            Ovhd µs         Max|err|[0]        Mean|err|[0]       Total|err|[0]
saxpy reference                    38.01               15.97               22.05            0.00e+00            0.00e+00            0.00e+00
saxpy gpu                       73151.41                1.07            73150.34            3.30e+06            1.66e+06            1.70e+09  <-- FAIL
```
In this case, `saxpy_gpu` failed.

### 2.4. Command Line Options -- Iterations
There are 2 commmand line options to control the number of warmup and benchmark iterations: 

#### `--warmups` Warmup Iterations <!-- omit from toc -->
When used with a unsigned 64 bit integer, this sets the number of warmup iterations before running benchmark runs.

Example:
```bash
./saxpy --warmups 1000
```

#### `--iters` Benchmark Iterations  <!-- omit from toc -->
When used with a unsigned 64 bit integer, this sets the number of benchmark iterations

Example:
```bash
./saxpy --iters 1000
```

## 3. Basic Working Principle

### 3.1. `HOTLOOP` Macros

HOTLOOP Macros essentially wrap your region of interest in a lambda before running the lambda across the warmup and benchmark iterations. 
The region of iterest is being timed across all the benchmark iterations to obtain an average runtime. 

In `comppare/comppare.hpp`, `HOTLOOPSTART` and `HOTLOOPEND` are defined as:
```cpp
#define HOTLOOPSTART \
    auto &&hotloop_body = [&]() { /* start of lambda */

#define HOTLOOPEND     \
    }; /* end lambda */ \
``` 

Therefore, any code in between the 2 macros are simply wrapped into the lambda `hotloop_body`:
```cpp
HOTLOOPSTART;
foo();
HOTLOOPEND;

/* is equivalent to */

auto &&hotloop_body = [&]() {
    foo();
};
```

After that, the lambda is being ran:
```cpp
/* Warm-up */                                                      
for (i = 0; i < warmup_iterations; ++i) 
    hotloop_body();                                                
                
/* Benchmark */
auto start = now();                        
for (i = 0; i < benchmark_iterations; ++i)  
    hotloop_body();                                                
auto end = now();                        
```

### 3.2. Output Comparison 

Each implementation follows the same signature:
```cpp
void impl(const Inputs&... in,     // read‑only inputs
        Outputs&...      out);     // outputs compared to reference
```
**Input** – Passed in by the framework. All candidates get exactly the same data.

**Output** – The framework creates private output objects and passes into the implementation by reference. Therefore any change in the object will be stored in the framework.

After each implementation has finished running, the framework compares each output against the reference implementation, and prints out the results in terms of difference/error and whether it has failed.


