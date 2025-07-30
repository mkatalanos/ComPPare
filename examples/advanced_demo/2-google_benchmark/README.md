# Advanced Demo 2 -- **Google Benchmark** <!-- omit from toc -->

- [1. Introduction](#1-introduction)
- [2. Quick Start](#2-quick-start)
  - [2.1. Build](#21-build)
  - [2.2. Run](#22-run)
  - [2.3. Troubleshoot](#23-troubleshoot)
    - [2.3.1. CMake cannot find Benchmark](#231-cmake-cannot-find-benchmark)
    - [2.3.2. Undefined Symbols in `make` stage](#232-undefined-symbols-in-make-stage)
    - [2.3.3. CMake could not find OpenMP\_CXX](#233-cmake-could-not-find-openmp_cxx)
- [3. Google Benchmark on top of ComPPare](#3-google-benchmark-on-top-of-comppare)
  - [3.1. How to use](#31-how-to-use)
  - [3.2. Advantages](#32-advantages)
    - [3.2.1. Less Code Change](#321-less-code-change)
    - [3.2.2. Additional Correctness Comparison](#322-additional-correctness-comparison)
  - [3.3. Limitations](#33-limitations)
    - [3.3.1. Unable to use Full Extent of Google Benchmark's capabilities](#331-unable-to-use-full-extent-of-google-benchmarks-capabilities)


## 1. Introduction
Google Benchmark is a provided external plugin with a some simple steps.

## 2. Quick Start
### 2.1. Build 
```bash 
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make 
```

### 2.2. Run 
```bash
./google_benchmark_demo
```

### 2.3. Troubleshoot
#### 2.3.1. CMake cannot find Benchmark
```bash
CMake Error at CMakeLists.txt:7 (find_package):
  Could not find a package configuration file provided by "benchmark" with
```
**Solution 1 -- build submodule benchmark**
Follow Guide provided by Google Benchmark at [Link](https://github.com/google/benchmark/blob/main/README.md) to build Google Benchmark

**Solution 2 -- Provide Absolute path of your own benchmark build**
To provide the location of another benchmark build, we need to provide CMake with the path via:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/your/path/to/benchmark/build #Change to your path
```
<br>

Alternatively, you can edit the CMakeLists.txt file, although this is not recommended:
```bash
BENCHMARK_ROOT=/your/path/to/benchmark/build #Change to your path
cd /your/path/to/ComPPare/examples/advanced_demo/2-google_benchmark # change dir to root of this example
sed -i.bak "s#\${CMAKE_SOURCE_DIR}/../../../benchmark/build#${BENCHMARK_ROOT}#g" CMakeLists.txt
```
If you are satisfied with the change, you can delete the Original CMakeLists.txt by:
```bash 
rm CMakeLists.txt.bak
```
If not satisfied with the change and want to revert to the original file:
```bash
rm CMakeLists.txt
mv CMakeLists.txt.bak CMakeLists.txt
```

---

#### 2.3.2. Undefined Symbols in `make` stage 
Usually a long list of functions are listed with Undefined symbols:
```bash
Undefined symbols for architecture arm64:
  "benchmark::internal::RegisterBenchmarkInternal( ...
  ...
9](sysinfo.cc.o)
   NOTE: a missing vtable usually means the first non-inline virtual member function has no definition.
ld: symbol(s) not found for architecture arm64
```
This is most likely being benchmark and the current demo is built with a different compiler -- in this case, benchmark is built with Apple Clang, while the demo is built with GNU g++.

Try to rebuild both benchmark and current demo with the keeping the compiler the same.


#### 2.3.3. CMake could not find OpenMP_CXX
```bash
Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS OpenMP_CXX_LIB_NAMES)
```
This is most probably due to the compiler you are using not supporting OpenMP, for instance AppleClang. Manually define compilers that supports OpenMP such as the GNU C++ compiler:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
```

**Note for Apple Users:**

 **`g++` is actually `AppleClang`**
, if you do:
```bash
g++ --version
>>> Apple clang version 15.0.0 (clang-1500.3.9.4)
```
Instead, you would have to install actual [GNU g++ with homebrew](https://formulae.brew.sh/formula/gcc)


## 3. Google Benchmark on top of ComPPare
### 3.1. How to use
**Original Code**
```cpp
void square(const std::vector<int> &vec, std::vector<int> &out)
{
    out.resize(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
    {
        out[i] = vec[i] * vec[i];
    }
}
```

**ComPPare**
As per usual, simply add the macros to the original code
```cpp
void ComPPare_square(const std::vector<int> &vec, std::vector<int> &out)
{
    out.resize(vec.size());
    HOTLOOPSTART;
    for (size_t i = 0; i < vec.size(); ++i)
    {
        out[i] = vec[i] * vec[i];
    }
    HOTLOOPEND;
}
```

To attach to google benchmark, when adding the function, simply add `.google_benchmark()`:
```cpp
compare.add("square", ComPPare_square).google_benchmark()
            ->Unit(benchmark::kMillisecond); // And any other Google Benchmark Arguments
```

### 3.2. Advantages 
#### 3.2.1. Less Code Change

**Google Benchmark**
Minimal example of using Google Benchmark incurs a substantial code change with the initialization moved into the function itself. 

```cpp
void GB_square(benchmark::State& state)
{
    /* Declare Variables */
    std::vector<float> out, vec;

    /* Initialize Variables */
    vec.resize(100);
    out.resize(100);
    std::iota(vec.begin(), vec.end(), 1); // Or your desiered way of initializing
    
    for (auto _ : state)
    {
        for (size_t i = 0; i < vec.size(); ++i)
        {
            out[i] = vec[i] * vec[i];
        }
    }
}
```
As shown, to write a function for Google Benchmark requires a certain degree of code addition. As for ComPPare, only 2 macros are needed for a basic case.

> Note on Setup in Google Benchmark:
> 
> You can do a setup and teardown function in Google Benchmark to encapsulate them. They will still be invoked once per Benchmark/Implementation. [(See here)](https://github.com/google/benchmark/blob/main/docs/user_guide.md#setupteardown)

#### 3.2.2. Additional Correctness Comparison
ComPPare's defining difference is being able to test for correctness. Apart from pure benchmarking.

### 3.3. Limitations
#### 3.3.1. Unable to use Full Extent of Google Benchmark's capabilities
Google Benchmark supports great capabilities such as Passing Arguments. 

**Passing Arguments**
Google Benchmark supports passing arguments by:
```cpp
void GB_square(benchmark::State& state)
{
    size_t N = state.range(0);
    /* Declare Variables */
    std::vector<float> out, vec;

    /* Initialize Variables */
    vec.resize(N);
    out.resize(N);
    std::iota(vec.begin(), vec.end(), 1); // Or your desiered way of initializing
    
    for (auto _ : state)
    {
        for (size_t i = 0; i < vec.size(); ++i)
        {
            out[i] = vec[i] * vec[i];
        }
    }
}

BENCHMARK(GB_square)->Arg(1 << 8)->Arg(1 << 16)
```
Where in this case, we are benchmarking the vector sizes of 2^8 and 2^16. However, in comppare, we cannot do this as our function is not written for Google Benchmark. 

Instead, you will have to write your own input cases with different sizes:
```cpp
int main(int argc, char** argcv)
{
    uint64_t N = std::stuoll(argv[1]);

    std::vector<float> out, vec;
    /* Initialize Variables */
    vec.resize(N);
    out.resize(N);
    std::iota(vec.begin(), vec.end(), 1);

    /* ComPPare section
        1. Create comppare instance
        2. Add functions
        3. Run
    */ 
}
```
Other functionality include multithreading support, filtering out benchmark cases in command line, memory usage etc. 

> Google Benchmark and ComPPare has different purpose in the for code optimisation. ComPPare is for correctness testing while being able to run microbenchmarks when prototyping; Google Benchmark is only for benchmark with more complete capabilities. To Fully use the extent of capabilities of Google Benchmark, writing specifically for Google Benchmark is otherwise suggested.