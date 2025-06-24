# ComPPare

`ComPPare` is a tiny header-only utility that lets you **compare any number of host-side
implementations**—CPU loops, CUDA launchers, OpenMP, TBB, etc.—that share the same *inputs* and
produce the same *outputs*.  
For each implementation the framework:

| What it does                                   | Column in the report |
|------------------------------------------------|----------------------|
| Runs the entire function         | **Func µs** (whole call) |
| User Defined Region of Interest within their function  | **ROI µs** |
| Difference in time between entire function call to ROI    | **Ovhd µs** (setup / transfers) |
| Computes correctness vs. the reference implementation        | **Max / Mean / Total err** + index of the worst element |

---

## 1  Quick start

```cpp
#include "compare_framework.hpp"

/*
See example/saxpy
*/
using ScalarType = float;
using VectorType = std::vector<float>;

ComPPare::
    InputContext<ScalarType, VectorType, VectorType>::
        OutputContext<VectorType>
            compare(a, x, y);

// Set reference implementation
compare.set_reference(/*function name*/"cpu serial", /*function pointer*/cpu_std);
// Add implementations to compare
compare.add(/*function name*/"cpu parallel", /*function pointer*/cpu_par);

/* run 50 iterations; 
tolerate max-abs error ≤ 1e-6; 
with warmup function call before benchmark
*/
compare.run(/*iterations*/50, /*tolerance*/1e-6, /*warmup*/true);
````

Example output:

```bash
=== SAXPY Benchmark Parameters ===
Vector size (N)     : 134217728
Scalar a            : 10.98
Iterations          : 100
Error tolerance     : 1e-08
===================================

Name                        Func µs          Core µs          Ovhd µs       Max|err|[0]      (MaxErr-idx)      Mean|err|[0]     Total|err|[0]
cpu serial                 134034.89         132343.70           1691.18          0.00e+00               —          0.00e+00          0.00e+00
cpu parallel                29604.00          27894.86           1709.14          0.00e+00               —          0.00e+00          0.00e+00
gpu std par                  4489.56           2774.86           1714.70          0.00e+00               —          0.00e+00          0.00e+00
gpu kernel                   6943.10           1936.11           5006.99          1.10e+13           5254268          2.74e+12          3.68e+20  <-- FAIL
```


---

## 2  Anatomy at a glance

```
ComPPare
│
└───InputContext<Inputs...>          // pack input arguments
     │  inputs_  : std::tuple<Inputs...>
     │
     └───OutputContext<Outputs...>   // pack output arguments
          │
          │  add(name, fn)           // add function/implementation
          │  run(iters, tol)         // benchmark & compare
          │
          │
          └─ ErrorStats              // Comparison of Error betweeen implementation and reference
```

---

## 3  Implementation signature

```cpp
void impl(const Inputs&... in,      // read-only input arguments
          Outputs&...      out,     // output arguments, each will be compared to refernece
          size_t           iters,   // iterations of a loop -- recommended for accurate benchmark to run multiple times
          double&          roi_us); // user defined roi timing, for instance time taken for the hot loop to complete
```

* The framework measures **Func µs** on its own.
* You measure the **roi µs** (just the inner loop / kernel/ anything you want).
* It prints **Ovhd µs = Func µs − roi µs**.

---

## 4  Error metrics (`ErrorStats`)

| Field                       | Meaning                    |
| --------------------------- | -------------------------- |
| `Max`                       | largest absolute error     |
| `MaxErr-idx`                | index where `max` occurred |
| `Mean`                      | average                    |
| `Total`                     | Σ                          |


The **first** added implementation is the reference.
prints **“<-- FAIL”** whenever `max > tol`.

---
