<style>
details summary {
  list-style: none;
  margin: 1em 0 0.5em;
  font-size: 1.125em;
  font-weight: bold;
  cursor: pointer;
}

details summary::-webkit-details-marker {
  display: none;
}

details summary::before {
  content: "▶";
  display: inline-block;
  font-size: 1.5em;
  line-height: 1;
  margin-right: 0.5em;
  transition: transform 0.2s ease;
}

details[open] summary::before {
  content: "▼";
}
</style>

# Advanced Demo 4 — **DoNotOptimize** <!-- omit from toc -->

- [1. Introduction](#1-introduction)
- [2. Quick Start](#2-quick-start)
  - [2.1. Build](#21-build)
  - [2.2. Run](#22-run)
  - [2.3. Results](#23-results)
    - [2.3.1. Interpreting Results](#231-interpreting-results)
- [3. Problem of Optimization](#3-problem-of-optimization)
    - [3.0.1. On AArch64 (Apple M2)](#301-on-aarch64-apple-m2)
    - [3.0.2. On x86\_64](#302-on-x86_64)
- [4. “Just don’t optimize”…?](#4-just-dont-optimize)
    - [4.0.1. On AArch64 (Apple M2)](#401-on-aarch64-apple-m2)
    - [4.0.2. On x86\_64](#402-on-x86_64)
  - [4.1. Why Optimize?](#41-why-optimize)
- [5. Google Benchmark's Solution](#5-google-benchmarks-solution)
  - [5.1. On AArch64 (Apple M2)](#51-on-aarch64-apple-m2)
  - [5.2. On x86\_64](#52-on-x86_64)
    - [5.2.1. Conclusion -- `DoNotOptimize()` works!](#521-conclusion----donotoptimize-works)
- [6. Inspecting `DoNotOptimize()`](#6-inspecting-donotoptimize)
  - [6.1. Breakdown 1: Assembly](#61-breakdown-1-assembly)
  - [6.2. Breakdown 2: Meaning of this asm code](#62-breakdown-2-meaning-of-this-asm-code)
  - [6.3. Breakdown 3: `volatile`](#63-breakdown-3-volatile)
  - [6.4. Conclusion](#64-conclusion)
  - [6.5. `DoNotOptimize()` in practice](#65-donotoptimize-in-practice)
- [7. Example Case](#7-example-case)
  - [7.1. Example Source Code](#71-example-source-code)
  - [7.2. "Performance" difference](#72-performance-difference)
  - [7.3. Compiler Warning](#73-compiler-warning)
  - [7.4. Inspection of the full asm code](#74-inspection-of-the-full-asm-code)
    - [7.4.1. On AArch64 (Apple M2)](#741-on-aarch64-apple-m2)
    - [7.4.2. On x86\_64](#742-on-x86_64)

## 1. Introduction
This example is used to demonstrate the functionality of `comppare::DoNotOptimize()`.
This function is an **EXACT COPY of Google Benchmark's [(github link)](https://github.com/google/benchmark) `benchmark::DoNotOptimize()`**

This document will provide the explanation of the purpose, and mechanism of the function `DoNotOptimize()`.

### References: <!-- omit from toc -->
1. [CppCon 2015: Chandler Carruth "Tuning C++: Benchmarks, and CPUs, and Compilers! Oh My!"](https://www.youtube.com/watch?v=nXaxk27zwlk&t=3319s) <br>
2. [Google Benchmark Github Repository](https://github.com/google/benchmark) <br>
3. [GNU Extended ASM Documentation](https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html#Clobbers-and-Scratch-Registers) <br>

### Disclaimer  <!-- omit from toc -->
I, LF Fung, am not the author of `DoNotOptimize()`. The implementation of `comppare::DoNotOptimize()` is a verbatim of Google Benchmark's `benchmark::DoNotOptimize()`.

## 2. Quick Start
### 2.1. Build
```bash 
mkdir build 
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make 
```

#### 1.1.1.1. (Optional) Build with Google Benchmark <!-- omit from toc -->
At the CMake step:
```bash 
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GOOGLE_BENCHMARK=ON
```

### 2.2. Run
```bash
./DoNotOptimize_demo --warmup 10000 --iters 10000
```

### 2.3. Results 
**On AArch64 Apple M2**
```bash
*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
============ ComPPare Framework ============
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

Number of implementations:             2
Warmup iterations:                 10000
Benchmark iterations:              10000
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

Implementation             Func µs/Iter        ROI µs/Iter       Ovhd µs/Iter
SAXPY                               0.00                0.00                0.00
SAXPY_DONOTOPTIMIZE                57.14               28.60               28.54
```

**Additional Results from Google Benchmark** <!-- omit from toc -->
```bash
--------------------------------------------------------------
Benchmark                    Time             CPU   Iterations
--------------------------------------------------------------
SAXPY                    0.000 ns        0.000 ns   1000000000000
SAXPY_DONOTOPTIMIZE      28594 ns        28427 ns        25001
```

---
#### 2.3.1. Interpreting Results
If you inspect `DoNotOptimize_demo.cpp`, both functions are practically the same, with the exception of an addition of `DoNotOptimize()`. Read below to breakdown the reason to this performance difference.

## 3. Problem of Optimization
Compiler optimization can sometimes remove operations and variables completely.

In `saxpy.cpp`:

```cpp
void SAXPY(const float a, const float* x, const float* y)
{
    float yout;
    for (int i = 0; i < N; ++i)
    {
        yout = a * x[i] + y[i];
    }
}
```

When compiling at high optimization, the compiler realizes `yout` is just a temporary that’s never used elsewhere. As a result, `yout` is optimized out, and thus the whole saxpy operation would be optimized out.

To verify that it’s been optimized away, compile to assembly and inspect the output:

```bash
export CXX=g++ #or your choice of compiler
${CXX} -O3 -std=c++20 -S -o saxpy.s saxpy.cpp
```

---

#### 3.0.1. On AArch64 (Apple M2)

`saxpy.s` compiled with `Apple clang version 15.0.0 (clang-1500.3.9.4)` on `arm64-apple-darwin23.1.0` (Apple M2 chip):

```asm
__Z5SAXPYfPKfS0_:                       ; @_Z5SAXPYfPKfS0_
	.cfi_startproc
; %bb.0:
	ret
	.cfi_endproc
                                        ; -- End function
```

The function body is practically empty: a single `ret` which is “return from subroutine” [Arm A64 Instruction Set: **RET**](https://developer.arm.com/documentation/dui0802/b/A64-General-Instructions/RET). In simple terms, it just returns — nothing happens.

---

#### 3.0.2. On x86\_64

The same phenomenon appears on x86 as well.

`saxpy.s` compiled with `g++ (GCC) 15.1.0` on `x86_64 Intel(R) Xeon(R) Platinum 8468`

```asm
_Z5SAXPYfPKfS0_:
.LFB0:
	.cfi_startproc
	ret
	.cfi_endproc
```

Intel® 64 and IA-32 Architectures Software Developer’s Manual
Volume 2 (2A, 2B, 2C, & 2D): Instruction Set Reference, A–Z: Vol. 2B, 4-560, **RET — Return from Procedure**.

---

## 4. “Just don’t optimize”…?

Compiling with `-O0` should prevent the loop from being optimized away.
For the sake of verifying the effects of no optimisation, we can again compile to assembly.
```bash
export CXX=g++ #or your choice of compiler
${CXX} -O0 -std=c++20 -S -o saxpy.s saxpy.cpp
```

---

#### 4.0.1. On AArch64 (Apple M2)
`saxpy.s` compiled with `Apple clang version 15.0.0 (clang-1500.3.9.4)` on `arm64-apple-darwin23.1.0` (Apple M2 chip):

<details>
<summary> Full AArch64 Assembly code of <code>SAXPY()</code> with NO optimization </summary>
<pre><code class="language-asm">
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">5</span> __Z5SAXPYfPKfS0_:                       ; @_Z5SAXPYfPKfS0_
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">6</span>       .cfi_startproc
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">7</span> ; %bb.0:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">8</span>       sub     sp, sp, #32
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">9</span>       .cfi_def_cfa_offset 32
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">10</span>      str     s0, [sp, #28]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">11</span>      str     x0, [sp, #16]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">12</span>      str     x1, [sp, #8]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">13</span>      str     wzr, [sp]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">14</span>      b       LBB0_1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">15</span> LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">16</span>      ldr     w8, [sp]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">17</span>      mov     w9, #16960
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">18</span>      movk    w9, #15, lsl #16
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">19</span>      subs    w8, w8, w9
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">20</span>      cset    w8, ge
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">21</span>      tbnz    w8, #0, LBB0_4
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">22</span>      b       LBB0_2
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">23</span> LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">24</span>      ldr     s0, [sp, #28]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">25</span>      ldr     x8, [sp, #16]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">26</span>      ldrsw   x9, [sp]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">27</span>      ldr     s1, [x8, x9, lsl #2]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">28</span>      ldr     x8, [sp, #8]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">29</span>      ldrsw   x9, [sp]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">30</span>      ldr     s2, [x8, x9, lsl #2]
<span style="display:inline-block;width:100%;white-space:pre;background-color:rgba(0, 255, 4, 0.49);border-radius:2px;"><span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">31</span>      fmadd   s0, s0, s1, s2</span>
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">32</span>      str     s0, [sp, #4]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">33</span>      b       LBB0_3
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">34</span> LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">35</span>      ldr     w8, [sp]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">36</span>      add     w8, w8, #1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">37</span>      str     w8, [sp]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">38</span>      b       LBB0_1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">39</span> LBB0_4:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">40</span>      add     sp, sp, #32
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">41</span>      ret
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">42</span>      .cfi_endproc
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">42</span>                                        ; -- End function
</code></pre>
</details>

<br>


<a id="saxpyappleM2isa"></a>
The SAXPY operation can be found as Fused-Multiply-Add instruction on line 31:
<pre><code class="language-asm">
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">31</span>      fmadd   s0, s0, s1, s2
</code></pre>

[Reference to Arm A64 Instruction Set: **FMADD**](https://developer.arm.com/documentation/ddi0602/2025-06/SIMD-FP-Instructions/FMADD--Floating-point-fused-multiply-add--scalar--)


---

#### 4.0.2. On x86\_64

`saxpy.s` compiled with `g++ (GCC) 15.1.0` on `x86_64 Intel(R) Xeon(R) Platinum 8468`

<details>
<summary> Full x86_64 Assembly code of <code>SAXPY()</code> with NO optimization </summary>
<pre><code class="language-asm">
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">5</span> _Z5SAXPYfPKfS0_:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">6</span> .LFB0:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">7</span>       .cfi_startproc
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">8</span>       pushq   %rbp
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">9</span>       .cfi_def_cfa_offset 16
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">10</span>      .cfi_offset 6, -16
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">11</span>      movq    %rsp, %rbp
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">12</span>      .cfi_def_cfa_register 6
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">13</span>      movss   %xmm0, -20(%rbp)
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">14</span>      movq    %rdi, -32(%rbp)
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">15</span>      movq    %rsi, -40(%rbp)
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">16</span>      movl    $0, -4(%rbp)
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">17</span>      jmp     .L2
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">18</span> .L3:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">19</span>      movl    -4(%rbp), %eax
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">20</span>      cltq
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">21</span>      leaq    0(,%rax,4), %rdx
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">22</span>      movq    -32(%rbp), %rax
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">23</span>      addq    %rdx, %rax
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">24</span>      movss   (%rax), %xmm0
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">25</span>      movaps  %xmm0, %xmm1
<span style="display:inline-block;width:100%;white-space:pre;background-color:rgba(0, 255, 4, 0.49);border-radius:2px;"><span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">26</span>      mulss   -20(%rbp), %xmm1</span>
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">27</span>      movl    -4(%rbp), %eax
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">28</span>      cltq
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">29</span>      leaq    0(,%rax,4), %rdx
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">30</span>      movq    -40(%rbp), %rax
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">31</span>      addq    %rdx, %rax
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">32</span>      movss   (%rax), %xmm0
<span style="display:inline-block;width:100%;white-space:pre;background-color:rgba(0, 255, 4, 0.49);border-radius:2px;"><span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">33</span>      addss   %xmm1, %xmm0</span>
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">34</span>      movss   %xmm0, -8(%rbp)
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">35</span>      addl    $1, -4(%rbp)
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">36</span> .L2:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">37</span>      cmpl    $999999, -4(%rbp)
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">38</span>      jle     .L3
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">39</span>      nop
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">40</span>      nop
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">41</span>      popq    %rbp
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">42</span>      .cfi_def_cfa 7, 8
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">43</span>      ret
</code></pre>
</details>

<br>

<a id="saxpy_x86_isa"></a>
The SAXPY operation can also be found as separate multiply and addition scalar operations:
<pre><code class="language-asm">
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">26</span>      mulss	-20(%rbp), %xmm1 
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">33</span>      addss	%xmm1, %xmm0
</code></pre>


Intel® 64 and IA-32 Architectures Software Developer’s Manual
Volume 2 (2A, 2B, 2C, & 2D): Instruction Set Reference, A–Z: 
Vol. 2B, 4-151, **MULSS — Multiply Scalar Single Precision Floating-Point Values**.
Vol. 2A, 3-24 , **ADDSS—Add Scalar Single Precision Floating-Point Values**.

---

### 4.1. Why Optimize?

From these results, it shows that to avoid compiler optimizing away the operation is to simply turn off optimization completely. However, benchmarking the operations without optimisation may not be relavant to real-world performance as often production code are optimized. Therefore, we still need optimization, at the same time not optimize away the operation completely.

---

## 5. Google Benchmark's Solution
Optimization is important to understand the performance of particular operations in production builds. This creates the conflicting ideas of `optimize` but `do not optimize away`. This was solved by Google in their [benchmark](https://github.com/google/benchmark) -- a microbenchmarking library. Google Benchmark provides `benchmark::DoNotOptimize()` to prevent variables from being optimized away.

With the same SAXPY function, we simply add DoNotOptimize() around the temporary variable `yout`
```cpp
void SAXPY_DONOTOPTIMIZE(const float a, const float* x, const float* y)
{
    float yout;
    for (int i = 0; i < N; ++i)
    {
        yout = a * x[i] + y[i];
        DoNotOptimize(yout);
    }
}
```

This `DoNotOptimize` call tells the compiler not to eliminate the temporary variable, so the operation itself won’t be optimized away.

---
Compile with aggressive optimization and inspect the assembly:

```bash
export CXX=g++ # or your choice of compiler
${CXX} -O3 -std=c++20 -S -o saxpy_DoNotOptimize.s saxpy_DoNotOptimize.cpp
```

### 5.1. On AArch64 (Apple M2)

Compiled with `Apple clang version 15.0.0 (clang-1500.3.9.4)` on `arm64-apple-darwin23.1.0`:

<!-- <details> -->
<summary> Full AArch64 Assembly code of <code>SAXPY_DONOTOPTIMIZE()</code> with <code>-O3</code> optimization </summary>
<pre><code class="language-asm">
<!-- <span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">1</span>       .section        __TEXT,__text,regular,pure_instructions
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">2</span>       .build_version macos, 14, 0     sdk_version 14, 4
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">3</span>       .globl  __Z19SAXPY_DONOTOPTIMIZEfPKfS0_ ; -- Begin function _Z19SAXPY_DONOTOPTIMIZEfPKfS0_
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">4</span>       .p2align        2 --><span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">5</span> __Z19SAXPY_DONOTOPTIMIZEfPKfS0_:        ; @_Z19SAXPY_DONOTOPTIMIZEfPKfS0_
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">6</span>       .cfi_startproc
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">7</span> ; %bb.0:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">8</span>       sub     sp, sp, #16
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">9</span>       .cfi_def_cfa_offset 16
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">10</span>      mov     w8, #16960
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">11</span>      movk    w8, #15, lsl #16
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">12</span>      add     x9, sp, #4
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">13</span>      add     x10, sp, #8
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">14</span> LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">15</span>      ldr     s1, [x0], #4
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">16</span>      ldr     s2, [x1], #4
<span style="display:inline-block;width:100%;white-space:pre;background-color:rgba(0, 255, 4, 0.49);border-radius:2px;"><span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">17</span>      fmadd   s1, s0, s1, s2</span>
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">18</span>      str     s1, [sp, #4]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">19</span>      str     x9, [sp, #8]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">20</span>      ; InlineAsm Start
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">21</span>      ; InlineAsm End
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">22</span>      subs    x8, x8, #1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">23</span>      b.ne    LBB0_1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">24</span> ; %bb.2:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">25</span>      add     sp, sp, #16
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">26</span>      ret
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">27</span>      .cfi_endproc
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">28</span>                                         ; -- End function
</code></pre>
<!-- </details> -->
<br>

Further inspection reveals the Fused-Multiply-Add instruction -- indicating that SAXPY operation was not optimized away.
<pre><code class="language-asm">
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">17</span>      fmadd   s0, s0, s1, s2
</code></pre>


[Reference to Arm A64 Instruction Set: **FMADD**](https://developer.arm.com/documentation/ddi0602/2025-06/SIMD-FP-Instructions/FMADD--Floating-point-fused-multiply-add--scalar--)

### 5.2. On x86\_64

Compiled with `g++ (GCC) 15.1.0` on `x86_64 Intel(R) Xeon(R) Platinum 8468`
<!-- 
```asm
_Z19SAXPY_DONOTOPTIMIZEfPKfS0_:
.LFB1:
	.cfi_startproc
	xorl	%eax, %eax
	.p2align 4
	.p2align 3
.L2:
	movss	(%rdi,%rax), %xmm1
	mulss	%xmm0, %xmm1
	addss	(%rsi,%rax), %xmm1
	movd	%xmm1, %edx
	addq	$4, %rax
	cmpq	$4000000, %rax
	jne	.L2
	ret
	.cfi_endproc
``` -->

<summary> Full x86_64 Assembly code of <code>SAXPY_DONOTOPTIMIZE()</code> with <code>-O3</code> optimization </summary>
<pre><code class="language-asm">
<!-- <span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">1</span>       .file   "saxpy_DoNotOptimize.cpp"
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">2</span>       .text
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">3</span>       .p2align 4
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">4</span>       .globl  _Z19SAXPY_DONOTOPTIMIZEfPKfS0_
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">5</span>       .type   _Z19SAXPY_DONOTOPTIMIZEfPKfS0_, @function --><span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">6</span> _Z19SAXPY_DONOTOPTIMIZEfPKfS0_:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">7</span> .LFB1:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">8</span>       .cfi_startproc
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">9</span>       movaps  %xmm0, %xmm1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">10</span>      xorl    %eax, %eax
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">11</span>      .p2align 4
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">12</span>      .p2align 3
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">13</span> .L2:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">14</span>      movss   (%rdi,%rax), %xmm0
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">15</span>      leaq    -4(%rsp), %rdx
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">16</span>      movq    %rdx, -24(%rsp)
<span style="display:inline-block;width:100%;white-space:pre;background-color:rgba(0, 255, 4, 0.49);border-radius:2px;"><span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">17</span>      mulss   %xmm1, %xmm0
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">18</span>      addss   (%rsi,%rax), %xmm0</span>
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">19</span>      movss   %xmm0, -4(%rsp)
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">20</span>      addq    $4, %rax
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">21</span>      cmpq    $4000000, %rax
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">22</span>      jne     .L2
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">23</span>      ret
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">24</span>      .cfi_endproc
<!-- <span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">25</span> .LFE1:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">26</span>      .size   _Z19SAXPY_DONOTOPTIMIZEfPKfS0_, .-_Z19SAXPY_DONOTOPTIMIZEfPKfS0_
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">27</span>      .ident  "GCC: (GNU) 15.1.0"
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">28</span>      .section        .note.GNU-stack,"",@progbits -->
</code></pre>


Again, further inspection shows that SAXPY operation is not optimized out.
<pre><code class="language-asm">
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">17</span>      mulss	-20(%rbp), %xmm1 
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">18</span>      addss	%xmm1, %xmm0
</code></pre>

Intel® 64 and IA-32 Architectures Software Developer’s Manual
Volume 2 (2A, 2B, 2C, & 2D): Instruction Set Reference, A–Z: 
Vol. 2B, 4-151, **MULSS — Multiply Scalar Single Precision Floating-Point Values**.
Vol. 2A, 3-24 , **ADDSS—Add Scalar Single Precision Floating-Point Values**.

---
#### 5.2.1. Conclusion -- `DoNotOptimize()` works!
Code can still be optimized while keeping the operation. Next section will examine how does it work.

---

## 6. Inspecting `DoNotOptimize()`

We can define DoNotOptimize minimally as:
```cpp
void DoNotOptimize(void * p)
{
    asm volatile("" : "+m,r"(p) : : "memory");
}
```
---
### 6.1. Breakdown 1: Assembly

`DoNotOptimize()` is an extended assembly code, the GNU documentation on [*Extended asm*](https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html):

```
asm asm-qualifiers ( AssemblerTemplate 
                 : OutputOperands 
                 [ : InputOperands
                 [ : Clobbers ] ])
```

* **AssemblerTemplate** is the assembly code/instructions.
* **OutputOperands** are the outputs back into C/C++.
* **InputOperands** are the inputs from C/C++.
* **Clobbers** list what this asm code modifies.

### 6.2. Breakdown 2: Meaning of this asm code

* **AssemblerTemplate** is empty: `""` — no actual instructions execute.

* **OutputOperands**:

  * `"+"` is a [Constraint Modifier Character](https://gcc.gnu.org/onlinedocs/gcc/Modifiers.html#Constraint-Modifier-Characters) indicating read *and* write by the asm.
  * `"m"` is a [Simple Constraint](https://gcc.gnu.org/onlinedocs/gcc/Simple-Constraints.html#Simple-Constraints-1) meaning the operand can be in main memory.
  * `"r"` is also a [Simple Constraint](https://gcc.gnu.org/onlinedocs/gcc/Simple-Constraints.html#Simple-Constraints-1) meaning the operand can be in register memory.
  * `"m,r"` provides [Multiple Alternatives](https://gcc.gnu.org/onlinedocs/gcc/Multi-Alternative.html#Multiple-Alternative-Constraints): the operand may be in main memory **or** register memory.

  **Combined:** `"+m,r"` means “treat `p` as read and written ("+"), where `p` is located either in main memory ("m") or a register("r").”

* **InputOperands**: none (the read/write behavior is implied by the output constraint).

* **Clobbers**: `"memory"` — indicates the asm may change memory, acting as a read/write memory barrier.

Putting it together: 
>“Read and write the memory location `p` (OutputOperands "+m,r"), with possible memory effects (Clobbers "memory"), with no actual operation (AssemblerTemplate "")”

### 6.3. Breakdown 3: `volatile`

The `volatile` asm-qualifier (see [GNU Extended Asm — Volatile](https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html#Volatile-1)) tells the compiler this asm block has important side effects. Without it, an optimizer could decide the output is unused and delete the asm — and, transitively, the memory location `p`. With `volatile`, the compiler must preserve the asm and thus the memory location `p`.


### 6.4. Conclusion

This asm line means:

> “This memory location `p` is read and written (`+m,r`), with observable effects on memory (`"memory"` clobber). Because the asm is `volatile`, you must not optimize away the variable or this operation.”


---

### 6.5. `DoNotOptimize()` in practice

In practice, `DoNotOptimize()` accepts any object type; the idea is the same:

```cpp
template <typename T>
inline __attribute__((always_inline)) void DoNotOptimize(T const &value)
{
    asm volatile("" : : "r,m"(value) : "memory");
}

template <typename T>
inline __attribute__((always_inline)) void DoNotOptimize(T &value)
{
    asm volatile("" : "+m,r"(value) : : "memory");
}

template <typename T>
inline __attribute__((always_inline)) void DoNotOptimize(T &&value)
{
    asm volatile("" : "+m,r"(value) : : "memory");
}
```

## 7. Example Case
Referring back to the example case, some behaviours can be observed.

### 7.1. Example Source Code
```cpp
void SAXPY(const float a, const float* x, const float* y)
{
    float yout;
    HOTLOOPSTART;
    for (int i = 0; i < N; ++i)
    {
        yout = a * x[i] + y[i];
    }
    HOTLOOPEND;
}

void SAXPY_DONOTOPTIMIZE(const float a, const float* x, const float* y)
{
    float yout;
    HOTLOOPSTART;
    for (int i = 0; i < N; ++i)
    {
        yout = a * x[i] + y[i];
        comppare::DoNotOptimize(yout);
    }
    HOTLOOPEND;
}
```

### 7.2. "Performance" difference
```bash
Implementation             Func µs/Iter        ROI µs/Iter       Ovhd µs/Iter
SAXPY                               0.00                0.00                0.00
SAXPY_DONOTOPTIMIZE                57.14               28.60               28.54
```
From the discussion above, the performance difference is actually due to `SAXPY` being an empty function, while `SAXPY_DONOTOPTIMIZE` actually computes the operation.

### 7.3. Compiler Warning
The build step also compiles `saxpy.cpp` and `saxpy_DoNotOptimize.cpp` into their respected assembly codes in `build/asm/`.

---

There is a compiler warning for `saxpy.cpp`:
```bash
/path/to/ComPPare/examples/advanced_demo/4-DoNotOptimize/saxpy.cpp: In function ‘void SAXPY(float, const float*, const float*)’:
/path/to/ComPPare/examples/advanced_demo/4-DoNotOptimize/saxpy.cpp:5:11: warning: variable ‘yout’ set but not used [-Wunused-but-set-variable]
    5 |     float yout;
      |           ^~~~
```
Which shows that the compiler identifies `yout` as a variable that is not needed, thus optimized away.

---

For `saxpy_DoNotOptimize.cpp`, there is no such warning, which shows that the compiler is being tricked by `DoNotOptimize()` that `yout` is being used and thus not optimized away.

### 7.4. Inspection of the full asm code
`build/asm/DoNotOptimize_demo.s` is the full asm code of the benchmark

#### 7.4.1. On AArch64 (Apple M2)
To look for SAXPY operations, based on findings in [Just don't Optimize -- ARM FMADD](#saxpyappleM2isa), we can find whether `fmadd` exists by:

```bash
grep -rnw fmadd build/asm/DoNotOptimize_demo.s 
```

**Results in:**
```bash
build/asm/DoNotOptimize_demo.s:86:      fmadd   s0, s8, s0, s1
build/asm/DoNotOptimize_demo.s:116:     fmadd   s0, s8, s0, s1
```

It resulted in 2 `fmadd`, this is due to extra code being "injected" via the ComPPare framework. The first `fmadd` at line 86 is the warmup runs, while line 116 is the actual benchmark runs.

It can be further validated when inspecting the function SAXPY_DONOTOPTIMIZE in asm

<details>
  <summary> AArch64 Assembly of <code>SAXPY_DONOTOPTIMIZE()</code> function within the benchmark </summary>
<pre><code class="language-asm">
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">45</span> __Z19SAXPY_DONOTOPTIMIZEfPKfS0_:        ; @_Z19SAXPY_DONOTOPTIMIZEfPKfS0_
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">46</span>      .cfi_startproc
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">47</span> ; %bb.0:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">48</span>      sub     sp, sp, #80
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">49</span>      .cfi_def_cfa_offset 80
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">50</span>      stp     d9, d8, [sp, #16]               ; 16-byte Folded Spill
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">51</span>      stp     x22, x21, [sp, #32]             ; 16-byte Folded Spill
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">52</span>      stp     x20, x19, [sp, #48]             ; 16-byte Folded Spill
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">53</span>      stp     x29, x30, [sp, #64]             ; 16-byte Folded Spill
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">54</span>      add     x29, sp, #64
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">55</span>      .cfi_def_cfa w29, 16
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">56</span>      .cfi_offset w30, -8
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">57</span>      .cfi_offset w29, -16
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">58</span>      .cfi_offset w19, -24
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">59</span>      .cfi_offset w20, -32
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">60</span>      .cfi_offset w21, -40
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">61</span>      .cfi_offset w22, -48
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">62</span>      .cfi_offset b8, -56
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">63</span>      .cfi_offset b9, -64
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">64</span>      mov     x19, x1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">65</span>      mov     x20, x0
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">66</span>      fmov    s8, s0
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">67</span> Lloh2:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">68</span>      adrp    x22, __ZZN8comppare6config8instanceEvE4inst@GOTPAGE
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">69</span> Lloh3:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">70</span>      ldr     x22, [x22, __ZZN8comppare6config8instanceEvE4inst@GOTPAGEOFF]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">71</span>      ldr     x8, [x22, #8]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">72</span>      cbz     x8, LBB1_5
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">73</span> ; %bb.1:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">74</span>      mov     x8, #0
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">75</span>      add     x9, sp, #12
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">76</span> LBB1_2:                                 ; =>This Loop Header: Depth=1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">77</span>                                         ;     Child Loop BB1_3 Depth 2
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">78</span>      mov     x10, x20
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">79</span>      mov     x11, x19
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">80</span>      mov     w12, #34464
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">81</span>      movk    w12, #1, lsl #16
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">82</span> LBB1_3:                                 ;   Parent Loop BB1_2 Depth=1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">83</span>                                         ; =>  This Inner Loop Header: Depth=2
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">84</span>      ldr     s0, [x10], #4
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">85</span>      ldr     s1, [x11], #4
<span style="display:inline-block;width:100%;white-space:pre;background-color:rgba(0, 255, 4, 0.49);border-radius:2px;"><span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">86</span>      fmadd   s0, s8, s0, s1</span>
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">87</span>      str     s0, [sp, #12]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">88</span>      ; InlineAsm Start
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">89</span>      ; InlineAsm End
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">90</span>      subs    x12, x12, #1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">91</span>      b.ne    LBB1_3
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">92</span> ; %bb.4:                                ;   in Loop: Header=BB1_2 Depth=1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">93</span>      add     x8, x8, #1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">94</span>      ldr     x10, [x22, #8]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">95</span>      cmp     x8, x10
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">96</span>      b.lo    LBB1_2
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">97</span> LBB1_5:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">98</span>      str     xzr, [x22]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">99</span>      bl      __ZNSt3__16chrono12steady_clock3nowEv
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">100</span>     mov     x21, x0
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">101</span>     ldr     x8, [x22, #16]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">102</span>     cbz     x8, LBB1_10
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">103</span> ; %bb.6:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">104</span>     mov     x8, #0
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">105</span>     add     x9, sp, #12
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">106</span> LBB1_7:                                 ; =>This Loop Header: Depth=1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">107</span>                                         ;     Child Loop BB1_8 Depth 2
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">108</span>     mov     x10, x20
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">109</span>     mov     x11, x19
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">110</span>     mov     w12, #34464
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">111</span>     movk    w12, #1, lsl #16
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">112</span> LBB1_8:                                 ;   Parent Loop BB1_7 Depth=1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">113</span>                                         ; =>  This Inner Loop Header: Depth=2
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">114</span>     ldr     s0, [x10], #4
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">115</span>     ldr     s1, [x11], #4
<span style="display:inline-block;width:100%;white-space:pre;background-color:rgba(0, 255, 4, 0.49);border-radius:2px;"><span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">116</span>     fmadd   s0, s8, s0, s1</span>
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">117</span>     str     s0, [sp, #12]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">118</span>     ; InlineAsm Start
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">119</span>     ; InlineAsm End
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">120</span>     subs    x12, x12, #1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">121</span>     b.ne    LBB1_8
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">122</span> ; %bb.9:                                ;   in Loop: Header=BB1_7 Depth=1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">123</span>     add     x8, x8, #1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">124</span>     ldr     x10, [x22, #16]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">125</span>     cmp     x8, x10
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">126</span>     b.lo    LBB1_7
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">127</span> LBB1_10:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">128</span>     bl      __ZNSt3__16chrono12steady_clock3nowEv
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">129</span>     ldr     d0, [x22]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">130</span>     fcmp    d0, #0.0
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">131</span>     b.ne    LBB1_12
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">132</span> ; %bb.11:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">133</span>     sub     x8, x0, x21
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">134</span>     scvtf   d0, x8
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">135</span>     mov     x8, #70368744177664
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">136</span>     movk    x8, #16527, lsl #48
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">137</span>     fmov    d1, x8
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">138</span>     fdiv    d0, d0, d1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">139</span>     str     d0, [x22]
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">140</span> LBB1_12:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">141</span>     ldp     x29, x30, [sp, #64]             ; 16-byte Folded Reload
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">142</span>     ldp     x20, x19, [sp, #48]             ; 16-byte Folded Reload
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">143</span>     ldp     x22, x21, [sp, #32]             ; 16-byte Folded Reload
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">144</span>     ldp     d9, d8, [sp, #16]               ; 16-byte Folded Reload
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">145</span>     add     sp, sp, #80
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">146</span>     ret
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">147</span>     .loh AdrpLdrGot Lloh2, Lloh3
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">148</span>     .cfi_endproc
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">149</span>                                         ; -- End function
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">150</span>     .globl  _main                           ; -- Begin function main
</code></pre>
</details>

#### 7.4.2. On x86_64 

To look for SAXPY operations, based on findings in [Just don't Optimize -- x86 MULSS ADDSS](#saxpy_x86_isa), we can find whether `mulss` and `addss` exists by:

```bash
grep -rnw mulss build/asm/DoNotOptimize_demo.s
grep -rnw addss build/asm/DoNotOptimize_demo.s
```

**Results in:**
```bash
build/asm/DoNotOptimize_demo.s:258:    mulss   %xmm0, %xmm1
build/asm/DoNotOptimize_demo.s:282:    mulss   %xmm0, %xmm1
build/asm/DoNotOptimize_demo.s:259:    addss   (%r12,%rax), %xmm1
build/asm/DoNotOptimize_demo.s:283:    addss   (%r12,%rax), %xmm1
```

Again, the doubling of both `mulss` and `addss` is due to warmup runs and benchmark runs.

<details>
  <summary> x86_64 Assembly of <code>SAXPY_DONOTOPTIMIZE()</code> function within the benchmark </summary>
<pre><code class="language-asm">
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">232</span> _Z19SAXPY_DONOTOPTIMIZEfPKfS0_:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">233</span> .LFB7449:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">234</span>     .cfi_startproc
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">235</span>     pushq   %rbp
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">236</span>     .cfi_def_cfa_offset 16
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">237</span>     .cfi_offset 6, -16
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">238</span>     xorl    %ecx, %ecx
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">239</span>     movq    %rsp, %rbp
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">240</span>     .cfi_def_cfa_register 6
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">241</span>     pushq   %r13
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">242</span>     pushq   %r12
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">243</span>     .cfi_offset 13, -24
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">244</span>     .cfi_offset 12, -32
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">245</span>     movq    %rsi, %r12
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">246</span>     pushq   %rbx
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">247</span>     .cfi_offset 3, -40
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">248</span>     movq    %rdi, %rbx
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">249</span>     subq    $24, %rsp
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">250</span>     cmpq    $0, _ZZN8comppare6config8instanceEvE4inst+8(%rip)
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">251</span>     je      .L36
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">252</span> .L35:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">253</span>     xorl    %eax, %eax
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">254</span>     .p2align 4,,10
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">255</span>     .p2align 3
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">256</span> .L37:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">257</span>     movss   (%rbx,%rax), %xmm1
<span style="display:inline-block;width:100%;white-space:pre;background-color:rgba(0, 255, 4, 0.49);border-radius:2px;"><span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">258</span>     mulss   %xmm0, %xmm1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">259</span>     addss   (%r12,%rax), %xmm1</span>
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">260</span>     movd    %xmm1, %edx
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">261</span>     addq    $4, %rax
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">262</span>     cmpq    $400000, %rax
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">263</span>     jne     .L37
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">264</span>     addq    $1, %rcx
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">265</span>     cmpq    _ZZN8comppare6config8instanceEvE4inst+8(%rip), %rcx
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">266</span>     jb      .L35
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">267</span> .L36:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">268</span>     movss   %xmm0, -36(%rbp)
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">269</span>     movq    $0x000000000, _ZZN8comppare6config8instanceEvE4inst(%rip)
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">270</span>     call    _ZNSt6chrono3_V212steady_clock3nowEv
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">271</span>     xorl    %ecx, %ecx
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">272</span>     cmpq    $0, _ZZN8comppare6config8instanceEvE4inst+16(%rip)
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">273</span>     movss   -36(%rbp), %xmm0
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">274</span>     movq    %rax, %r13
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">275</span>     je      .L39
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">276</span> .L38:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">277</span>     xorl    %eax, %eax
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">278</span>     .p2align 4,,10
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">279</span>     .p2align 3
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">280</span> .L40:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">281</span>     movss   (%rbx,%rax), %xmm1
<span style="display:inline-block;width:100%;white-space:pre;background-color:rgba(0, 255, 4, 0.49);border-radius:2px;"><span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">282</span>     mulss   %xmm0, %xmm1
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">283</span>     addss   (%r12,%rax), %xmm1</span>
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">284</span>     movd    %xmm1, %edx
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">285</span>     addq    $4, %rax
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">286</span>     cmpq    $400000, %rax
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">287</span>     jne     .L40
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">288</span>     addq    $1, %rcx
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">289</span>     cmpq    _ZZN8comppare6config8instanceEvE4inst+16(%rip), %rcx
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">290</span>     jb      .L38
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">291</span> .L39:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">292</span>     call    _ZNSt6chrono3_V212steady_clock3nowEv
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">293</span>     pxor    %xmm0, %xmm0
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">294</span>     ucomisd _ZZN8comppare6config8instanceEvE4inst(%rip), %xmm0
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">295</span>     jp      .L34
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">296</span>     jne     .L34
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">297</span>     subq    %r13, %rax
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">298</span>     pxor    %xmm0, %xmm0
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">299</span>     cvtsi2sdq       %rax, %xmm0
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">300</span>     divsd   .LC1(%rip), %xmm0
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">301</span>     movsd   %xmm0, _ZZN8comppare6config8instanceEvE4inst(%rip)
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">302</span> .L34:
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">303</span>     addq    $24, %rsp
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">304</span>     popq    %rbx
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">305</span>     popq    %r12
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">306</span>     popq    %r13
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">307</span>     popq    %rbp
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">308</span>     .cfi_def_cfa 7, 8
<span style="opacity:0.5; font-size:smaller; display:inline-block; width:3em; text-align:right;">309</span>     ret
</code></pre>
</details>