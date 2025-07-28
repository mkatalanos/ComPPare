#define N 1000000

// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This implementation is verbatim from Google Benchmark’s benchmark::DoNotOptimize(),
// licensed under Apache 2.0. No changes have been made.
template <typename T>
inline __attribute__((always_inline)) void DoNotOptimize(T &value)
{
#if defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

void SAXPY_DONOTOPTIMIZE(const float a, const float* x, const float* y)
{
    float yout;
    for (int i = 0; i < N; ++i)
    {
        yout = a * x[i] + y[i];
        DoNotOptimize(yout);
    }
}