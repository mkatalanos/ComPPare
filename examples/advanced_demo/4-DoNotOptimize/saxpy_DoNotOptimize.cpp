#define N 1000000

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