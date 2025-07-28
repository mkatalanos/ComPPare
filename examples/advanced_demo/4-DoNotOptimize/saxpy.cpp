#define N 1000000

void SAXPY(const float a, const float* x, const float* y)
{
    float yout;
    for (int i = 0; i < N; ++i)
    {
        yout = a * x[i] + y[i];
    }
}