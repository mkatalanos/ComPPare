#include <vector>
#include <numeric>
#include <random>

#include "ComPPare.hpp"
#include "saxpy_cpu.hpp"
#include "init_saxpy.hpp"
#if (HAVE_CUDA)
#include "saxpy_gpu.cuh"
#endif

int main(int argc, char **argv)
{
    // Initialize SAXPY configuration
    SaxpyConfig cfg = init_saxpy(argc, argv);

    // Define the input and output types for the comparison framework instance
    using ScalarType = float;
    using VectorType = std::vector<float>;

    ComPPare::
        InputContext<ScalarType, VectorType, VectorType>::
            OutputContext<VectorType>
                compare(cfg.a, cfg.x, cfg.y);

    // Set reference implementation
    compare.set_reference("cpu serial", cpu_std);
    // Add implementations to compare
    compare.add("cpu parallel", cpu_par);
#if (HAVE_NVHPC)
    compare.add("gpu std par", gpu_std_par);
#endif
#if (HAVE_CUDA)
    compare.add("gpu kernel", gpu_std);
#endif

    // Run the comparison with specified iterations and tolerance
    compare.run(cfg.iters, cfg.tol, cfg.warmup);

    return 0;
}
