#include <vector>

#include <comppare/comppare.hpp>

#include "saxpy_cpu.hpp"
#include "init_saxpy.hpp"
#if (HAVE_CUDA)
#include "saxpy_gpu.cuh"
#endif

int main(int argc, char **argv)
{
    // Initialize SAXPY configuration
    SaxpyConfig cfg = init_saxpy(argc, argv);
    comppare::config::set_fp_tolerance(float(1.0f));

    // Define the input and output types for the comparison framework instance
    using ScalarType = float;
    using VectorType = std::vector<float>;

    comppare::
        InputContext<ScalarType, VectorType, VectorType>::
            OutputContext<VectorType>
                compare(cfg.a, cfg.x, cfg.y);

    // Set reference implementation
    compare.set_reference("cpu serial", cpu_std);
    // Add implementations to compare
    compare.add("cpu OpenMP", cpu_omp);
    compare.add("cpu cpp threads", cpu_par);
#if (HAVE_CUDA)
    compare.add("CUDA", gpu_std);
#endif
    // Run the comparison with specified iterations and tolerance
    compare.run(argc, argv);

    return 0;
}