#include "max_cpu.hpp"
#include "init_max.hpp"
#include "common.hpp"
#ifdef HAVE_CUDA
#include "max_gpu.cuh"
#endif

#include <comppare/comppare.hpp>

int main(int argc, char **argv)
{
    // Initialize configuration
    MaxConfig cfg = init_max(argc, argv);

    // Define the input and output types for the comparison framework instance
    comppare::
        InputContext<std::span<const float>>::
            OutputContext<float>
                compare(cfg.data);

    // Set reference implementation
    compare.set_reference("cpu serial", cpu_max_serial);
    // Add implementations to compare
    compare.add("cpu omp", cpu_max_omp);
    compare.add("cpu thread", cpu_max_thread);
#ifdef HAVE_CUDA
    compare.add("gpu kernel", gpu_max<max_kernel>);
    compare.add("gpu kernel opt", gpu_max<max_kernel_warpsemantics>);
#endif

    // Run the comparison with specified iterations and tolerance
    compare.run(argc, argv);

    return 0;
}