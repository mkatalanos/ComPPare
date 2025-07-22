#pragma once
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <string_view>
#include <concepts>

#include "param.hpp"
#include "ErrorStats.hpp"
#include "Gbench.hpp"

namespace ComPPare
{
    namespace bench
    {
        template <class InTup, class OutTup>
        class BenchmarkAdapter
        {
        public:
            virtual ~BenchmarkAdapter() = default;
            virtual void initialize(int& /*argc*/, char** /*argv*/) {}
            virtual void finalize() {}
        };

        template <class InTup, class OutTup>
        class GoogleBenchmarkAdapter final : public BenchmarkAdapter<InTup, OutTup>
        {
            using Self = GoogleBenchmarkAdapter<InTup,OutTup>;
            using Func = std::function<void(const InTup&, OutTup&)>;
            internal::gbench_manager gb_;
        public:
            using handle_type = ::benchmark::internal::Benchmark*;

            GoogleBenchmarkAdapter(const GoogleBenchmarkAdapter&)            = delete;
            GoogleBenchmarkAdapter& operator=(const GoogleBenchmarkAdapter&) = delete;

            static std::shared_ptr<Self> instance()
            {
                static std::shared_ptr<Self> inst{new Self};
                return inst;
            }

            template <class F>
            handle_type register_impl(const std::string& name,
                                    F&&                user_fn,
                                    const InTup&       inputs,
                                    OutTup&            outs)
            {
                return std::apply([&](auto const&... in_vals)
                {
                    return std::apply([&](auto&&... outs_vals)
                    {
                        return gb_.add_gbench(name.c_str(),
                                                std::forward<F>(user_fn),
                                                in_vals..., outs_vals...);
                    }, outs);
                }, inputs);
            }

            void initialize(int& argc, char** argv) override { gb_.initialize(argc, argv); }
            void finalize() override                         { gb_.run();                 }

        private:
            GoogleBenchmarkAdapter() = default;
        };

    } // namespace bench
    
    static void DoNotOptimize(void *p)
    {
        asm volatile("" : : "g"(p) : "memory");
    }

    static void ClobberMemory()
    {
        asm volatile("" ::: "memory");
    }

    /*
    InputContext class template to hold input parameters for the comparison framework.
    */
    template <typename... Inputs>
    class InputContext
    {
    public:
        /*
        OutputContext class template to hold output parameters and functions for the comparison framework.
        */
        template <typename... Outputs>
        class OutputContext
        {
        private:
            // Alias for the user-provided function signature:
            // (const Inputs&..., Outputs&..., size_t iterations, double& roi_us)
            using Func = std::function<void(const Inputs &..., Outputs &...)>;

            // Alias for the error statistics class
            using ErrorStats = ComPPare::internal::ErrorStats;

            // Holds each input and output type in a tuple
            using InTup = std::tuple<Inputs...>;
            using OutTup = std::tuple<Outputs...>;

            // reference to output parameter/data
            using OutPtr = std::shared_ptr<OutTup>;
            using OutVec = std::vector<OutPtr>;

            // Tuple to hold all input parameters/data
            InTup inputs_;
            // Reference output tuple to hold the outputs of the first implementation
            OutVec outputs_;

            std::vector<std::shared_ptr<bench::BenchmarkAdapter<InTup,OutTup>>> adapters_;

                    void register_adapter(const std::shared_ptr<bench::BenchmarkAdapter<InTup,OutTup>>& p)
        {
            if (std::find(adapters_.begin(), adapters_.end(), p) == adapters_.end())
                adapters_.push_back(p);
        }


            struct Impl
            {
                std::string name;
                Func fn;
                
                InTup* inputs_ptr; 
                OutputContext* parent_ctx;

                std::unique_ptr<OutTup> externalbench_output = nullptr; // for external benchmarks

                decltype(auto) gbench()
                {
                    return attach<bench::GoogleBenchmarkAdapter>();
                }

                template <template<class,class> class Adapter>
                requires std::is_base_of_v<bench::BenchmarkAdapter<InTup,OutTup>, Adapter<InTup,OutTup>>
                decltype(auto) attach()
                {
                    using A = Adapter<InTup,OutTup>;
                    auto adp = A::instance();

                    parent_ctx->register_adapter(adp);

                    externalbench_output = std::make_unique<OutTup>();

                    // if constexpr (requires { typename A::handle_type; })
                    // {
                        return adp->register_impl(name, fn, *inputs_ptr, *externalbench_output);
                    // }
                    // else
                    // {
                    //     adp->register_impl(name, fn, *inputs_ptr, *externalbench_output);
                    //     return *this;
                    // }
                }
            };

            // Vector to hold all implementations
            std::vector<Impl> impls_;

            inline OutPtr get_output_by_index_(const size_t idx) const
            {
                if (outputs_.empty())
                    throw std::logic_error("run() has not been executed");
                if (idx >= outputs_.size())
                    throw std::out_of_range("Index out of range for outputs");

                return outputs_[idx];
            }

            inline OutPtr get_output_by_name_(const std::string_view name) const
            {
                if (outputs_.empty())
                    throw std::logic_error("run() has not been executed");
                for (size_t i = 0; i < impls_.size(); ++i)
                {
                    if (impls_[i].name == name)
                        return outputs_[i];
                }
                std::stringstream os;
                os << "Output with name '" << name << "' not found";
                throw std::invalid_argument(os.str());
            }

            void unpack_output_(const OutTup &outtup, Outputs *...outs) const
            {
                std::apply(
                    [&](auto &...outtup_elem)
                    {
                        ((*outs = outtup_elem), ...);
                    },
                    outtup);
            }

        public:
            // Constructor to initialize the OutputContext with inputs
            // This is used to hold and pass the same input arguments/data for all implementations
            // The inputs are perfectly forwarded -- for instance taking ownership when moving
            template <typename... Ins>
            explicit OutputContext(Ins &&...ins)
                : inputs_(std::forward<Ins>(ins)...)
            {
            }

            // copy constructor does NOT copy the reference output
            OutputContext(const OutputContext &other) : inputs_(other.inputs_), impls_(other.impls_) {}
            // copy assignment operator
            OutputContext &operator=(const OutputContext &other)
            {
                if (this != &other)
                {
                    inputs_ = other.inputs_;
                    impls_ = other.impls_;
                }
                return *this;
            }

            // move constructor -- might consider not allowing move semantics for OutputContext
            OutputContext(OutputContext &&other) noexcept
            {
                if (this != &other)
                {
                    inputs_ = std::move(other.inputs_);
                    outputs_ = std::move(other.outputs_);
                    impls_ = std::move(other.impls_);
                }
            }
            // move assignment operator
            OutputContext &operator=(OutputContext &&other) noexcept
            {
                if (this != &other)
                {
                    inputs_ = std::move(other.inputs_);
                    outputs_ = std::move(other.outputs_);
                    impls_ = std::move(other.impls_);
                }
                return *this;
            }

            // Function to set a reference implementation
            template <typename F>
                requires std::invocable<F, const Inputs &..., Outputs &...>
            Impl &set_reference(std::string name, F &&f)
            {
                impls_.insert(impls_.begin(), {std::move(name), Func(std::forward<F>(f)), &inputs_, this});
                return impls_.front();
            }

            // Function to add an implementation to the comparison
            template <typename F>
                requires std::invocable<F, const Inputs &..., Outputs &...>
            Impl &add(std::string name, F &&f)
            {
                impls_.push_back({std::move(name), Func(std::forward<F>(f)), &inputs_, this});
                return impls_.back();
            }

            // // Function to set a reference implementation
            // template <typename F>
            //     requires std::invocable<F, const Inputs &..., Outputs &...>
            // decltype(auto) set_reference(std::string name, F &&f)
            // {
            //     impls_.insert(impls_.begin(), {std::move(name), Func(std::forward<F>(f))});
            //     return *this;
            // }

            // // Function to add an implementation to the comparison
            // template <typename F>
            //     requires std::invocable<F, const Inputs &..., Outputs &...>
            // decltype(auto) add(std::string name, F &&f)
            // {
            //     impls_.push_back({std::move(name), Func(std::forward<F>(f))});
            //     return *this;
            // }

            /*
            Getter for the output results
            */

            // returns a shared pointer to the reference output
            // std::shared_ptr<std::tuple<Outputs...>>
            const OutPtr get_reference_output() const
            {
                return get_output_by_index_(0);
            }

            const OutPtr get_output(const size_t idx) const
            {
                return get_output_by_index_(idx);
            }

            const OutPtr get_output(const std::string_view name) const
            {
                return get_output_by_name_(name);
            }

            // Unpack the outputs into the provided pointers
            void get_reference_output(Outputs *...outs) const
            {
                const auto &outtup = *get_output_by_index_(0);
                unpack_output_(outtup, outs...);
            }

            void get_output(const size_t idx, Outputs *...outs) const
            {
                const auto &outtup = *get_output_by_index_(idx);
                unpack_output_(outtup, outs...);
            }

            void get_output(const std::string_view name, Outputs *...outs) const
            {
                const auto &outtup = *get_output_by_name_(name);
                unpack_output_(outtup, outs...);
            }

            /*
            Runs the comparison for all added implementations.
            Arguments:
            iterations -- the number of iteratins inside each custom function
            tol -- the tolerance for error comparison
            warmup -- if true, runs a warmup iteration before the actual timing
            */
            void run(int &argc,
                     char **argv,
                     double tol = 1e-6)
            {
                if (impls_.empty())
                {
                    std::cerr << "\n*----------*\nNo implementations added to the ComPPare Framework.\n*----------*\n";
                    return;
                }

                outputs_.reserve(impls_.size()); // reserve space for outputs -- resize and use index works too.

                // Number of output arguments -- sizeof... is used to get the number of elements in a pack
                // https://en.cppreference.com/w/cpp/language/sizeof....html
                constexpr size_t NUM_OUT = sizeof...(Outputs);

                // Print header for the output table
                constexpr int COL_W = 18;
                std::cout << std::left
                          << std::setw(COL_W) << "Name"
                          << std::right
                          << std::setw(COL_W) << "Func µs"
                          << std::setw(COL_W) << "ROI µs"
                          << std::setw(COL_W) << "Ovhd µs";

                for (size_t v = 0; v < NUM_OUT; ++v)
                    std::cout << std::setw(COL_W) << ("Max|err|[" + std::to_string(v) + "]")
                              << std::setw(COL_W) << "(MaxErr-idx)"
                              << std::setw(COL_W) << ("Mean|err|[" + std::to_string(v) + "]")
                              << std::setw(COL_W) << ("Total|err|[" + std::to_string(v) + "]");

                std::cout << std::endl;

                // Main loop to iterate over all implementations
                for (size_t k = 0; k < impls_.size(); ++k)
                {
                    // Get the current implementation
                    auto &impl = impls_[k];

                    OutTup outs;

                    auto t0 = std::chrono::high_resolution_clock::now();
                    /*
                    use std::apply to unpack the inputs and outputs completely to do 1 function call of the implementation
                    this is equivalent to calling:
                    impl.fn(inputs[0], inputs[1], ..., outputs[0], outputs[1], iters, roi_us);
                    */
                    std::apply([&](auto const &...in)
                               { std::apply(
                                     [&](auto &...out)
                                     { impl.fn(in..., out...); },
                                     outs); }, inputs_);
                    auto t1 = std::chrono::high_resolution_clock::now();

                    // Calculate the time taken by the function in microseconds
                    double roi_us = ComPPare::param::get_roi_us();
                    double func_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                    double ovhd_us = func_us - roi_us;

                    // Create an array to hold error statistics for each output
                    std::array<ErrorStats, NUM_OUT> error_per_output{};

                    // calculate error if not reference implementation
                    if (k != 0)
                    {
                        OutPtr ref_out_ptr = outputs_[0];
                        OutTup &ref_out = *ref_out_ptr;
                        /*
                        unpack the reference outputs and current outputs one by one
                        it is like calling:
                        for (size_t v = 0; v < NUM_OUT; ++v)
                        {
                            error_per_output[v] = error_stats(ref_out[v], outs[v], tol);
                        }
                        */
                        std::apply([&](auto &...outR)
                                   { std::apply([&](auto &...outT)
                                                {
                    size_t v = 0;
                    ((error_per_output[v].error_stats(outT, outR, tol), ++v), ...); }, outs); }, ref_out);
                    }

                    // first impl is the reference
                    outputs_.push_back(std::make_shared<OutTup>(std::move(outs)));

                    // Print the results for the current implementation
                    std::cout << std::left << std::setw(COL_W) << impl.name
                              << std::right << std::setw(COL_W) << std::fixed << std::setprecision(2) << func_us
                              << std::setw(COL_W) << roi_us
                              << std::setw(COL_W) << ovhd_us;

                    // per-output print error
                    for (size_t v = 0; v < NUM_OUT; ++v)
                    {
                        const auto &es = error_per_output[v];
                        double maxerr = (k == 0) ? 0.0 : es.max();
                        double meanerr = (k == 0) ? 0.0 : es.mean();
                        double sumerr = (k == 0) ? 0.0 : es.sum();

                        if (k && maxerr > tol) // fail threshold
                            std::cout
                                << std::setw(COL_W) << std::scientific << maxerr
                                << std::setw(COL_W) << es.max_pos(); // print max error and location (index)
                        else
                            std::cout
                                << std::setw(COL_W) << std::scientific << 0.0
                                << std::setw(COL_W) << "—";

                        // print mean and total
                        std::cout
                            << std::setw(COL_W) << std::scientific << meanerr
                            << std::setw(COL_W) << std::scientific << sumerr;
                    }
                    bool fail = false;
                    for (const auto &es : error_per_output)
                        if (es.max() > tol)
                        {
                            fail = true;
                            break;
                        }

                    if (k && fail)
                        std::cout << "  <-- FAIL";

                    std::cout << '\n';
                } /* for impls */

                // bool any_gb = false;
                // for (const auto &impl : impls_)
                //     if (impl.wants_gb)
                //     {
                //         any_gb = true;
                //         break;
                //     }

                // call adapter hooks once each – adapters_ already unique
                for (auto& a : adapters_) a->initialize(argc, argv);
                for (auto& a : adapters_) a->finalize();
            } /* run */
        }; /* OutputContext */
    }; /* InputContext */
} // namespace ComPPare

#define HOTLOOPSTART \
    auto &&hotloop_body = [&]() { /* start of lambda */

#define COMPPARE_HOTLOOPEND                                 \
    /* Warm-up */                                           \
    for (std::size_t i = 0; i < COMPPARE_WARMUP_ITERS; ++i) \
        hotloop_body();                                     \
                                                            \
    /* Timed */                                             \
    auto t0 = std::chrono::steady_clock::now();             \
    for (std::size_t i = 0; i < COMPPARE_BENCH_ITERS; ++i)  \
        hotloop_body();                                     \
    auto t1 = std::chrono::steady_clock::now();             \
                                                            \
    ComPPare::param::roi_us =                               \
        std::chrono::duration<double, std::micro>(t1 - t0).count();

#define GBENCH_HOTLOOPEND                              \
    benchmark::State &st = *ComPPare::param::gb_state; \
    for (auto _ : st)                                  \
    {                                                  \
        hotloop_body();                                \
    }

#define HOTLOOPEND                      \
    }                                   \
    ; /* end of lambda */               \
                                        \
    if (ComPPare::param::inside_gbench) \
    {                                   \
        GBENCH_HOTLOOPEND               \
    }                                   \
    else                                \
    {                                   \
        COMPPARE_HOTLOOPEND             \
    }

#define HOTLOOP(LOOP_BODY) \
    HOTLOOPSTART LOOP_BODY HOTLOOPEND