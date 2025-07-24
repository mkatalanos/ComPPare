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

#include <comppare/internal/config.hpp>
#include <comppare/internal/policy.hpp>
#include <comppare/internal/helper.hpp>
#include <comppare/plugin/plugin.hpp>

#ifdef HAVE_GOOGLE_BENCHMARK
#include "comppare/plugin/google_benchmark/google_benchmark.hpp"
#endif

namespace comppare
{
    template <typename Value, typename Policy = void>
    struct spec;

    template <typename Value, typename Policy>
    struct spec<spec<Value, Policy>, void>
    {
        using value_t = Value;
        using policy_t = Policy;
    };

    template <typename Value>
        requires internal::policy::autopolicy::SupportedByAutoPolicy<Value>
    struct spec<Value, void>
    {
        using value_t = Value;
        using policy_t = internal::policy::autopolicy::AutoPolicy_t<Value>;
    };

    template <typename Value, typename Policy>
        requires comppare::internal::policy::ErrorPolicy<Value, Policy>
    struct spec<Value, Policy>
    {
        using value_t = Value;
        using policy_t = Policy;
    };

    template <typename Value, typename Policy>
    using set_policy = spec<Value, Policy>;

    template <typename T>
    concept OutSpec =
        comppare::internal::policy::ErrorPolicy<
            typename spec<T>::value_t,
            typename spec<T>::policy_t>;

    template <typename T>
    inline __attribute__((always_inline)) void DoNotOptimize(T const &value)
    {
        asm volatile("" : : "r,m"(value) : "memory");
    }

    template <typename T>
    inline __attribute__((always_inline)) void DoNotOptimize(T &value)
    {
#if defined(__clang__)
        asm volatile("" : "+r,m"(value) : : "memory");
#else
        asm volatile("" : "+m,r"(value) : : "memory");
#endif
    }

    template <typename T>
    inline __attribute__((always_inline)) void DoNotOptimize(T &&value)
    {
#if defined(__clang__)
        asm volatile("" : "+r,m"(value) : : "memory");
#else
        asm volatile("" : "+m,r"(value) : : "memory");
#endif
    }

    inline __attribute__((always_inline)) void ClobberMemory()
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
        template <OutSpec... Specs>
        class OutputContext
        {
        private:
            template <typename S>
            using val_t = typename spec<S>::value_t;
            template <typename S>
            using pol_t = typename spec<S>::policy_t;

            // using Outputs = typename Specs::value_t;
            // Alias for the user-provided function signature:
            // (const Inputs&..., Outputs&..., size_t iterations, double& roi_us)
            using Func = std::function<void(const Inputs &..., val_t<Specs> &...)>;

            // Holds each input and output type in a tuple
            using InTup = std::tuple<Inputs...>;
            using OutTup = std::tuple<val_t<Specs>...>;
            using PolicyTup = std::tuple<pol_t<Specs>...>;

            // reference to output parameter/data
            using OutPtr = std::shared_ptr<OutTup>;
            using OutVec = std::vector<OutPtr>;

            // Tuple to hold all input parameters/data
            InTup inputs_;
            // Reference output tuple to hold the outputs of the first implementation
            OutVec outputs_;
            PolicyTup policies_ref_;

            // Number of output arguments -- sizeof... is used to get the number of elements in a pack
            // https://en.cppreference.com/w/cpp/language/sizeof....html
            static constexpr size_t NUM_OUT = sizeof...(Specs);

            std::vector<std::shared_ptr<plugin::Plugin<InTup, OutTup>>> plugins_;

            void register_plugin(const std::shared_ptr<plugin::Plugin<InTup, OutTup>> &p)
            {
                if (std::find(plugins_.begin(), plugins_.end(), p) == plugins_.end())
                    plugins_.push_back(p);
            }

            struct Impl
            {
                std::string name;
                Func fn;

                InTup *inputs_ptr;
                OutputContext *parent_ctx;

                std::unique_ptr<OutTup> plugin_output = nullptr; // output for plugin runs

#ifdef HAVE_GOOGLE_BENCHMARK
                decltype(auto) google_benchmark()
                {
                    return attach<plugin::google_benchmark::GoogleBenchmarkPlugin>();
                }
#endif

                template <template <class, class> class Plugin>
                    requires std::is_base_of_v<plugin::Plugin<InTup, OutTup>, Plugin<InTup, OutTup>>
                decltype(auto) attach()
                {
                    auto adp = Plugin<InTup, OutTup>::instance();

                    parent_ctx->register_plugin(adp);

                    plugin_output = std::make_unique<OutTup>();

                    return adp->register_impl(name, fn, *inputs_ptr, *plugin_output);
                }
            };

            // Vector to hold all implementations
            std::vector<Impl> impls_;

            // helpers -----------------------------------------------------------
            template <std::size_t I>
            static constexpr auto spec_metric_count() { return std::tuple_element_t<I, PolicyTup>::metric_count(); }
            template <std::size_t I>
            static constexpr auto spec_metric_name(std::size_t m) { return std::tuple_element_t<I, PolicyTup>::metric_name(m); }

            static constexpr int PRINT_COL_WIDTH = 20;
            template <std::size_t... I>
            void header_row(std::index_sequence<I...>) const
            {
                (([&]()
                  { 
                    for(std::size_t m=0;m<spec_metric_count<I>();++m) 
                    {
                        std::cout<<std::setw(PRINT_COL_WIDTH)<< (std::string(spec_metric_name<I>(m))+"["+std::to_string(I)+"]");
                    } }()),
                 ...);
            }
            template <std::size_t... I>
            void compute_errors(PolicyTup &errs, const OutTup &test, const OutTup &ref, double tol, std::index_sequence<I...>)
            {
                (comppare::internal::policy::compute_error(std::get<I>(errs), std::get<I>(test), std::get<I>(ref), tol), ...);
            }
            template <std::size_t... I>
            bool any_fail(const PolicyTup &errs, double tol, std::index_sequence<I...>) const
            {
                bool fail = false;
                ((fail |= comppare::internal::policy::is_fail(std::get<I>(errs), tol)), ...);
                return fail;
            }
            template <std::size_t... I>
            void print_metrics(const PolicyTup &errs, std::index_sequence<I...>) const
            {
                (([&]()
                  {
                      for (std::size_t m = 0; m < spec_metric_count<I>(); ++m)
                      {
                          std::cout << std::setw(PRINT_COL_WIDTH) << std::scientific << std::get<I>(errs).metric(m);
                      } }()),
                 ...);
            }

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

            void unpack_output_(const OutTup &outtup, val_t<Specs> *...outs) const
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
                : inputs_(std::forward<Ins>(ins)...) {}

            OutputContext(const OutputContext &other) = delete;
            OutputContext &operator=(const OutputContext &other) = delete;
            OutputContext(OutputContext &&other) = delete;
            OutputContext &operator=(OutputContext &&other) = delete;

            // Function to set a reference implementation
            template <typename F>
                requires std::invocable<F, const Inputs &..., val_t<Specs> &...>
            Impl &set_reference(std::string name, F &&f)
            {
                impls_.insert(impls_.begin(), {std::move(name), Func(std::forward<F>(f)), &inputs_, this});
                return impls_.front();
            }

            // Function to add an implementation to the comparison
            template <typename F>
                requires std::invocable<F, const Inputs &..., val_t<Specs> &...>
            Impl &add(std::string name, F &&f)
            {
                impls_.push_back({std::move(name), Func(std::forward<F>(f)), &inputs_, this});
                return impls_.back();
            }

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
            void get_reference_output(val_t<Specs> *...outs) const
            {
                const auto &outtup = *get_output_by_index_(0);
                unpack_output_(outtup, outs...);
            }

            void get_output(const size_t idx, val_t<Specs> *...outs) const
            {
                const auto &outtup = *get_output_by_index_(idx);
                unpack_output_(outtup, outs...);
            }

            void get_output(const std::string_view name, val_t<Specs> *...outs) const
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
                comppare::internal::helper::parse_args(argc, argv);
                // print number of warmup and iterations
                std::cout << "Warmup iterations: " << comppare::config::warmup_iters() << "\n"
                          << "Benchmark iterations: " << comppare::config::bench_iters() << "\n"
                          << "Tolerance: " << tol << "\n";

                if (impls_.empty())
                {
                    std::cerr << "\n*----------*\nNo implementations added to the ComPPare Framework.\n*----------*\n";
                    return;
                }

                outputs_.reserve(impls_.size()); // reserve space for outputs -- resize and use index works too.

                // Print header for the output table
                std::cout << std::left
                          << std::setw(PRINT_COL_WIDTH) << "Name"
                          << std::right
                          << std::setw(PRINT_COL_WIDTH) << "Func µs"
                          << std::setw(PRINT_COL_WIDTH) << "ROI µs"
                          << std::setw(PRINT_COL_WIDTH) << "Ovhd µs";

                header_row(std::make_index_sequence<NUM_OUT>{});

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
                    double roi_us = comppare::config::get_roi_us();
                    double func_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                    double ovhd_us = func_us - roi_us;

                    PolicyTup errs{};
                    if (k)
                    {
                        compute_errors(errs, outs, *outputs_[0], tol, std::make_index_sequence<NUM_OUT>{});
                    }
                    outputs_.push_back(std::make_shared<OutTup>(std::move(outs)));
                    // print row
                    std::cout << std::left << std::setw(PRINT_COL_WIDTH) << impl.name << std::right << std::setw(PRINT_COL_WIDTH) << std::fixed << std::setprecision(2) << func_us << std::setw(PRINT_COL_WIDTH) << roi_us << std::setw(PRINT_COL_WIDTH) << ovhd_us;
                    print_metrics(errs, std::make_index_sequence<NUM_OUT>{});
                    if (k && any_fail(errs, tol, std::make_index_sequence<NUM_OUT>{}))
                        std::cout << "  <-- FAIL";
                    std::cout << '\n';

                } /* for impls */

                comppare::current_state::set_using_plugin(true);
                for (auto &a : plugins_)
                {
                    a->initialize(argc, argv);
                    a->run();
                }
                comppare::current_state::set_using_plugin(false);
            } /* run */
        }; /* OutputContext */
    }; /* InputContext */
} // namespace comppare

#define HOTLOOPSTART \
    auto &&hotloop_body = [&]() { /* start of lambda */

#define COMPPARE_HOTLOOPEND                                            \
    /* Warm-up */                                                      \
    for (std::size_t i = 0; i < comppare::config::warmup_iters(); ++i) \
        hotloop_body();                                                \
                                                                       \
    /* Timed */                                                        \
    auto t0 = comppare::config::clock_t::now();                        \
    for (std::size_t i = 0; i < comppare::config::bench_iters(); ++i)  \
        hotloop_body();                                                \
    auto t1 = comppare::config::clock_t::now();                        \
                                                                       \
    comppare::config::set_roi_us(t0, t1);

#ifdef PLUGIN_HOTLOOPEND
#define HOTLOOPEND                               \
    }                                            \
    ; /* end lambda */                           \
                                                 \
    if (comppare::current_state::using_plugin()) \
    {                                            \
        PLUGIN_HOTLOOPEND;                       \
    }                                            \
    else                                         \
    {                                            \
        COMPPARE_HOTLOOPEND;                     \
    }
#else
#define HOTLOOPEND     \
    }                  \
    ; /* end lambda */ \
                       \
    COMPPARE_HOTLOOPEND;
#endif

#define HOTLOOP(LOOP_BODY) \
    HOTLOOPSTART LOOP_BODY HOTLOOPEND