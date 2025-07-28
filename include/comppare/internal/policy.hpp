#pragma once
#include <concepts>
#include <ranges>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <limits>
#include <ostream>
#include <sstream>
#include <variant>
#include <stdexcept>

#include <comppare/internal/concepts.hpp>
#include <comppare/internal/ansi.hpp>

namespace comppare::internal::policy
{
    /*
    MetricValue is a wrapper for a value that can be streamed to an output stream.
    It provides an overloaded operator<< to stream the value or Error Message if the value is invalid or fails.

    T value_ is the value of the metric.
    bool is_fail_ indicates if the metric has failed.
    bool valid_ indicates if the metric is valid. (eg. invalid if size mismatch between 2 vectors)
    std::string_view err_msg_ is an error message if the metric is invalid. (eg. outputs "size mismatch" if the size of 2 vectors is different)

    Note: This could be replaced with std::optional<T, string> in C++23
    */
    template <comppare::internal::concepts::Streamable T>
    class MetricValue
    {
        T value_;
        bool is_fail_{false};

        bool valid_{true};
        std::string_view err_msg_;

    public:
        MetricValue(T v) : value_(v), err_msg_(""), valid_(true), is_fail_(false) {}
        MetricValue(T v, bool is_fail) : value_(v), is_fail_(is_fail), valid_(true), err_msg_("") {}
        MetricValue(T v, bool is_fail, bool valid, std::string_view msg) : value_(v), is_fail_(is_fail), valid_(valid), err_msg_(msg) {}

        // overloaded operator<< to stream the value or error message
        friend std::ostream &
        operator<<(std::ostream &os, MetricValue<T> const &mv)
        {
            std::ios saved(nullptr);
            saved.copyfmt(os);

            std::ostringstream tmp;
            tmp.copyfmt(os);
            if (mv.valid_ && !mv.is_fail_)
                tmp << mv.value_;
            else if (mv.valid_ && mv.is_fail_)
                tmp << comppare::internal::ansi::RED(mv.value_);
            else
                tmp << comppare::internal::ansi::RED(mv.err_msg_);
            std::string body = std::move(tmp).str();

            os.width(0);
            os << body;

            os.copyfmt(saved);
            return os;
        }
    };
    template <typename>
    struct is_metric_value : std::false_type
    {
    };

    template <typename U>
    struct is_metric_value<MetricValue<U>> : std::true_type
    {
    };

    template <typename M>
    inline constexpr bool is_metric_value_v = is_metric_value<std::remove_cv_t<M>>::value;

    template <typename M>
    concept MetricValueSpec = is_metric_value_v<M>;

    /*
    Concept for a valid Error Policy

    It requires:
    - metric_count() to return the number of metrics
    - metric_name(std::size_t i) to return the name of the metric at index i
    - compute_error(const Val &a, const Val &b, double tol) to compute the error
      between two values a and b with a given tolerance tol -- or not
    - metric(std::size_t) to return the value of the metric as MetricValue<T>
    - is_fail() to return true if the error exceeds the tolerance
    */
    template <typename Val, typename EP>
    concept ErrorPolicy = requires
    // static members
    {
        { EP::metric_count() } -> std::convertible_to<std::size_t>;
        { EP::metric_name(std::size_t{}) } -> std::convertible_to<std::string_view>;
    } &&
                          // compute error -- either with or without tolerance
                          (requires(EP ep, const Val &a, const Val &b, double t) { ep.compute_error(a, b, t); } || requires(EP ep, const Val &a, const Val &b) { ep.compute_error(a, b); }) &&
                          // metric() returns value of the metric
                          (requires(EP ep, std::size_t i) {
            { ep.metric(i) } -> MetricValueSpec; } || requires(EP ep, std::size_t i) {
            { ep.metric(i) } -> std::convertible_to<double>; } || requires(EP ep, std::size_t i) {
            { ep.metric(i) } -> std::same_as<std::string>; }) &&
                          // is_fail() -- either with or without tolerance
                          (requires(EP ep, double t) {
            { ep.is_fail(t) } -> std::convertible_to<bool>; } || requires(EP ep) {
            { ep.is_fail() } -> std::convertible_to<bool>; });

    template <class EP, class V, class Tol>
    inline void compute_error(EP &ep, const V &a, const V &b, Tol tol)
    {
        ep.compute_error(a, b, tol);
    }

    template <class EP, class V>
    inline void compute_error(EP &ep, const V &a, const V &b)
    {
        ep.compute_error(a, b);
    }

    template <class EP, class Tol>
    inline bool is_fail(const EP &ep, Tol tol)
    {
        return ep.is_fail(tol);
    }

    template <class EP>
    inline bool is_fail(const EP &ep)
    {
        return ep.is_fail();
    }

    namespace autopolicy
    {
        template <typename T>
        concept SupportedByAutoPolicy =
            comppare::internal::concepts::Arithmetic<T> || 
            comppare::internal::concepts::String<T> || 
            comppare::internal::concepts::RangeOfArithmetic<T>;

        /*
        Error Policy for scalar/numbers
        */
        template <typename T>
            requires comppare::internal::concepts::Arithmetic<T>
        class ArithmeticErrorPolicy
        {
            T error_ = T(0);
            std::string err_msg_;
            bool valid_ = true;

            static constexpr std::array names{"Total|err|"};

        public:
            static constexpr std::size_t metric_count() { return 1; }
            static constexpr std::string_view metric_name(std::size_t) { return names[0]; }

            MetricValue<T> metric(std::size_t) const
            {
                return MetricValue<T>(error_, is_fail(), valid_, err_msg_);
            }

            bool is_fail() const { return valid_ || error_ > comppare::config::fp_tolerance<T>(); }

            void compute_error(const T &a, const T &b)
            {
                if constexpr (!std::is_floating_point_v<T>)
                {
                    if (!std::isfinite(a) || !std::isfinite(b))
                    {
                        error_ = std::numeric_limits<T>::quiet_NaN();
                        err_msg_ = "NAN/INF";
                        valid_ = false;
                        return;
                    }
                }

                T e = std::abs(a - b);
                if (e <= comppare::config::fp_tolerance<T>())
                    return;
                error_ = e;
            }
        };

        /*
        Error Policy for strings
        */
        class StringEqualPolicy
        {
            bool eq_{true};

            static constexpr std::array names{"Equal?"};

        public:
            static constexpr std::size_t metric_count() { return 1; }
            static constexpr std::string_view metric_name(std::size_t) { return names[0]; }

            MetricValue<std::string> metric(std::size_t) const
            {
                return MetricValue<std::string>(eq_ ? "true" : "false", is_fail());
            }

            bool is_fail() const { return !eq_; }

            void compute_error(const std::string &a, const std::string &b) { eq_ = (a == b); }
        };

        /*
        Error Policy for ranges of arithmetic types
        eg. std::vector<int>, std::deque<float>, etc.
        */
        template <typename R>
            requires comppare::internal::concepts::RangeOfArithmetic<R>
        class RangeErrorPolicy
        {
            using T = std::remove_cvref_t<std::ranges::range_value_t<R>>;

            T max_error_ = T(0);
            T total_error_ = T(0);
            std::size_t elem_cnt_ = 0;

            bool valid_ = true;
            std::string err_msg_;

            static constexpr std::array names{"Max|err|", "Mean|err|", "Total|err|"};

            MetricValue<T> get_max() const
            {
                return MetricValue<T>(max_error_, is_fail(), valid_, err_msg_);
            }

            MetricValue<T> get_mean() const
            {
                if (elem_cnt_ && valid_)
                    return MetricValue<T>(total_error_ / static_cast<T>(elem_cnt_), is_fail(), valid_, err_msg_);
                else
                    return MetricValue<T>(T(0), is_fail(), valid_, err_msg_);
            }

            MetricValue<T> get_total() const
            {
                return MetricValue<T>(total_error_, is_fail(), valid_, err_msg_);
            }

        public:
            static constexpr std::size_t metric_count() { return names.size(); }
            static constexpr std::string_view metric_name(std::size_t i) { return names[i]; }

            MetricValue<T> metric(std::size_t i) const
            {
                switch (i)
                {
                case 0:
                    return get_max();
                case 1:
                    return get_mean();
                case 2:
                    return get_total();
                default:
                    throw std::out_of_range("Invalid metric index");
                }
            }

            bool is_fail() const
            {
                if (!valid_)
                    return true;

                if constexpr (std::is_floating_point_v<T>)
                    return max_error_ > comppare::config::fp_tolerance<T>();
                else // integral types
                    return max_error_ > T(0);
            }

            void compute_error(const R &a, const R &b)
            {
                if (std::ranges::size(a) != std::ranges::size(b))
                {
                    valid_ = false;
                    err_msg_ = "Size mismatch";
                    elem_cnt_ = 0;
                    return;
                }

                auto ia = std::ranges::begin(a);
                auto ib = std::ranges::begin(b);
                for (; ia != std::ranges::end(a) && ib != std::ranges::end(b); ++ia, ++ib)
                {
                    if constexpr (std::is_floating_point_v<T>)
                    {
                        if (!std::isfinite(*ia) || !std::isfinite(*ib))
                        {
                            max_error_ = std::numeric_limits<T>::quiet_NaN();
                            total_error_ = std::numeric_limits<T>::quiet_NaN();
                            elem_cnt_ = 0;

                            valid_ = false;
                            err_msg_ = "NAN/INF";
                            return;
                        }
                    }

                    T diff = std::abs(*ia - *ib);
                    if constexpr (std::is_floating_point_v<T>)
                    {
                        if (diff <= comppare::config::fp_tolerance<T>())
                            continue;
                    }
                    total_error_ += diff;
                    max_error_ = std::max(max_error_, diff);
                    ++elem_cnt_;
                }
            }
        };

        /*
        AutoPolicy is a helper to deduce the appropriate error policy
        Currently only supports: scalar types, strings, and ranges of arithmetic types (see above concepts)
        */
        template <typename T>
        struct AutoPolicy;

        template <typename T>
            requires comppare::internal::concepts::Arithmetic<T>
        struct AutoPolicy<T>
        {
            using type = ArithmeticErrorPolicy<std::remove_cvref_t<T>>;
        };

        template <typename T>
            requires comppare::internal::concepts::String<T>
        struct AutoPolicy<T>
        {
            using type = StringEqualPolicy;
        };

        template <typename T>
            requires comppare::internal::concepts::RangeOfArithmetic<T>
        struct AutoPolicy<T>
        {
            using type = RangeErrorPolicy<T>;
        };

        template <typename T>
        using AutoPolicy_t = typename AutoPolicy<T>::type;
    }
}
