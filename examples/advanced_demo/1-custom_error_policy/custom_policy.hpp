#pragma once

#include <ostream>
#include <sstream>
#include <string>

#include "data_structure.hpp"
#include <comppare/comppare.hpp>

/*
Concept for a valid Error Policy

It requires:
- metric_count() to return the number of metrics
- metric_name(std::size_t i) to return the name of the metric at index i
- compute_error(const Val &a, const Val &b, double tol) to compute the error
  between two values a and b with a given tolerance tol -- or not
- metric(std::size_t) to return the value of the metric
- is_fail() to return true if the error exceeds the tolerance


template <typename Val, typename EP>
concept ErrorPolicy = requires
// static members
{
    { EP::metric_count() } -> std::convertible_to<std::size_t>;
    { EP::metric_name(std::size_t{}) } -> std::convertible_to<std::string_view>;
} &&
// compute error -- either with or without tolerance
(
    requires(EP ep, const Val &a, const Val &b, double t) {
        ep.compute_error(a, b, t);
    } ||
    requires(EP ep, const Val &a, const Val &b) {
        ep.compute_error(a, b);
    }
) &&
// metric() returns value of the metric
    requires(EP ep, std::size_t i) {
         { ep.metric(i) } -> std::convertible_to<double>;
    } &&
// is_fail() -- either with or without tolerance
(
    requires(EP ep, double t) {
        { ep.is_fail(t) } -> std::convertible_to<bool>;
    } ||
    requires(EP ep) {
        { ep.is_fail() } -> std::convertible_to<bool>;
    }
);

*/

class IsSameBrand
{
private:
    bool same_brand_{true};

public:
    static constexpr std::size_t metric_count() { return 1; }
    static constexpr std::string_view metric_name(std::size_t) { return "SameBrand?"; }

    void compute_error(const car &a, const car &b)
    {
        same_brand_ = (a.make == b.make);
    }

    bool is_fail() const
    {
        return !same_brand_;
    }
#define USE_METRIC_VALUE
#ifdef USE_METRIC_VALUE
    comppare::internal::policy::MetricValue<std::string> metric(std::size_t i) const
    {
        if (i == 0)
        {
            return comppare::internal::policy::MetricValue<std::string>(same_brand_ ? "true" : "false", is_fail());
        }
        throw std::out_of_range("Invalid metric index");
    }
#else
    std::string metric(std::size_t i) const
    {
        if (i == 0)
        {
            return same_brand_ ? "true" : "false";
        }
    }
#endif
};
