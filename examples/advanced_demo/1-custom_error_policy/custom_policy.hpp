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
    /* Rule 1: metric_count() function to return the number of metrics */
    static constexpr std::size_t metric_count() { return 1; }

    /* Rule 2: metric_name(std::size_t i) function to return the name of the metric at index i */
    static std::string_view metric_name(std::size_t i)
    {
        if (i == 0)
            return "SameBrand?";

        throw std::out_of_range("Invalid metric index");
    }

    /* Rule 3: compute_error(const T&&a, const T&&b) function to compute the error metrics */
    void compute_error(const car &a, const car &b)
    {
        same_brand_ = (a.make == b.make);
    }

    /* Rule 4: is_fail() function to return true if the implementation did not pass */
    bool is_fail() const
    {
        return !same_brand_;
    }


#define USE_METRIC_VALUE
#ifndef USE_METRIC_VALUE
    /* Rule 5 -- option 1: metric(std::size_t) function to return the value of the metric */
    std::string metric(std::size_t i) const
    {
        if (i == 0)
        {
            return same_brand_ ? "true" : "false";
        }
    }
#else
    /* Rule 5 -- option 2: metric(std::size_t) function to return the value of the metric within Wrapper class for aesthetics purpose*/
    comppare::internal::policy::MetricValue<std::string> metric(std::size_t i) const
    {
        if (i == 0)
        {
            return comppare::internal::policy::MetricValue<std::string>(same_brand_ ? "true" : "false", is_fail());
        }
        throw std::out_of_range("Invalid metric index");
    }
#endif
};
