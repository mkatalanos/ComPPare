#pragma once
#include <cmath>
#include <concepts>
#include <ranges>
#include <type_traits>
#include <limits>

namespace ComPPare::internal
{
    /*
    class to hold error statistics for comparing outputs of different implementations.
    */
    class ErrorStats
    {

    private:
        double max_error_ = 0.0;                              // Maximum error observed
        double total_error_ = 0.0;                            // Sum of all errors
        size_t error_count_ = 0;                              // Count of errors
        size_t element_count_ = 0;                             // Count of elements compared
        size_t max_pos_ = std::numeric_limits<size_t>::max(); // Position of the maximum error

    public:
        /*default constructor*/
        ErrorStats() = default;

        /*copy constructor*/
        ErrorStats(ErrorStats &) = default;
        ErrorStats &operator=(ErrorStats &) = default;

        /*move constructor*/
        ErrorStats(ErrorStats &&) = default;
        ErrorStats &operator=(ErrorStats &&) = default;

        /* Getter */
        [[nodiscard]] double mean() const { return element_count_ ? total_error_ / element_count_ : 0.0; }
        [[nodiscard]] double max() const { return max_error_; }
        [[nodiscard]] double sum() const { return total_error_; }
        [[nodiscard]] size_t error_count() const { return error_count_; }
        [[nodiscard]] size_t element_count() const { return element_count_; }
        [[nodiscard]] size_t max_pos() const { return max_pos_; }

        /*
    Only for floating point type (float, double etc)
    Computes the error statistics between two value
    */
        template <std::floating_point T>
        inline void error_stats(const T &a, const T &b, const double tol)
        {
            double e = std::abs(static_cast<double>(a) - static_cast<double>(b));
            // If the error is below the tolerance, we return an empty ErrorStats
            if (e < tol)
                return;

            // Update the error statistics
            total_error_ = e;
            error_count_ = 1;
            element_count_ = 1;
            max_error_ = e;
            max_pos_ = 0;
        }

        /*
        For integral types (int, long, etc)
        Computes the error statistics between two value
        */
        template <std::integral T>
        inline void error_stats(const T &a, const T &b, const double tol)
        {
            double e = static_cast<double>(std::abs(a - b));
            // If the error is below the tolerance, we return an empty ErrorStats
            if (e < tol)
                return;

            // Update the error statistics
            total_error_ = e;
            error_count_ = 1;
            element_count_ = 1;
            max_error_ = e;
            max_pos_ = 0;
        }

        /*
        For ranges of arithmetic types (integral or floating point)
        Computes the error statistics between two ranges -- std::vector, std::array, etc.
        The ranges must be forward ranges and the elements must be arithmetic type
        */
        template <std::ranges::forward_range R>
        requires std::is_arithmetic_v<std::ranges::range_value_t<R> > // std::ranges::range_value_t<R> returns the type of element within the range --> must be arithmetic
            inline void error_stats(const R &a, const R &b, const double tol)
        {
            // iterator of both ranges
            auto it_a = std::ranges::begin(a);
            auto it_b = std::ranges::begin(b);
            size_t idx = 0;

            // loop through both ranges until one of them ends
            for (; it_a != std::ranges::end(a) && it_b != std::ranges::end(b);
                 ++it_a, ++it_b, ++idx)
            {
                element_count_++; // increment the count of elements compared

                // compute the absolute error between the two elements
                double e = std::abs(double(*it_a - *it_b));

                if (e < tol)
                    continue; // skip if error is below tolerance

                // accumulate the error statistics
                total_error_ += e;
                // increment the count of errors
                error_count_++;

                if (e > max_error_) // if the current error is greater than the maximum observed error
                {
                    // update the maximum error
                    max_error_ = e;
                    // store the index of the vector where the maximum error occurred
                    max_pos_ = idx;
                }
            }
        }

    }; // class ErrorStats
} // namespace ComPPare::internal