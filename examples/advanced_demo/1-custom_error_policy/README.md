# Advanced Demo 1 — **Custom Error Policy** <!-- omit from toc -->

## Introduction 
This example is used to demonstrate how to make your own Error Policy -- your custom method for comparing outputs of your functions.
At the time of writing, comppare provides error testing ONLY for types:

1. Floating Point Number
2. Integral Number
3. Containers of **Numbers** that is able to **Iterate Forward** (eg. std::vector&lt;int&gt;, std::deque&lt;double&gt; etc) [[see std::ranges::forward_range]](https://en.cppreference.com/w/cpp/ranges/forward_range.html)
4. std::string

Any other types, or custom types require their own Error Policy to compare.

## Quick Start
### Build
```bash
mkdir build 
cd build 
cmake .. 
make 
```
### Run
```bash
./custom_error_policy_demo
```
### Results
```bash
*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
============ ComPPare Framework ============
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

Number of implementations:             3
Warmup iterations:                   100
Benchmark iterations:                100
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

Implementation             Func µs/Iter        ROI µs/Iter       Ovhd µs/Iter       SameBrand?[0]
Toyota Prius                        0.04                0.02                0.02                true                   
Toyota Corolla                      0.04                0.02                0.02                true                   
Porsche 911                         0.04                0.02                0.02               false            <-- FAIL 
```


## Example Case
This example is found in code.

### Custom Data Structure -- `data_structure.hpp`
Here we have a custom struct called `car` which stores the following information:
```cpp
struct car
{
    std::string make;
    std::string model;
    float mileage;
    uint64_t year;
};
```

### Error Policy: Same Make/Brand? -- `custom_policy.hpp`
We want to compare different functions whether they write out the `same make/brand`
for instance, function `car1` writes the make/brand as "Toyota"
```cpp
void car1(car &c)
{
    c.make = "Toyota";
    c.model = "Prius";
    c.mileage = 6006.13f;
    c.year = 2020;
}
```
and function `car2` also writes "Toyota"
```cpp
void car2(car &c)
{
    c.make = "Toyota";
    c.model = "Corolla";
    c.mileage = 101010.1f;
    c.year = 1998;
}
```

Therefore, based on this comparison criteria, `car2()` is the same implementation as `car1()`

## Error Policy Rules
With this [Example](#example-case) in mind, we can start writing the custom Error Policy `IsSameBrand`

To be a valid Error Policy, there are a couple rules to follow:

### 1. How many Metrics -- `metric_count()`
For our case, we only need a `single metric`: is it the `Same Brand?`

Therefore, our first function `metric_count()` would simply return 1

```cpp
static constexpr std::size_t metric_count() { return 1; }
```

> Note: Some types can have a few metrics. For instance, std::vector&lt;double&gt; can be 3 criteria: Total Error, Mean Error, Max Error. 


### 2. What is the Name of the metric(s)?
We only have 1 metric on whether is it the same make, lets name it `SameBrand?`
This is used to print the column name. So any human readable, easily understood name would be perfect!

Therefore, we have our second function `metric_name(size_t i)`, where i is the index of the metrics
```cpp
static std::string_view metric_name(std::size_t i)
{
    if (i == 0)
        return "SameBrand?";

    throw std::out_of_range("Invalid metric index");
}
```

### 3. Way to compute the Error
To compute the error, we need to provide a reference object against a object you want to test 

```cpp
void compute_error(const car &a, const car &b)
{
    same_brand_ = (a.make == b.make);
}
```

### 4. Did it pass?
```cpp
bool is_fail() const
{
    return !same_brand_;
}
```

### 5. Print out the results 

#### Option 1: Returning a String <!-- omit from toc -->

```cpp
std::string metric(std::size_t i) const
{
    if (i == 0)
    {
        return same_brand_ ? "true" : "false";
    }
}
```

> Note: You can return a numerical value too.



#### Option 2: Returning a Wrapper class of a "Streamable" type <!-- omit from toc -->

The only advantage to use the wrapper to return is for colouring of the results. The values are coloured red if it fails.
```cpp
comppare::internal::policy::MetricValue<std::string> metric(std::size_t i) const
{
    if (i == 0)
    {
        return comppare::internal::policy::MetricValue<std::string>(same_brand_ ? "true" : "false", is_fail());
    }
    throw std::out_of_range("Invalid metric index");
}
```
**"Streamable" Type?**
It means the type must have an `operator <<` to perform stream output. [(See cppreference)](https://en.cppreference.com/w/cpp/string/basic_string/operator_ltltgtgt.html)

(aka, can you do `std::cout << X;` ?)

Example:
```cpp
float a = 1.0f;
std::cout << a; // Valid 

std::vector<float> b = {1.0f};
std::cout << b // Invalid -- std::vector does not have operator <<
```

Hence:
```cpp
comppare::internal::policy::MetricValue<float>; // Valid 

comppare::internal::policy::MetricValue<std::vector<float>> // Invalid -- std::vector does not have operator <<
```


### Summary of ErrorPolicy Required Funcions
#### 1. `static size_t metric_count()`
Returns the number of metrics 

#### 2. `static std::string_view metric_name(std::size_t i)`
Returns the Name of Metric based on input index `i`, for `i < metric_count()`.

#### 3. `void compute_error(const T&&a, const T&&b)`
Void function requiring input of 2 objects to compute the error metrics

#### 4. `bool is_fail()`
Returns whether the case has failed or not.

#### 5 -- option 1 `T metric(size_t i) const`
**T must be either a Numerical Value or std::string**
Returns the Value of Metric based on input index `i`, for `i < metric_count()`.

#### 5 -- option 2 `comppare::internal::policy::MetricValue<T> metric(size_t i) const`
**T must be able to print to out stream**
In other words:
```cpp
T a;
std::cout << a; // if valid
comppare::internal::policy::MetricValue<T> // then this is valid 
```

