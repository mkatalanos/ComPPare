/*

Copyright 2025 | Leong Fan FUNG | funglf | stanleyfunglf@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
#pragma once
#include <argp.h>
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include <comppare/internal/config.hpp>

namespace comppare::internal::helper {

static struct argp_option options[] = {
    {"warmup", 'w', "UINT64", 0, "Warmup iterations"},
    {"iter", 'i', "UINT64", 0, "Number of iterations"},
    {"tolerance", 't', "LONGDOUBLE", 0, "Tolerance value"},
    {0}};

struct arguments {
  uint64_t warmup = 0;
  uint64_t iter = 0;
  long double tolerance = 0;

  bool warmup_set = false;
  bool iter_set = false;
  bool tolerance_set = false;
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  arguments *args = static_cast<arguments *>(state->input);

  switch (key) {
  case 'w':
    args->warmup = std::strtoull(arg, nullptr, 10);
    args->warmup_set = true;
    break;
  case 'i':
    args->iter = std::strtoull(arg, nullptr, 10);
    args->iter_set = true;
    break;
  case 't':
    args->tolerance = std::strtold(arg, nullptr);
    args->tolerance_set = true;
    break;

  case ARGP_KEY_ARG:
    return ARGP_ERR_UNKNOWN; // No positional arguments allowed
  case ARGP_KEY_END:
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = {options, parse_opt, "", ""};

static inline void parse_args(int argc, char **argv) {
  arguments args;

  argp_parse(&argp, argc, argv, 0, 0, &args);
  if (args.warmup_set)
    comppare::config::set_warmup_iters(args.warmup);
  if (args.iter_set)
    comppare::config::set_bench_iters(args.iter);
  if (args.tolerance_set)
    comppare::config::set_all_fp_tolerance(args.tolerance);
}
} // namespace comppare::internal::helper
