#pragma once
// Host fixture compatibility only; no native numerical qualification is inferred.
#include <cmath>
using std::rintf;
#define __builtin_sqrtf std::sqrt
#define __builtin_fmaf std::fma
