#pragma once
// Host fixture only; these mappings do not qualify inference arithmetic.
#include "../audit565/msvc_math.hpp"
#define __builtin_rintf std::rint
