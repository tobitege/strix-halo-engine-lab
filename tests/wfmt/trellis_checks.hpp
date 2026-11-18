#pragma once
#include <cmath>

namespace strix::wfmt {
inline bool coefficient_matches(double got,double reference,double magnitude) {
    return std::isfinite(got) && std::isfinite(reference) && std::isfinite(magnitude) &&
           std::fabs(got-reference)<=1e-3*magnitude+1e-30;
}

struct TrellisCoverage {
    bool glm_available,gemma_available,gemma_only,require_engines;
    const char* error() const {
        if(gemma_only && !gemma_available) return "Gemma-only requested but Gemma is unavailable";
        if(require_engines && (gemma_only || !glm_available || !gemma_available))
            return "--require-engines needs GLM and Gemma built and all cases enabled";
        return nullptr;
    }
    bool run_glm() const { return glm_available && !gemma_only; }
    bool run_gemma() const { return gemma_available; }
};
}  // namespace strix::wfmt
