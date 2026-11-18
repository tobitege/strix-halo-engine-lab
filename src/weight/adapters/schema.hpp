#pragma once
#include "strix/weight/admission.hpp"
namespace strix::weight::detail {
struct RolePattern {
    std::string_view pattern; Role role; bool standalone=false; bool suffix=false;
    bool matches(std::string_view name) const {
        return suffix ? name.ends_with(pattern) : name.find(pattern)!=std::string_view::npos;
    }
};
struct FamilySchema {
    std::string_view family; std::span<const FormatId> linear, plain;
    std::span<const RolePattern> roles;
    bool transformed, native_fp8, channel_quant;
    bool text_only;
    std::span<const FormatId> lookup;
    bool unsigned_trellis=false;
    bool fixed_mul1=false;
    std::string_view text_prefix={}, text_head={};
};
Status bind_model(const TensorIndex&,const FamilySchema&,WeightMap&);
}
