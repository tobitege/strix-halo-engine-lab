#pragma once
// Compatibility API; HIP operand operations live in the backend source home.
#include "../../src/hip/weight/gsq_group_codec.hpp"
namespace strix::gsq_a8 {
using ::strix::weight::hip::gsq_detail::IqGroup;
using ::strix::weight::hip::gsq_detail::pack4;
using ::strix::weight::hip::gsq_detail::sdot;
using ::strix::weight::hip::gsq_detail::iq_group;
using ::strix::weight::hip::gsq_detail::DenseKind;
using ::strix::weight::hip::gsq_detail::dense_kind;
using ::strix::weight::hip::gsq_detail::dense_group;
using ::strix::weight::hip::gsq_detail::dense_entry;
using ::strix::weight::hip::gsq_detail::iq_apply;
}
