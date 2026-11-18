#pragma once
#include "source.hpp"
#include "gsq_group_codec.hpp"
#include "gsq_f16_codec.hpp"
#include <type_traits>

namespace strix::weight::hip {
// Wire-format tags carry storage identity only; arithmetic is a separate policy.
template<std::uint32_t WireType> struct GsqTag {
    static constexpr std::uint32_t wire_type=WireType;
};
using Iq3STag=GsqTag<::strix::gsq_a8::type_iq3_s>;
using Iq3XxsTag=GsqTag<::strix::gsq_a8::type_iq3_xxs>;
using Iq2STag=GsqTag<::strix::gsq_a8::type_iq2_s>;
using Iq4NlTag=GsqTag<::strix::gsq_a8::type_iq4_nl>;
using Iq4XsTag=GsqTag<::strix::gsq_a8::type_iq4_xs>;
using Q20Tag=GsqTag<::strix::gsq_a8::type_q2_0>;
struct GsqA8Arithmetic {};
struct GsqDenseArithmetic {};
struct GsqF16Arithmetic {};

template<std::uint32_t T,class Arch>
struct OperandSource<GsqTag<T>,SignedA8Tag,Arch,GsqA8Arithmetic> {
    static_assert(::strix::gsq_a8::iq_type(T)||T==::strix::gsq_a8::type_q2_0);
    using View=MatrixView;
    using Group=gsq_detail::IqGroup;
    static constexpr unsigned group_k=32,alignment=2,shared_bytes=0;
    static constexpr bool has_input_transform=false,has_output_transform=false;
    static constexpr std::uint32_t wire_type=T;
    static constexpr unsigned block_elements=::strix::gsq_a8::dense_block_elements(T);
    static constexpr unsigned block_bytes=::strix::gsq_a8::dense_block_bytes(T);
    static constexpr std::uint64_t row_bytes(std::uint32_t k) {
        return ::strix::gsq_a8::iq_type(T)?::strix::gsq_a8::row_bytes(T,k):
            ::strix::gsq_a8::dense_row_bytes(T,k);
    }
    __device__ __forceinline__ static Group group(const unsigned char* row,std::uint32_t g) {
        if constexpr(::strix::gsq_a8::iq_type(T))return gsq_detail::iq_group<T>(row,g);
        else return gsq_detail::dense_group<T>(row,g);
    }
    __device__ __forceinline__ static Group signed_group(const View& view,std::uint64_t row,std::uint64_t g) {
        return group(view.blocks+row*view.row_stride_bytes,static_cast<std::uint32_t>(g));
    }
    __device__ __forceinline__ static float apply(const Group& g,const int* codes,float delta) {
        return gsq_detail::iq_apply<T>(g,codes,delta);
    }
    static Status validate(const WeightView&,const LaunchShape&);
};

template<std::uint32_t T,class Arch>
struct OperandSource<GsqTag<T>,SignedA8Tag,Arch,GsqDenseArithmetic> {
    static_assert(::strix::gsq_a8::dense_type(T));
    using View=MatrixView;
    using Group=gsq_detail::IqGroup;
    static constexpr unsigned group_k=32,alignment=4,shared_bytes=0;
    static constexpr bool has_input_transform=false,has_output_transform=false;
    static constexpr std::uint32_t wire_type=T;
    __device__ __forceinline__ static Group group(const unsigned char* row,std::uint32_t g) {
        return gsq_detail::dense_group<T>(row,g);
    }
    __device__ __forceinline__ static Group signed_group(const View& view,std::uint64_t row,std::uint64_t g) {
        return group(view.blocks+row*view.row_stride_bytes,static_cast<std::uint32_t>(g));
    }
    __device__ __forceinline__ static float2 entry(const Group& g,const int (&codes)[8],float delta) {
        return gsq_detail::dense_entry<T>(g,codes,delta);
    }
    // Codec adapter only; no dense registry/admission contract is integrated yet.
};

// The arch owns instruction packing; the source owns wire addressing/decode.
// Legacy named decode structs remain forwarding adapters at the tuned core boundary.
template<std::uint32_t T,class Arch>
struct OperandSource<GsqTag<T>,F16TileTag,Arch,GsqF16Arithmetic> {
    static_assert(std::is_same_v<Arch,GsqGfx1151>, "GSQ F16 codec requires the gfx1151 instruction policy");
    static_assert(::strix::gsq_a8::iq_type(T)||T==::strix::gsq_a8::type_q2_0);
    using View=MatrixView;
    template<unsigned TPC> using Block=gsq_detail::F16Block<T,TPC>;  // gfx1151 instructions (gsq_f16_codec.hpp)
    static constexpr unsigned group_k=32,alignment=4,shared_bytes=Block<1>::table_bytes;
    static constexpr bool has_input_transform=false,has_output_transform=false;
    static constexpr std::uint32_t wire_type=T;
    static constexpr std::uint64_t row_bytes(std::uint32_t k) {
        return ::strix::gsq_a8::row_bytes(T,k);
    }
    static constexpr bool supports_row(std::uint32_t k) {
        constexpr auto be=::strix::gsq_a8::block_elements(T),bb=::strix::gsq_a8::block_bytes(T);
        return k%be==0&&row_bytes(k)%4==0&&(bb!=18||(k/be)%2==0);
    }
    static Status validate(const WeightView&,const LaunchShape&);
};

template<std::uint32_t T,class Arch=Gfx1151>
using GsqSignedSource=OperandSource<GsqTag<T>,SignedA8Tag,Arch,GsqA8Arithmetic>;
template<std::uint32_t T,class Arch=Gfx1151>
using GsqDenseSource=OperandSource<GsqTag<T>,SignedA8Tag,Arch,GsqDenseArithmetic>;
template<std::uint32_t T,class Arch>
using GsqF16Source=OperandSource<GsqTag<T>,F16TileTag,Arch,GsqF16Arithmetic>;
} // namespace strix::weight::hip
