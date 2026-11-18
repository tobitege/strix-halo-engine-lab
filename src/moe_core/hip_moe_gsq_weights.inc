// GSQ routed-expert weight decoders of the prefill kernels (hip_moe_gsq_f16_prefill.inc, hip_moe_gsq_wmma_prefill.inc).
// Included by hip_moe_q4k_a8_known_rows.hip after down_prefill_f16x, inside its anonymous namespace.
//
// Q2_0 (type 42): 18-byte blocks of 64 weights (FP16 d, 16 code bytes; weight k at bits 2(k%4) of byte k/4), value
// d * float(q - 1), one block per K64 step. Every value is an FP16 value (q - 1 in {-1, 0, 1, 2}), so the FP16 rounding is
// exact except 2d beyond the FP16 range (infinity; g, u or the raw dot is then non-finite and the row fails closed).
// Rows hold an even block count (K % 128 == 0), so an even block starts at a dword and an odd one two bytes after a
// dword: a thread loads the five dwords from the block's dword-aligned start. FP16 conversion: the four values of a block
// (q = 0..3) are formed once per thread and step; their low and high bytes make two dword tables, and one v_perm with
// selector bytes (c0, c0 + 4, c1, c1 + 4) returns the FP16 pair of codes c0, c1 (~2 VALU per weight, no FP32 work).
// The device decoders (Q2_0 raw fetch/tables/halves, IQ group fetch/convert, B operands) live in the HIP weight source home
// (src/hip/weight/gsq_f16_codec.hpp, instruction policy arch/gfx1151_gsq.hpp); these are the incumbent names bound to the
// gfx1151 instructions.
namespace gsq_detail=::strix::weight::hip::gsq_detail;
using gsq_detail::GsqQ20Raw;
using gsq_detail::gsq_q20_fetch;
using gsq_detail::gsq_q20_halves;
template<std::uint32_t T> using GsqGroup=gsq_detail::F16Group<T>;
template<std::uint32_t W,unsigned TPC> using GsqB=gsq_detail::F16Block<W,TPC>;
// IQ decoder self-test (gsq_iq_decode_selftest): rows of `width` weights of type T (2560 for the 256-weight types, 640
// for IQ4_NL), block b with random bytes and FP16 scale bits (b + seed) & 65535; every 32-weight group decoded through
// GsqGroup<T> (LDS grids) against round_f16(hip_ggml::block_element<T>) (bitwise; a NaN reference requires a NaN).
// counts: [0] words tested, [1] words differing, [2] NaN references.
template<std::uint32_t T>
__global__ void gsq_iq_fill_kernel(unsigned char* bytes,unsigned blocks,unsigned seed) {
    const unsigned b=blockIdx.x*blockDim.x+threadIdx.x;
    if(b>=blocks)return;
    constexpr unsigned bb=gsq_a8::block_bytes(T);
    unsigned char* p=bytes+static_cast<std::uint64_t>(b)*bb;
    unsigned h=b*0x9e3779b9U^seed*0x85ebca6bU;
    for(unsigned i=0;i<bb;++i){h^=h<<13U;h^=h>>17U;h^=h<<5U;p[i]=static_cast<unsigned char>(h+(h>>11U)+i);}
    const unsigned d=(b+seed)&65535U;
    p[0]=static_cast<unsigned char>(d);p[1]=static_cast<unsigned char>(d>>8U);
}
template<std::uint32_t T>
__global__ void gsq_iq_decode_kernel(const unsigned char* bytes,unsigned groups,unsigned width,unsigned long long* counts) {
    __shared__ unsigned tables[GsqGroup<T>::table_bytes?GsqGroup<T>::table_bytes/4U:1U];
    GsqGroup<T>::load_tables(tables,threadIdx.x,blockDim.x);
    __syncthreads();
    const unsigned gi=blockIdx.x*blockDim.x+threadIdx.x;
    if(gi>=groups)return;
    const unsigned per_row=width/32U,g=gi%per_row;
    const unsigned char* row=bytes+static_cast<std::uint64_t>(gi/per_row)*gsq_a8::row_bytes(T,width);
    typename GsqGroup<T>::Raw raw;
    GsqGroup<T>::fetch(row,g,raw);
    uint4 out[4];
    GsqGroup<T>::convert(raw,g,tables,out);
    constexpr unsigned be=gsq_a8::block_elements(T),bb=gsq_a8::block_bytes(T);
    const unsigned char* block=row+(g*32U/be)*bb;
    unsigned long long bad=0,nan=0;
    for(unsigned q=0;q<4U;++q){const unsigned w[4]{out[q].x,out[q].y,out[q].z,out[q].w};
        for(unsigned i=0;i<8U;++i) {
            const unsigned k=8U*q+i;
            const _Float16 r=static_cast<_Float16>(hip_ggml::block_element<T>(block,(g*32U)%be+k));
            const unsigned short ref=__builtin_bit_cast(unsigned short,r),got=static_cast<unsigned short>(w[i/2U]>>(16U*(i%2U)));
            const bool rn=(ref&0x7fffU)>0x7c00U,gn=(got&0x7fffU)>0x7c00U;
            nan+=rn;bad+=rn?!gn:got!=ref;
        }
    }
    atomicAdd(counts,32ULL);atomicAdd(counts+1,bad);atomicAdd(counts+2,nan);
}

template<std::uint32_t Type> struct QwenGsqDecodeOps {
    template<unsigned TPC> using Block=GsqB<Type,TPC>;
};
template<std::uint32_t Type> struct QwenGsqWeightDecode : QwenGsqDecodeOps<Type> {};
template<> struct QwenGsqWeightDecode<gsq_a8::type_q2_0>
     : strix::moe_core::GroupedWeightDecode<strix::moe_core::GroupedWeightFormat::q2_0,QwenGsqDecodeOps<gsq_a8::type_q2_0>> {
    static __device__ __forceinline__ GsqQ20Raw fetch_raw(const unsigned char* row,unsigned block) { return gsq_q20_fetch(row,block); }
    template<unsigned TPC> static __device__ __forceinline__ void halves(const GsqQ20Raw& r,unsigned step,unsigned part,uint4 (&out)[8U/TPC]) {
        gsq_q20_halves<TPC>(r,step,part,out);
    }
};
template<> struct QwenGsqWeightDecode<gsq_a8::type_iq2_s>
    : strix::moe_core::GroupedWeightDecode<strix::moe_core::GroupedWeightFormat::iq2_s,QwenGsqDecodeOps<gsq_a8::type_iq2_s>> {};
template<> struct QwenGsqWeightDecode<gsq_a8::type_iq3_xxs>
    : strix::moe_core::GroupedWeightDecode<strix::moe_core::GroupedWeightFormat::iq3_xxs,QwenGsqDecodeOps<gsq_a8::type_iq3_xxs>> {};
template<> struct QwenGsqWeightDecode<gsq_a8::type_iq3_s>
    : strix::moe_core::GroupedWeightDecode<strix::moe_core::GroupedWeightFormat::iq3_s,QwenGsqDecodeOps<gsq_a8::type_iq3_s>> {};
template<> struct QwenGsqWeightDecode<gsq_a8::type_iq4_xs>
    : strix::moe_core::GroupedWeightDecode<strix::moe_core::GroupedWeightFormat::iq4_xs,QwenGsqDecodeOps<gsq_a8::type_iq4_xs>> {};
template<> struct QwenGsqWeightDecode<gsq_a8::type_iq4_nl>
    : strix::moe_core::GroupedWeightDecode<strix::moe_core::GroupedWeightFormat::iq4_nl,QwenGsqDecodeOps<gsq_a8::type_iq4_nl>> {};
template<std::uint32_t Type> using QwenGsqCore=strix::moe_core::GroupedPrefillCore<QwenGsqWeightDecode<Type>>;
template<class Operations> using QwenIq2XxsWeightDecode=strix::moe_core::GroupedWeightDecode<strix::moe_core::GroupedWeightFormat::iq2_xxs,Operations>;
using QwenQ20Core=QwenGsqCore<gsq_a8::type_q2_0>;
