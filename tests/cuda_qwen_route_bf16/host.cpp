#include "strix/cuda/qwen_route_bf16.hpp"
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

// Host adapter for the ACTUAL shared storage helper. This checks host
// rounding/layout; it is never evidence of CUDA compilation or execution.
#define __device__
#define __forceinline__ inline
static float __uint_as_float(unsigned b) { return std::bit_cast<float>(b); }
static unsigned __float_as_uint(float f) { return std::bit_cast<unsigned>(f); }
#include "../../src/hip_moe_route_bf16.inc"
#undef __device__
#undef __forceinline__

namespace qr = strix::cuda::qwen_route;
static unsigned long long checks = 0;
static int failures = 0;
static void check(bool ok, const char* message) {
    ++checks;
    if (!ok) { if (failures < 20) std::fprintf(stderr, "FAIL %s\n", message); ++failures; }
}
template<class F> static void refused(F f, const char* message) {
    bool failed = false;
    try { f(); } catch (const std::exception&) { failed = true; }
    check(failed, message);
}
static std::uint16_t oracle(std::uint32_t bits) {
    auto upper = bits >> 16U;
    const auto lower = bits & 65535U;
    if ((bits & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>(upper | 64U);
    if (lower > 32768U || (lower == 32768U && upper % 2U)) ++upper;
    return static_cast<std::uint16_t>(upper);
}
static void rounding() {
    for (std::uint32_t upper = 0; upper < 65536U; ++upper)
        for (const auto lower : {0U,1U,0x7ffeU,0x7fffU,0x8000U,0x8001U,0xfffeU,0xffffU}) {
            const auto b = upper * 65536U + lower;
            check(strix::moe_route_bf16_bits(b) == oracle(b), "every BF16 binade/tie/sign/nonfinite boundary");
        }
    std::uint32_t rng = 22;
    for (unsigned i = 0; i < 1000000U; ++i) {
        rng ^= rng << 13U; rng ^= rng >> 17U; rng ^= rng << 5U;
        check(strix::moe_route_bf16_bits(rng) == oracle(rng), "random RNE reference");
    }
    for (const auto& [b, expected] : std::array<std::pair<std::uint32_t,std::uint32_t>,14>{{
        {0U,0U},{0x80000000U,0x8000U},{1U,0U},{0x80000001U,0x8000U},
        {0x00008000U,0U},{0x00018000U,2U},{0x007fffffU,0x0080U},
        {0x3f808000U,0x3f80U},{0x3f818000U,0x3f82U},{0x7f800000U,0x7f80U},
        {0xff800000U,0xff80U},{0x7f800001U,0x7fc0U},{0xff800001U,0xffc0U},{0x7f7fffffU,0x7f80U}}})
        check(strix::moe_route_bf16_bits(b) == expected, "explicit zero/subnormal/NaN/inf/tie contract");
}
static void layouts() {
    for (const bool bf16 : {false,true}) {
        check(qr::row_bytes(bf16) == (bf16 ? 51200U : 102400U), "storage bytes");
        for (const std::uint32_t n : {1U,2U,3U,4U,5U,6U,7U,8U,9U,15U,16U,17U,63U,64U,65U,1024U,8192U,qr::max_csr_rows}) {
            const auto l = qr::layout(n,bf16);
            check(l.bytes == static_cast<std::uint64_t>(n)*l.stride, "actual allocation accounting");
            qr::validate_view(4096,l.stride,l.bytes,n,bf16);
            refused([&] { qr::validate_view(4096,l.stride,l.bytes-1U,n,bf16); }, "short final row");
            refused([&] { qr::validate_view(4097,l.stride,l.bytes,n,bf16); }, "misaligned base");
            refused([&] { qr::validate_view(4096,l.stride-1U,l.bytes,n,bf16); }, "misaligned stride");
        }
    }
    for (const std::uint32_t n : {1U,2U,8U,9U,16U,17U,65U}) {
        const auto width = qr::row_bytes(true), stride = width + 32U;
        const auto extent = qr::checked_extent(n,stride,width);
        std::vector<std::uint64_t> aligned(static_cast<std::size_t>((extent+64U+7U)/8U),0xa7a7a7a7a7a7a7a7ULL);
        auto* bytes = reinterpret_cast<std::uint8_t*>(aligned.data());
        auto* start = bytes + 16;
        qr::validate_view(reinterpret_cast<std::uintptr_t>(start),stride,extent,n,true);
        std::uint32_t state = 0x1234U;
        for (std::uint32_t row=0; row<n; ++row) {
            auto* base = reinterpret_cast<float*>(start+row*stride);
            for (std::uint64_t index=0; index<qr::elements_per_row; ++index) {
                state=state*1664525U+1013904223U;
                const auto f=std::bit_cast<float>(state);
                const auto stored=moe_route_store<true>(base,index,f);
                const auto expected=static_cast<std::uint32_t>(oracle(state))<<16U;
                check(std::bit_cast<std::uint32_t>(stored)==expected,"actual writer stored value");
                check(std::bit_cast<std::uint32_t>(moe_route_load<true>(base,index))==expected,"actual matching reader");
            }
        }
        for (std::size_t i=0; i<aligned.size()*8U; ++i) {
            const bool data=i>=16U && i-16U<extent && (i-16U)%stride<width;
            if (!data) check(bytes[i]==0xa7U,"prefix/row/tail sentinel preserved");
        }
    }
    std::array<float,4> off{};
    moe_route_store<false>(off.data(),3,std::bit_cast<float>(0x80000000U));
    check(std::bit_cast<std::uint32_t>(moe_route_load<false>(off.data(),3))==0x80000000U,"OFF exact store/load");
    refused([] { qr::layout(0,true); },"zero rows");
    refused([] { qr::layout(qr::max_csr_rows+1U,true); },"CSR rows");
    refused([] { qr::checked_extent(1,0,0); },"zero stride/width");
    refused([] { qr::checked_extent(UINT64_MAX,32,16); },"extent multiplication overflow");
    refused([] { qr::validate_view(UINTPTR_MAX-15U,51200,51200,1,true); },"address overflow");
    refused([] { qr::validate_view(0,51200,51200,1,true); },"null base");
}
static void policy() {
    for (const auto option : {std::optional<bool>{},std::optional<bool>{false},std::optional<bool>{true}})
    for (const char* request : {static_cast<const char*>(nullptr),"","0","0suffix","1","yes"})
    for (const char* baseline : {static_cast<const char*>(nullptr),"0","1","1suffix"})
    for (const char* defaults : {static_cast<const char*>(nullptr),"","0","0suffix","1"})
    for (const char* exact : {static_cast<const char*>(nullptr),"0","1","1suffix"}) {
        const bool wants=option.value_or(request && request[0] && request[0]!='0');
        const bool suppress=(baseline && std::string_view(baseline)=="1") ||
            (defaults && defaults[0]=='0') || (exact && std::string_view(exact)=="1");
        const auto p=qr::resolve(option,request,baseline,defaults,exact);
        check(p.requested==wants && p.enabled==(wants && !suppress),"policy admission/master/OFF matrix");
    }
    std::set<std::uint64_t> identities;
    for (const bool p : {false,true}) for (const bool d : {false,true}) {
        const auto id=qr::identity(p,d);
        check(identities.insert(id.hash).second,"phase pair must not collide");
        check(id.storage.find("fp16")==std::string_view::npos,"CUDA A8 is not HIP FP16");
        check(id.hash!=strix::moe_prefill_storage_identity(true) &&
              id.hash!=strix::moe_prefill_storage_identity(false),"HIP/CUDA identity separation");
    }
    check(qr::hash_part(qr::hash_part(7,"ab"),"c")!=qr::hash_part(qr::hash_part(7,"a"),"bc"),"framed identity");
    for (const auto c : {qr::Composition::graphs,qr::Composition::cohort,qr::Composition::multisegment,
        qr::Composition::mtp_sidecar,qr::Composition::prefix_cache,qr::Composition::untagged_state_import,
        qr::Composition::node_handoff,qr::Composition::overlap,qr::Composition::other_weight_pack}) {
        qr::require_composition(false,c);
        try { qr::require_composition(true,c); check(false,"unqualified composition accepted"); }
        catch (const std::invalid_argument& e) {
            check(std::string_view(e.what()).find(qr::composition_name(c))!=std::string_view::npos,"named decline");
        }
    }
}
static void snapshots() {
    const auto binding=qr::new_engine_binding(), other=qr::new_engine_binding();
    check(binding!=other,"lifetime binding is not a reusable pointer");
    const auto target=qr::identity(true,true).hash;
    const std::array<std::uint64_t,12> good={target,target,binding,binding,7,32,7*27648,27648,1024,1024,20480,20480};
    const auto validate=[](const auto& a) { qr::validate_snapshot(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11]); };
    validate(good);
    for (const auto index : {0U,2U,4U,5U,6U,7U,8U,10U}) {
        auto bad=good;
        if (index==4U || index==5U || index==7U) bad[index]=0;
        else ++bad[index];
        refused([&] { validate(bad); },"snapshot identity/binding/extent refusal");
    }
    auto overflow=good; overflow[4]=UINT64_MAX-1U; overflow[5]=UINT64_MAX;
    refused([&] { validate(overflow); },"snapshot overflow");
}
int main() {
    rounding(); layouts(); policy(); snapshots();
    for (const bool p : {false,true}) for (const bool d : {false,true}) {
        const auto id=qr::identity(p,d);
        std::printf("pair p%d d%d %016llx %s %s %s\n",p,d,
            static_cast<unsigned long long>(id.hash),id.prefill.data(),id.decode.data(),id.storage.data());
    }
    std::printf("route BF16 HOST %llu checks, %d failures\n",checks,failures);
    return failures ? 1 : 0;
}
