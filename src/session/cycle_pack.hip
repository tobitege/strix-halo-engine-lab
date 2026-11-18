#include "strix/session/cycle_pack.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strix::session {
namespace {

constexpr std::uint32_t kPoison = 0xA5A5A5A5U;
constexpr unsigned kGatherThreads = 64;

void check(hipError_t status, const char* what) {
    if (status != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
}

struct GatherArgs {
    const std::uint32_t* source[CyclePackLayout::max_records];
    std::uint32_t offset[CyclePackLayout::max_records];
    std::uint32_t words[CyclePackLayout::max_records];
    std::uint32_t* destination;
};

// One workgroup per record: word-wise copy into the mapped host block.
__global__ void __launch_bounds__(kGatherThreads) cycle_pack_gather_kernel(GatherArgs args) {
    const unsigned r = blockIdx.x;
    const std::uint32_t* source = args.source[r];
    std::uint32_t* destination = args.destination + args.offset[r];
    for (unsigned w = threadIdx.x; w < args.words[r]; w += kGatherThreads) destination[w] = source[w];
}

}  // namespace

struct CyclePack::Impl {
    CyclePackLayout layout{};
    std::array<const void*, CyclePackLayout::max_records> source{};
    std::uint32_t* host = nullptr;
    std::uint32_t* device_view = nullptr;
    bool gathered = false;
    std::uint64_t cycles = 0, records = 0;
};

CyclePack::CyclePack() : impl_(std::make_unique<Impl>()) {
    void* p = nullptr;
    check(hipHostMalloc(&p, CyclePackLayout::max_words * sizeof(std::uint32_t), hipHostMallocMapped | hipHostMallocCoherent),
        "allocate cycle pack host block");
    impl_->host = static_cast<std::uint32_t*>(p);
    void* d = nullptr;
    const auto mapped = hipHostGetDevicePointer(&d, p, 0);
    if (mapped != hipSuccess) {
        (void)hipHostFree(p);
        check(mapped, "map cycle pack host block");
    }
    impl_->device_view = static_cast<std::uint32_t*>(d);
}

CyclePack::~CyclePack() {
    if (impl_ && impl_->host) (void)hipHostFree(impl_->host);
}

void CyclePack::begin() noexcept {
    impl_->layout.reset();
    impl_->gathered = false;
}

const void* CyclePack::add(const void* device_source, std::size_t bytes) {
    auto& i = *impl_;
    if (!device_source) throw std::invalid_argument("cycle pack record needs a device source");
    if (i.gathered) throw std::logic_error("cycle pack record added after the cycle's gather");
    const auto first = i.layout.add(bytes);
    i.source[i.layout.records - 1U] = device_source;
    std::fill(i.host + first, i.host + first + i.layout.size[i.layout.records - 1U], kPoison);
    return i.host + first;
}

void CyclePack::gather(void* stream) {
    auto& i = *impl_;
    if (i.gathered) throw std::logic_error("cycle pack gathered twice in one cycle");
    i.gathered = true;
    if (!i.layout.records) return;
    GatherArgs args{};
    for (std::uint32_t r = 0; r < i.layout.records; ++r) {
        args.source[r] = static_cast<const std::uint32_t*>(i.source[r]);
        args.offset[r] = i.layout.offset[r];
        args.words[r] = i.layout.size[r];
    }
    args.destination = i.device_view;
    hipLaunchKernelGGL(cycle_pack_gather_kernel, dim3(i.layout.records), dim3(kGatherThreads), 0,
        static_cast<hipStream_t>(stream), args);
    check(hipGetLastError(), "launch cycle pack gather");
    ++i.cycles;
    i.records += i.layout.records;
}

void CyclePack::wait(void* stream) {
    if (!impl_->gathered) throw std::logic_error("cycle pack waited before its gather");
    check(hipStreamSynchronize(static_cast<hipStream_t>(stream)), "complete packed serve cycle");
}

std::uint32_t CyclePack::records() const noexcept { return impl_->layout.records; }
std::uint64_t CyclePack::cycles() const noexcept { return impl_->cycles; }
std::uint64_t CyclePack::gathered() const noexcept { return impl_->records; }

}  // namespace strix::session
