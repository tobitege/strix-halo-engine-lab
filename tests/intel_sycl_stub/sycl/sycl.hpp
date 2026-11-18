#pragma once
// Host-only fault injection double. Never a substitute for compiling/running oneAPI.
#include <cstddef>
#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <deque>
#include <memory>
#include <cstring>

namespace sycl {
struct exception : std::runtime_error { exception() : std::runtime_error("injected SYCL failure") {} };
using exception_list = std::vector<std::exception_ptr>;
namespace test {
enum class Fault { none, construct, info, device_alloc, shared_alloc, host_alloc, release, copy, submit, wait };
inline Fault fault = Fault::none;
inline bool async_wait = false, async_destroy = false, null_alloc = false;
inline int live = 0, delivered = 0, missing_handler = 0, info_calls = 0;
inline char storage = 0;
inline void fail(Fault at) { if (fault == at) throw exception{}; }
// Optional deferred execution for ordering/retirement tests; fault tests stay inert.
struct Work {
    std::deque<std::function<void()>> pending;
    std::size_t submitted=0,completed=0;
};
inline bool execute=false;
inline std::vector<std::shared_ptr<Work>> queues;
inline void submit(const std::shared_ptr<Work>& work,std::function<void()> action) {
    work->pending.push_back(std::move(action)); ++work->submitted;
}
inline void drain(const std::shared_ptr<Work>& work,std::size_t until) {
    while(work->completed<until) {
        auto action=std::move(work->pending.front()); work->pending.pop_front();
        action(); ++work->completed;
    }
}
}
namespace info {
enum class device_type { gpu };
namespace device {
struct name { using type = std::string; };
struct max_compute_units { using type = std::uint32_t; };
struct global_mem_size { using type = std::uint64_t; };
struct local_mem_size { using type = std::uint64_t; };
}
}
namespace ext::intel::info::device {
struct device_id { using type = std::uint32_t; };
struct uuid { using type = std::array<unsigned char, 16>; };
struct pci_address { using type = std::string; };
}
namespace property::queue { struct in_order {}; }
enum class backend { ext_oneapi_level_zero };
enum class aspect { ext_intel_device_id, ext_intel_device_info_uuid, ext_intel_pci_address };
struct device {
    static std::vector<device> get_devices(info::device_type) { return {device{}}; }
    backend get_backend() const { return backend::ext_oneapi_level_zero; }
    bool has(aspect) const { return true; }
    template<class T> typename T::type get_info() const {
        if (test::fault == test::Fault::info && ++test::info_calls == 3) throw exception{};
        if constexpr (std::is_arithmetic_v<typename T::type>) return static_cast<typename T::type>(0xe200U);
        else return typename T::type{};
    }
};
template<int N> struct range {
    range(std::size_t, std::size_t) {}
};
template<int N> struct id {
    std::size_t operator[](std::size_t) const { return 0; }
};
class queue {
    std::shared_ptr<test::Work> work_=std::make_shared<test::Work>();
    std::function<void(exception_list)> handler_;
    void deliver() {
        ++test::delivered;
        if (!handler_) { ++test::missing_handler; throw exception{}; } // Never terminate the test runner.
        handler_({std::make_exception_ptr(exception{})});
    }
public:
    queue(const device&, property::queue::in_order) { test::fail(test::Fault::construct); ++test::live; test::queues.push_back(work_); }
    template<class H> queue(const device&, H handler, property::queue::in_order)
        : handler_(std::move(handler)) { test::fail(test::Fault::construct); ++test::live; test::queues.push_back(work_); }
    ~queue() noexcept {
        if (test::async_destroy) {
            try { deliver(); } catch (...) { /* recorded by delivered; test checks handler presence */ }
        }
        --test::live;
    }
    void wait_and_throw() {
        test::fail(test::Fault::wait);
        if(test::execute) test::drain(work_,work_->submitted);
        if (test::async_wait) { test::async_wait = false; deliver(); }
    }
    void memcpy(void* dst, const void* src, std::size_t bytes) {
        test::fail(test::Fault::copy);
        if(test::execute) test::submit(work_,[=] { std::memcpy(dst,src,bytes); });
    }
    template<class F> void parallel_for(range<2>, F action) {
        test::fail(test::Fault::submit);
        if(test::execute) test::submit(work_,[=] { action(id<2>{}); });
    }
};
inline void* allocation(test::Fault kind) {
    test::fail(kind);
    return test::null_alloc ? nullptr : &test::storage;
}
inline void* malloc_device(std::size_t, queue&) { return allocation(test::Fault::device_alloc); }
inline void* malloc_shared(std::size_t, queue&) { return allocation(test::Fault::shared_alloc); }
inline void* malloc_host(std::size_t, queue&) { return allocation(test::Fault::host_alloc); }
inline void free(void*, queue&) { test::fail(test::Fault::release); }
} // namespace sycl
