#pragma once
// A preparation stage on its own thread while the caller does other work, shared by the H3 image and clip runners:
// the text encoder alongside the DiT residency, the VAE load during sampling, the USB4 pairing from program entry.
// The job does the same work in the same order as the serial form (only the wall time overlaps), so the outputs do not
// change. get() on the returned future waits for the job and rethrows its exception; the future's destructor waits too,
// so a job never outlives the scope that started it.
#include <chrono>
#include <future>
#include <type_traits>
#include <utility>

namespace strix::h3 {

// The job's result and its own wall time (from its start on the worker to its end).
template <class T>
struct Timed {
    T value;
    double seconds = 0;
};
template <>
struct Timed<void> {
    double seconds = 0;
};

template <class F>
auto start_alongside(F&& f) {
    using R = std::invoke_result_t<std::decay_t<F>&>;
    return std::async(std::launch::async, [fn = std::forward<F>(f)]() mutable {
        const auto t0 = std::chrono::steady_clock::now();
        const auto since = [t0] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
        if constexpr (std::is_void_v<R>) {
            fn();
            return Timed<void>{since()};
        } else {
            R v = fn();
            return Timed<R>{std::move(v), since()};
        }
    });
}

}  // namespace strix::h3
