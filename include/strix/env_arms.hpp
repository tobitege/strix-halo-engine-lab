#pragma once
#include "strix/env_flags.hpp"
// In-process A/B arms (shared host layer, every rank app can bind it): one process loads the weights once and runs
// several switch sets ("arms") in a strict ABAB order, so a bundle ticket measures all arms of an ABAB without one
// reload per arm. Spec: "LABEL:K=V+K=V;LABEL2:-" ('/' also separates arms: shell-safe; "-" = the launch environment). Before each arm every key named by
// any arm is restored to its launch value, then that arm's pairs are set. Only switches read per call (getenv /
// envcfg, not cached at startup) are valid arm keys; both ranks of a pair must get the same spec (same CLI).
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::env_arms {

// Arms generation: apply() / restore() bump it. A switch cached for the process lifetime ("resolve once", so every
// reader in a prefill agrees) holds its value in Resolved<T>: resolved on first use and again only after an arm changed
// the environment (between prefills, on the thread that runs them). Without arms it is the plain once-per-process value.
inline std::atomic<unsigned>& generation_ref() {
    static std::atomic<unsigned> g{0};
    return g;
}
inline unsigned generation() { return generation_ref().load(std::memory_order_acquire); }

template <class T> class Resolved {
public:
    template <class F> T get(F&& resolve) {
        const unsigned g = generation();
        if (gen_.load(std::memory_order_acquire) == g) return value_.load(std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(m_);
        if (gen_.load(std::memory_order_relaxed) != g) {
            value_.store(resolve(), std::memory_order_relaxed);
            gen_.store(g, std::memory_order_release);
        }
        return value_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<unsigned> gen_{~0U};
    std::atomic<T> value_{};
    std::mutex m_;
};

struct Arm {
    std::string label;
    std::vector<std::pair<std::string, std::string>> kv;
};

struct Arms {
    std::vector<Arm> arms;
    std::vector<std::pair<std::string, std::optional<std::string>>> launch;  // every key named by any arm
};

inline void set_env(const std::string& k, const std::optional<std::string>& v) {
#ifdef _WIN32
    _putenv_s(k.c_str(), v ? v->c_str() : "");
#else
    if (v) setenv(k.c_str(), v->c_str(), 1);
    else unsetenv(k.c_str());
#endif
}

inline Arms parse(const std::string& spec) {
    Arms out;
    std::size_t a = 0;
    while (a < spec.size()) {
        const std::size_t b = std::min(spec.find_first_of(";/", a), spec.size());
        const std::string item = spec.substr(a, b - a);
        const std::size_t c = item.find(':');
        if (c == std::string::npos || c == 0) throw std::invalid_argument("arms: LABEL:K=V+K=V;LABEL:- expected, got '" + item + "'");
        Arm arm{item.substr(0, c), {}};
        const std::string body = item.substr(c + 1);
        if (body != "-") {
            std::size_t p = 0;
            while (p <= body.size()) {
                const std::size_t q = std::min(body.find('+', p), body.size());
                const std::string kv = body.substr(p, q - p);
                const std::size_t e = kv.find('=');
                if (e == std::string::npos || e == 0) throw std::invalid_argument("arms: K=V expected, got '" + kv + "'");
                arm.kv.emplace_back(kv.substr(0, e), kv.substr(e + 1));
                p = q + 1;
            }
        }
        for (const auto& [k, v] : arm.kv) {
            bool seen = false;
            for (const auto& l : out.launch) seen = seen || l.first == k;
            if (!seen) {
                const char* raw = envcfg::text(k.c_str());
                out.launch.emplace_back(k, raw ? std::optional<std::string>(raw) : std::nullopt);
            }
        }
        out.arms.push_back(std::move(arm));
        a = b + 1;
    }
    if (out.arms.empty()) throw std::invalid_argument("arms: empty spec");
    return out;
}

inline void apply(const Arms& all, const Arm& arm) {
    if (all.launch.empty() && arm.kv.empty()) return;
    for (const auto& [k, v] : all.launch) set_env(k, v);
    for (const auto& [k, v] : arm.kv) set_env(k, v);
    generation_ref().fetch_add(1, std::memory_order_acq_rel);
}

inline void restore(const Arms& all) {
    if (all.launch.empty()) return;
    for (const auto& [k, v] : all.launch) set_env(k, v);
    generation_ref().fetch_add(1, std::memory_order_acq_rel);
}

}  // namespace strix::env_arms
