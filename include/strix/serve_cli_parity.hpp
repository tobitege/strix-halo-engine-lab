#pragma once
// Server / CLI configuration parity (every family): the HTTP server and the CLI of an engine must end with the same
// effective values for the same request. A family resolves one request through both entry points, hands the two
// key=value configurations to compare() together with the differences serving really requires (each named, with its
// reason), and the gate fails on
//   - an unexplained difference (a key whose value differs, or that only one entry point has), and
//   - a stale allowance (a listed key that no longer differs: the list cannot outlive its reason).
// Host only, no GPU. Used by tests/serve_cli_parity_tests.cpp; tools/serve_cli_ratio.py reports the measured ratio.
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::parity {

using Config = std::map<std::string, std::string>;

struct Allowed {
    std::string_view key;     // exact key, or a prefix ending in '*'
    std::string_view reason;  // why serving requires its own value; never empty
};

struct Finding {
    std::string key, cli, server, reason;
};

struct Report {
    std::vector<Finding> unexplained, explained, stale;
    [[nodiscard]] bool ok() const noexcept { return unexplained.empty() && stale.empty(); }
};

inline constexpr std::string_view kAbsent = "<absent>";

[[nodiscard]] inline bool matches(std::string_view pattern, std::string_view key) noexcept {
    if (!pattern.empty() && pattern.back() == '*') return key.substr(0, pattern.size() - 1U) == pattern.substr(0, pattern.size() - 1U);
    return pattern == key;
}

[[nodiscard]] inline Report compare(const Config& cli, const Config& server, std::span<const Allowed> allowed) {
    Report report;
    std::vector<bool> used(allowed.size(), false);
    const auto visit = [&](const std::string& key, std::string_view a, std::string_view b) {
        if (a == b) return;
        Finding f{key, std::string(a), std::string(b), {}};
        for (std::size_t i = 0; i < allowed.size(); ++i)
            if (matches(allowed[i].key, key) && !allowed[i].reason.empty()) {
                used[i] = true;
                f.reason = std::string(allowed[i].reason);
                report.explained.push_back(std::move(f));
                return;
            }
        report.unexplained.push_back(std::move(f));
    };
    for (const auto& [key, value] : cli) {
        const auto it = server.find(key);
        visit(key, value, it == server.end() ? kAbsent : std::string_view(it->second));
    }
    for (const auto& [key, value] : server)
        if (!cli.contains(key)) visit(key, kAbsent, value);
    for (std::size_t i = 0; i < allowed.size(); ++i)
        if (!used[i]) report.stale.push_back({std::string(allowed[i].key), {}, {}, std::string(allowed[i].reason)});
    return report;
}

// "key=value" (or "key: value") per line; lines without a separator are kept as "line.N".
[[nodiscard]] inline Config from_lines(std::string_view text) {
    Config out;
    std::size_t n = 0;
    while (!text.empty()) {
        const auto end = text.find('\n');
        auto line = text.substr(0, end);
        text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1U);
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.remove_suffix(1U);
        if (line.empty()) continue;
        const auto eq = line.find('=');
        if (eq == std::string_view::npos || eq == 0U) out["line." + std::to_string(n++)] = std::string(line);
        else out[std::string(line.substr(0, eq))] = std::string(line.substr(eq + 1U));
    }
    return out;
}

// Flag list ("--a", "v", "--b") as a configuration: "arg:--a" = "v", "arg:--b" = "" (value = the words up to the next flag).
[[nodiscard]] inline Config from_args(std::span<const std::string> args) {
    Config out;
    std::string key;
    for (const auto& a : args) {
        if (a.rfind("--", 0) == 0) {
            key = "arg:" + a;
            out[key];
        } else if (!key.empty()) {
            auto& v = out[key];
            v += v.empty() ? a : " " + a;
        } else out["arg:positional." + std::to_string(out.size())] = a;
    }
    return out;
}

[[nodiscard]] inline std::string text(std::string_view family, const Report& r) {
    std::string out;
    const auto row = [&](std::string_view tag, const Finding& f) {
        out += std::string(family) + " " + std::string(tag) + " " + f.key;
        if (!f.cli.empty() || !f.server.empty()) out += " cli=" + f.cli + " server=" + f.server;
        if (!f.reason.empty()) out += " (" + f.reason + ")";
        out += "\n";
    };
    for (const auto& f : r.unexplained) row("UNEXPLAINED", f);
    for (const auto& f : r.stale) row("STALE-ALLOWANCE", f);
    for (const auto& f : r.explained) row("serving-only", f);
    return out;
}

}  // namespace strix::parity
