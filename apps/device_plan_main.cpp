// strix-device-plan: the device catalogue and the shared memory plan, without a GPU.
//
//   strix-device-plan --list                         catalogue as a table
//   strix-device-plan --toml                         catalogue data file (data/devices/profiles.toml)
//   strix-device-plan --devices FILE [--needs N.toml] [--ram-total-gib G]
//                                                    budget per declared GPU entry; with route needs, the plan
//                                                    (planned shapes `count = N`: per-device plan + prediction)
//   strix-device-plan --profile NAME --needs N.toml [--vram-total-gib V] [--ram-total-gib G]
//
// Output: one JSON object per line (strix-memory-budget-v1 / strix-memory-plan-v1 / strix-shape-plan-v1).
// Exit 0 = every plan fits, 3 = at least one plan refused (the line says the minimum), 2 = bad input.
#include "strix/deploy/memory_plan.hpp"
#include "strix/device/budget.hpp"
#include "strix/device/declaration.hpp"
#include "strix/device/profiles.hpp"

#include <cstdio>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif
#include <fstream>
#include <optional>
#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::invalid_argument("cannot read " + path);
    std::ostringstream os;
    os << in.rdbuf();
    return os.str();
}

void list() {
    std::printf("| profile | arch | memory GiB | GB/s | units | FP32 | FP16 | BF16 | FP8 | FP4 | INT8 | host link | measured |\n"
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n");
    for (const auto& p : strix::device::catalogue()) {
        if (p.name == "custom") continue;
        const auto& k = p.peaks;
        std::printf("| %.*s | %.*s | %.2f | %.0f | %u %.*s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.*s | %.*s |\n",
                    static_cast<int>(p.name.size()), p.name.data(), static_cast<int>(p.arch.size()), p.arch.data(),
                    static_cast<double>(p.memory_bytes) / 1073741824.0, p.memory_gbytes_ps, p.units, static_cast<int>(p.unit_kind.size()),
                    p.unit_kind.data(), k.fp32, k.fp16, k.bf16, k.fp8, k.fp4, k.int8, static_cast<int>(p.host_link.size()),
                    p.host_link.data(), static_cast<int>(p.measured.size()), p.measured.empty() ? "-" : p.measured.data());
    }
}

} // namespace

int main(int argc, char** argv) {
    using namespace strix;
    std::string devices, needs_path, profile;
    device::BudgetOverrides overrides;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            const auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                return argv[++i];
            };
            if (a == "--list") return list(), 0;
            if (a == "--toml") {
#if defined(_WIN32)
                _setmode(_fileno(stdout), _O_BINARY); // the data file has LF line ends on every host
#endif
                return std::fputs(device::catalogue_toml().c_str(), stdout), 0;
            }
            if (a == "--devices") devices = next();
            else if (a == "--needs") needs_path = next();
            else if (a == "--profile") profile = next();
            else if (a == "--vram-total-gib") overrides.vram_total_gib = device::parse_total_gib(a, next());
            else if (a == "--ram-total-gib") overrides.ram_total_gib = device::parse_total_gib(a, next());
            else throw std::invalid_argument("unknown argument " + a + " (see the header of apps/device_plan_main.cpp)");
        }
        if (devices.empty() == profile.empty()) throw std::invalid_argument("give one of --devices FILE or --profile NAME");
        std::optional<deploy::RouteNeeds> needs;
        if (!needs_path.empty()) needs = deploy::parse_route_needs(read_file(needs_path));
        device::Declaration declaration;
        if (!devices.empty()) {
            declaration = device::load_declaration(devices);
        } else {
            if (!device::find_profile(profile))
                throw std::invalid_argument("unknown profile " + profile + "; known: " + device::profile_names());
            device::Descriptor d;
            d.id = profile;
            d.profile = profile;
            d.planned = true;
            declaration.devices.push_back(d);
        }
        const auto* host = device::host_unit(declaration);
        bool refused = false;
        for (const auto& d : declaration.devices) {
            if (d.kind != device::Kind::gpu || d.excluded) continue;
            if (!needs) {
                std::cout << device::resolve_budget(d, host, 0, overrides).json() << "\n";
                continue;
            }
            if (d.count > 1) {
                const auto s = deploy::plan_shape(d, host, *needs);
                refused |= s.per_device.refused;
                std::cout << s.json() << "\n";
            } else {
                // A single device: the plan, and the bandwidth-model prediction beside it.
                const auto s = deploy::plan_shape(d, host, *needs);
                const auto p = deploy::plan_memory(device::resolve_budget(d, host, 0, overrides), *needs);
                refused |= p.refused;
                std::cout << p.json() << "\n";
                if (!overrides.vram_total_gib && !overrides.ram_total_gib) std::cout << s.json() << "\n";
            }
        }
        return refused ? 3 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-device-plan: %s\n", e.what());
        return 2;
    }
}
