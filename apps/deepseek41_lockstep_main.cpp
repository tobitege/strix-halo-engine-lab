#include "strix/device/hip_gate.hpp"
#include "strix/deepseek41_device_forward.hpp"
#include "strix/deepseek41_host_forward.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Host/device lockstep of a trimmed deepseek41 model (layers [0, N) + head): both forwards run the same decode plan
// token by token in one process and every operation's output hash is compared. Exit status 0 only when every hash of
// every step matches (the device logits are then bitwise the host reference's).
//   strix-deepseek41-lockstep MODEL.gguf --layers 4 --prompt-ids 0,671,... [--generate G] [--logits OUT.f32] [--verbose]
// The fed sequence is the prompt followed by the device's greedy tokens; --logits writes the device logits per step.
int main(int argc, char** argv) try {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    std::string model, prompt, logits_path;
    std::uint32_t layers = 4, generate = 8;
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return std::string(argv[++i]);
        };
        if (a == "--layers") layers = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--prompt-ids") prompt = next();
        else if (a == "--generate") generate = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--logits") logits_path = next();
        else if (a == "--verbose") verbose = true;
        else if (model.empty()) model = a;
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (model.empty() || prompt.empty())
        throw std::invalid_argument("usage: strix-deepseek41-lockstep MODEL.gguf --layers N --prompt-ids a,b,c [--generate G] [--logits OUT.f32]");
    std::vector<std::uint32_t> seq;
    std::stringstream ss(prompt);
    for (std::string t; std::getline(ss, t, ',');) seq.push_back(static_cast<std::uint32_t>(std::stoul(t)));
    const std::size_t total = seq.size() + generate;

    const auto t0 = std::chrono::steady_clock::now();
    strix::deepseek41::HostForwardOptions options;
    options.layers = layers;
    strix::deepseek41::HostForward host(model, options);
    strix::deepseek41::DeviceForward device(model, options);
    std::printf("plan: layers [0, %u), %zu steps; device weights %.2f GiB; host stand-ins:", layers, host.plan().steps.size(),
                static_cast<double>(device.weight_bytes()) / 1073741824.0);
    for (const auto op : strix::deepseek41::DeviceForward::host_ops()) std::printf(" %s", std::string(strix::deepseek41::to_string(op)).c_str());
    if (strix::deepseek41::DeviceForward::host_ops().empty()) std::printf(" none");
    std::printf("\n");
    std::ofstream logits_out;
    if (!logits_path.empty()) logits_out.open(logits_path, std::ios::binary);
    std::size_t compared = 0, mismatched_steps = 0;
    for (std::size_t pos = 0; pos < seq.size(); ++pos) {
        host.step(seq[pos]);
        const auto& logits = device.step(seq[pos]);
        const auto& a = host.last_step_hashes();
        const auto& b = device.last_step_hashes();
        if (a.size() != b.size()) throw std::logic_error("host and device hashed a different number of outputs");
        std::size_t bad = 0;
        std::string first;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a[i].first != b[i].first) throw std::logic_error("label mismatch: " + a[i].first + " vs " + b[i].first);
            ++compared;
            if (a[i].second != b[i].second) {
                if (bad++ == 0) first = a[i].first;
                if (verbose) std::printf("  DIFF %s host %016llx device %016llx\n", a[i].first.c_str(), static_cast<unsigned long long>(a[i].second),
                                         static_cast<unsigned long long>(b[i].second));
            }
        }
        if (bad != 0) ++mismatched_steps;
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("pos %zu fed %u host %u device %u ops %zu %s%s (%.1f s)\n", pos, seq[pos], host.next_token(), device.next_token(), a.size(),
                    bad == 0 ? "IDENTICAL" : "DIFFER first ", bad == 0 ? "" : first.c_str(), s);
        if (logits_out) logits_out.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * 4U));
        if (pos + 1U == seq.size() && seq.size() < total) seq.push_back(device.next_token());
    }
    std::printf("%s: %zu op outputs compared over %zu steps, %zu steps differ\n", mismatched_steps == 0 ? "PASS" : "FAIL", compared, seq.size(),
                mismatched_steps);
    return mismatched_steps == 0 ? 0 : 1;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
