// Greedy continuations after a prefill of each precision arm (quality test 3 of the approximate prefill paths): two
// ranks, the production prefill (row split, cone) with the arm's precision, its attention state handed to a decode
// BatchForward that shares the weights (as the serve), then G exact greedy steps. One JSON line per (case, arm).
// usage: strix-deepseek41-prefill-continue MODEL manifest.tsv rank G [budget_seconds]
//   manifest rows: name n ids_path arms (comma list, labels of deepseek41_quality_arms.hpp)
#include "strix/device/hip_gate.hpp"
#include "deepseek41_quality_arms.hpp"
#include "strix/deepseek41_batch_forward.hpp"
#include "strix/deepseek41_prefill.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/ds41_expert_tp.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) try {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    const auto start = std::chrono::steady_clock::now();
    if (argc != 5 && argc != 6) throw std::runtime_error("MODEL manifest.tsv rank G [budget_seconds]");
    const unsigned rank = static_cast<unsigned>(std::stoul(argv[3])), gen = static_cast<unsigned>(std::stoul(argv[4]));
    const double budget = argc == 6 ? std::stod(argv[5]) : 0;
    if (rank > 1 || gen == 0) throw std::runtime_error("rank / G");
    struct Row { std::string name; std::vector<std::uint32_t> ids; std::vector<std::string> arms; };
    std::vector<Row> rows;
    std::size_t longest = 0;
    {
        std::ifstream manifest(argv[2]);
        if (!manifest) throw std::runtime_error("manifest");
        for (std::string line; std::getline(manifest, line);) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream f(line);
            Row r;
            std::size_t n = 0;
            std::string path, arms;
            if (!(f >> r.name >> n >> path >> arms)) throw std::runtime_error("manifest row");
            std::ifstream in(path);
            for (std::uint32_t id; in >> id;) r.ids.push_back(id);
            if (r.ids.size() != n) throw std::runtime_error("prompt length " + r.name);
            std::stringstream as(arms);
            for (std::string a; std::getline(as, a, ',');) r.arms.push_back(a);
            longest = std::max(longest, n);
            rows.push_back(std::move(r));
        }
    }
    strix::dist::Usb4AutoOptions u;
    u.rank = rank;
    u.timeout = std::chrono::seconds(30);
    strix::dist::Communicator comm(strix::dist::Group{rank, 2, 1, 0x636f6e74696e7565ULL}, strix::dist::usb4_host_auto(u, nullptr),
                                   std::size_t(32) << 20);
    // as the serve: the decode keeps the expert-parallel tensors and the prefill reads every weight in place
    strix::ds41::expert_tp::set_decode_default(false);
    strix::deepseek41::BatchForwardOptions bo;
    bo.layers = 40;
    bo.max_positions = static_cast<std::uint32_t>(longest + gen + 16U);
    bo.slots = 1U;
    bo.rank = rank;
    bo.world = 2U;
    bo.communicator = &comm;
    bo.tp_attention = true;
    bo.engram_direct_io = true;
    strix::deepseek41::BatchForward fwd(argv[1], bo);
    strix::deepseek41::PrefillOptions o;
    o.rank = rank;
    o.world = 2;
    o.communicator = &comm;
    o.max_positions = static_cast<std::uint32_t>(longest);
    o.shared_weights = fwd.file_form_weights();
    strix::deepseek41::DevicePrefill pre(argv[1], o);
    std::printf("loaded rank=%u overlay=%s seconds=%.1f\n", rank, pre.overlay_description().c_str(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    std::fflush(stdout);
    for (const auto& r : rows)
        for (const auto& label : r.arms) {
            // both ranks stop at the same arm: the larger elapsed time decides
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), peer = 0;
            std::vector<std::byte> out(sizeof elapsed), in(sizeof elapsed);
            std::memcpy(out.data(), &elapsed, sizeof elapsed);
            fwd.exchange_host(out, in);
            std::memcpy(&peer, in.data(), sizeof peer);
            const double need = (r.ids.size() > 4096 ? 30.0 : 12.0) + gen * 0.05;
            if (budget != 0 && std::max(elapsed, peer) + need > budget) {
                std::printf("budget_stop next=%s arm=%s elapsed=%.1f\n", r.name.c_str(), label.c_str(), elapsed);
                return 0;
            }
            if (label.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_") != std::string::npos) throw std::runtime_error("arm label");
            pre.set_precision_for_new_sequence(strix::deepseek41::quality::arm_precision(label));
            const auto res = pre.run(r.ids);
            auto state = pre.export_state();
            state.tokens = r.ids;
            state.next_token = res.next_token;
            fwd.reset(0);
            fwd.import_state(0, state);
            std::vector<std::uint32_t> tokens{res.next_token};
            const auto t0 = std::chrono::steady_clock::now();
            while (tokens.size() < gen) {
                const strix::deepseek41::BatchRow row{0, tokens.back()};
                tokens.push_back(fwd.step({&row, 1})[0]);
            }
            const double dsec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("{\"case\":\"%s\",\"arm\":\"%s\",\"rank\":%u,\"prompt\":%zu,\"prefill_seconds\":%.3f,\"decode_seconds\":%.3f,\"tokens\":[",
                        r.name.c_str(), label.c_str(), rank, r.ids.size(), res.seconds, dsec);
            for (std::size_t j = 0; j < tokens.size(); ++j) std::printf("%s%u", j ? "," : "", tokens[j]);
            std::printf("]}\n");
            std::fflush(stdout);
        }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "prefill-continue: %s\n", e.what());
    return 2;
}
