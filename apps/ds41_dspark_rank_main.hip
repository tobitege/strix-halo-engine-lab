// One rank of the DSpark drafter with the head split by vocabulary over two ranks (two nodes).
//   strix-ds41-dspark-rank TARGET.gguf DRAFT.gguf --rank R (--transport SPEC | --listen PORT | --connect HOST:PORT | --usb4 DEV,DEV,...)
//                          [--vocab SPEC] [--steps N]
// Each rank first drafts alone (world 1, the reference), then resets and drafts the same sequence with
// the head split (world 2); the tokens and confidences must be identical. Inputs are synthetic target
// rows (same seeds on both ranks). Transports: the communicator's USB4 stream backend (--usb4: the
// stream devices of every live cable, in the paired order of the detected topology, never fixed
// numbers) or TCP (debug).

#include "strix/device/hip_gate.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/dist/open_transport.hpp"
#include "strix/ds41_dspark_files.hpp"
#include "strix/ds41_dspark_vocab.hpp"
#include "strix/hip_ds41_dspark.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace strix::ds41::dspark;

std::vector<float> normal_values(std::size_t n, unsigned seed) {
    std::mt19937 random(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> v(n);
    for (auto& x : v) x = normal(random);
    return v;
}

struct Run {
    std::vector<Draft> drafts;
    double head_ms = 0.0, total_ms = 0.0;
};
Run run(DeviceDrafter& d, std::uint32_t steps) {
    const auto seed = normal_values(static_cast<std::size_t>(130) * main_width, 77);
    const auto accepted = normal_values(static_cast<std::size_t>(3) * main_width, 78);
    d.append_hidden(std::span<const float>(seed).first(static_cast<std::size_t>(128) * main_width), 128);
    d.append_hidden(std::span<const float>(seed).subspan(static_cast<std::size_t>(128) * main_width), 2);
    Run out;
    std::uint32_t token = 1000;
    for (std::uint32_t i = 0; i < steps; ++i) {
        DraftTimes t;
        out.drafts.push_back(d.draft(token, false, &t));
        if (i > 0U) { out.head_ms += t.head_ms; out.total_ms += t.total_ms; }
        d.append_hidden(accepted, 3);
        token = out.drafts.back().tokens[3];
    }
    out.head_ms /= steps - 1U;
    out.total_ms /= steps - 1U;
    return out;
}

} // namespace

int main(int argc, char** argv) try {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    std::string target, draft_file, listen, connect, usb4, spec, vocab_spec = "full";
    std::uint32_t rank = 0, steps = 8;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return std::string(argv[++i]);
        };
        if (a == "--rank") rank = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--listen") listen = next();
        else if (a == "--connect") connect = next();
        else if (a == "--usb4") usb4 = next();
        else if (a == "--transport") spec = next();
        else if (a == "--vocab") vocab_spec = next();
        else if (a == "--steps") steps = static_cast<std::uint32_t>(std::stoul(next()));
        else if (target.empty()) target = a;
        else if (draft_file.empty()) draft_file = a;
        else throw std::invalid_argument("unknown argument " + a);
    }
    const int transports = !listen.empty() + !connect.empty() + !usb4.empty() + !spec.empty();
    if (target.empty() || draft_file.empty() || transports != 1 || rank > 1U || steps < 2U)
        throw std::invalid_argument("usage: strix-ds41-dspark-rank TARGET DRAFT --rank R (--transport SPEC | --listen PORT | --connect HOST:PORT | --usb4 DEV,...)");
    // The shorthand flags are transport specs of the shared opener.
    if (!listen.empty()) spec = "tcp-listen:" + listen;
    if (!connect.empty()) spec = "tcp-connect:" + connect;
    if (!usb4.empty()) spec = "usb4:" + usb4;
    // Connect first (before the weights) so that the peer does not wait on our loading.
    auto opened = strix::dist::open_pair(spec, rank, {std::chrono::seconds(600)});
    strix::dist::Communicator comm({rank, 2U, 1U, 89U}, std::move(opened.host), std::move(opened.direct), 64U);
    const auto weights = load_weights(target, draft_file);
    DeviceDrafter d(weights->weights);
    const auto choice = choose_vocabulary(vocab_spec, {});
    d.set_vocabulary(choice.vocabulary);
    std::printf("rank %u: draft vocabulary %s, transport %s\n", rank, choice.label.c_str(), comm.transport_name().c_str());
    const auto alone = run(d, steps);
    d.reset();
    std::uint64_t request = 0;
    d.set_rank(rank, 2U, [&](void* stream, const void* local, void* remote, std::size_t bytes) {
        comm.exchange(static_cast<hipStream_t>(stream), local, remote, bytes,
                      strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::argmax), 0U, request++});
    });
    const auto split = run(d, steps);
    std::size_t bad = 0;
    for (std::uint32_t i = 0; i < steps; ++i) {
        bad += alone.drafts[i].tokens != split.drafts[i].tokens || alone.drafts[i].confidence != split.drafts[i].confidence;
        std::printf("step %u tokens", i);
        for (const auto t : split.drafts[i].tokens) std::printf(" %u", t);
        std::printf("\n");
    }
    std::printf("rank %u: alone head %.3f ms (draft %.3f ms); split head %.3f ms (draft %.3f ms), %llu exchanges %.3f ms total; %zu of %u "
                "drafts differ from the one-rank drafts\n",
                rank, alone.head_ms, alone.total_ms, split.head_ms, split.total_ms, static_cast<unsigned long long>(comm.stats().exchanges),
                comm.stats().seconds * 1e3, bad, steps);
    return bad == 0U ? 0 : 3;
} catch (const std::exception& error) {
    std::cerr << "strix-ds41-dspark-rank: " << error.what() << '\n';
    return 1;
}
