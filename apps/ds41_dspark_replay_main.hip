// Offline replay of the DSpark drafter on dumped target rows (strix-deepseek41-dspark-rank --mode dump): at every
// generated position p the drafter holds the committed rows 0..p-1 and drafts from the token at p, as the runtime does
// when a window starts at p (same device drafter, same bits). The drafts are compared with the target's greedy
// continuation, so the statistics are those of teacher-forced windows at every position.
//   strix-ds41-dspark-replay TARGET.gguf DRAFT.gguf [--vocab SPEC] [--top-rank] DUMP-PREFIX...
// Per dump and in total: P(d_j ok | d_<j ok), accepted drafts per window (fixed 5), how often the target's token is
// the drafter's 2nd/3rd/<=8th choice at depth 1 (with the Markov bias of the input token, --top-rank), the depth-1
// hit rate by the target's own top-1 probability (over its 32 largest logits) and by the drafter's confidence.
// --streams-raw FILE replaces the drafter rows of the (single) dump with rows made from other target streams, e.g. the
// official model's with FP8 activations, to see how the target's activation identity moves acceptance.
#include "strix/device/hip_gate.hpp"
#include "strix/ds41_dspark_files.hpp"
#include "strix/ds41_dspark_vocab.hpp"
#include "strix/hip_ds41_dspark.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ds = strix::ds41::dspark;

struct Dump {
    std::size_t prompt = 0;
    std::vector<std::uint32_t> tokens;  // prompt then generated
    std::vector<float> hidden;          // [steps][15360]
    std::vector<std::array<std::pair<std::uint32_t, float>, 32>> top;
};

Dump read_dump(const std::string& prefix) {
    Dump d;
    std::ifstream ft(prefix + ".tokens");
    if (!ft) throw std::runtime_error("cannot read " + prefix + ".tokens");
    std::string line;
    std::getline(ft, line);
    d.prompt = std::stoul(line);
    std::getline(ft, line);
    std::stringstream ss(line);
    for (std::string t; std::getline(ss, t, ',');) d.tokens.push_back(static_cast<std::uint32_t>(std::stoul(t)));
    std::ifstream fh(prefix + ".hidden", std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::size_t>(fh.tellg());
    fh.seekg(0);
    d.hidden.resize(bytes / 4U);
    fh.read(reinterpret_cast<char*>(d.hidden.data()), static_cast<std::streamsize>(bytes));
    std::ifstream fp(prefix + ".top", std::ios::binary);
    const std::size_t steps = d.hidden.size() / ds::main_width;
    d.top.resize(steps);
    for (auto& row : d.top)
        for (auto& [id, v] : row) {
            fp.read(reinterpret_cast<char*>(&id), 4);
            fp.read(reinterpret_cast<char*>(&v), 4);
        }
    if (steps + 1U != d.tokens.size()) throw std::runtime_error(prefix + ": steps and tokens disagree");
    return d;
}

struct Stats {
    std::array<std::uint32_t, ds::block_size> reached{}, hit{};
    std::uint32_t windows = 0, accepted = 0;
    std::array<std::uint32_t, 4> rank_bins{};      // depth 1 target token rank: 1, 2, 3, 4..8 (else miss)
    std::uint32_t out_of_vocab = 0, nomarkov_hit = 0, ranked = 0;  // --top-rank: depth-1 hit without the Markov bias
    std::array<std::uint32_t, 5> conf_seen{}, conf_hit{};  // depth 1 by the target's top-1 - top-2 logit margin
    std::array<std::uint32_t, 8> dconf_seen{}, dconf_hit{};  // depth 1 by drafter confidence bins
    void add(const Stats& o) {
        for (std::size_t j = 0; j < reached.size(); ++j) reached[j] += o.reached[j], hit[j] += o.hit[j];
        windows += o.windows, accepted += o.accepted, out_of_vocab += o.out_of_vocab, nomarkov_hit += o.nomarkov_hit, ranked += o.ranked;
        for (std::size_t j = 0; j < 4; ++j) rank_bins[j] += o.rank_bins[j];
        for (std::size_t j = 0; j < 5; ++j) conf_seen[j] += o.conf_seen[j], conf_hit[j] += o.conf_hit[j];
        for (std::size_t j = 0; j < 8; ++j) dconf_seen[j] += o.dconf_seen[j], dconf_hit[j] += o.dconf_hit[j];
    }
    void print(const char* label) const {
        std::printf("%s: windows %llu, accepted %.3f per window (fixed 5)\n  P(d_j ok | d_<j ok):", label,
                    static_cast<unsigned long long>(windows), windows ? static_cast<double>(accepted) / windows : 0.0);
        for (std::size_t j = 0; j < reached.size(); ++j)
            std::printf(" %.3f (%llu)", reached[j] ? static_cast<double>(hit[j]) / reached[j] : 0.0, static_cast<unsigned long long>(reached[j]));
        const double n = static_cast<double>(windows ? windows : 1);
        std::printf("\n  depth 1, target token's rank in the drafter: 1st %.3f 2nd %.3f 3rd %.3f 4-8th %.3f, out of the draft vocabulary %.3f\n",
                    rank_bins[0] / n, rank_bins[1] / n, rank_bins[2] / n, rank_bins[3] / n, out_of_vocab / n);
        if (ranked != 0U)
            std::printf("  depth 1 without the Markov bias (argmax of the head logits alone): hit %.3f (with it: %.3f)\n",
                        static_cast<double>(nomarkov_hit) / ranked, static_cast<double>(rank_bins[0]) / ranked);
        static const char* tb[5] = {"<0.5", "0.5-1", "1-2", "2-4", ">=4"};
        std::printf("  depth 1 hit by the target's top-1 - top-2 logit margin:");
        for (std::size_t j = 0; j < 5; ++j)
            std::printf(" %s %.2f (%.0f%%)", tb[j], conf_seen[j] ? static_cast<double>(conf_hit[j]) / conf_seen[j] : 0.0, 100.0 * conf_seen[j] / n);
        std::printf("\n  depth 1 hit by the drafter's confidence bin (SurvivalController bins):");
        for (std::size_t j = 0; j < 8; ++j)
            std::printf(" %.2f(%llu)", dconf_seen[j] ? static_cast<double>(dconf_hit[j]) / dconf_seen[j] : 0.0,
                        static_cast<unsigned long long>(dconf_seen[j]));
        std::printf("\n");
    }
};

}  // namespace

int main(int argc, char** argv) try {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    std::string target, draft_file, vocab_spec = "full", streams_raw, features;
    bool top_rank = false, all_positions = false;
    std::vector<std::string> dumps;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--vocab" && i + 1 < argc) vocab_spec = argv[++i];
        else if (s == "--top-rank") top_rank = true;
        else if (s == "--streams-raw" && i + 1 < argc) streams_raw = argv[++i];
        else if (s == "--features" && i + 1 < argc) features = argv[++i];
        else if (s == "--all-positions") all_positions = true;
        else if (target.empty()) target = s;
        else if (draft_file.empty()) draft_file = s;
        else dumps.push_back(s);
    }
    if (target.empty() || draft_file.empty() || dumps.empty()) throw std::invalid_argument("usage: strix-ds41-dspark-replay TARGET DRAFT [--vocab SPEC] [--top-rank] DUMP...");
    const auto weights = ds::load_weights(target, draft_file);
    ds::DeviceDrafter drafter(weights->weights);
    const auto& head = weights->weights.markov_head;
    Stats total;
    std::vector<double> prompt_s1, prompt_acc;
    for (const auto& prefix : dumps) {
        auto d = read_dump(prefix);
        if (!streams_raw.empty()) {  // the drafter rows from other target streams: f32 [steps][3][4][5120] (inputs of 37-39)
            std::ifstream fr(streams_raw, std::ios::binary);
            const std::size_t per = std::size_t{3} * 4U * 5120U, steps = d.hidden.size() / ds::main_width;
            std::vector<float> raw(per);
            for (std::size_t p = 0; p < steps; ++p) {
                if (!fr.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(per * 4U)))
                    throw std::runtime_error("streams-raw shorter than the dump");
                const std::span<const float> r(raw);
                const auto row = ds::main_hidden_row(r.subspan(0, per / 3U), r.subspan(per / 3U, per / 3U), r.subspan(2U * per / 3U, per / 3U));
                std::copy(row.begin(), row.end(), d.hidden.begin() + static_cast<std::ptrdiff_t>(p * ds::main_width));
            }
            std::printf("rows replaced from %s\n", streams_raw.c_str());
        }
        const std::span<const std::uint32_t> prompt(d.tokens.data(), d.prompt);
        auto choice = ds::choose_vocabulary(vocab_spec, prompt);
        drafter.reset();
        drafter.set_vocabulary(choice.vocabulary);
        (void)drafter.add_vocabulary(prompt);
        auto vocab = drafter.vocabulary();
        const std::size_t steps = d.tokens.size() - 1U;
        // committed rows 0..prompt-1, then one more row before each later position
        std::size_t appended = 0;
        const auto append_to = [&](std::size_t n) {
            while (appended < n) {
                const auto k = static_cast<std::uint32_t>(std::min<std::size_t>(128U, n - appended));
                drafter.append_hidden(std::span<const float>(d.hidden).subspan(appended * ds::main_width, std::size_t{k} * ds::main_width), k);
                appended += k;
            }
        };
        Stats st;
        std::FILE* ff = nullptr;
        if (!features.empty()) {
            ff = std::fopen((features + "-" + std::to_string(&prefix - dumps.data()) + ".feat").c_str(), "wb");
            if (ff == nullptr) throw std::runtime_error("cannot write the features");
        }
        const std::size_t first = all_positions ? std::min<std::size_t>(16U, d.prompt) : d.prompt;
        for (std::size_t p = first; p + 1U < d.tokens.size() && p < steps; ++p) {
            append_to(p);
            const std::uint32_t context[1] = {d.tokens[p]};
            if (drafter.add_vocabulary(context) != 0U) vocab = drafter.vocabulary();
            const auto dr = drafter.draft(d.tokens[p], top_rank || ff != nullptr);
            bool labelled = p + ds::block_size <= steps;
            for (std::uint32_t j = 0; labelled && j < 5U; ++j) labelled = d.top[p + j][0].first != 0xFFFFFFFFU;
            if (ff != nullptr && labelled) {
                // record: tokens p..p+5 (u32 x 6), the target's argmax after steps p..p+4 (u32 x 5), its top-32 at
                // those steps (u32 id, f32 logit) x 32 x 5, head_in [5][5120] as bf16 (round to nearest even)
                std::uint32_t tk[6], lab[5];
                for (std::uint32_t j = 0; j < 6U; ++j) tk[j] = p + j < d.tokens.size() ? d.tokens[p + j] : 0U;
                for (std::uint32_t j = 0; j < 5U; ++j) lab[j] = d.top[p + j][0].first;
                std::fwrite(tk, 4, 6, ff);
                std::fwrite(lab, 4, 5, ff);
                for (std::uint32_t j = 0; j < 5U; ++j)
                    for (const auto& [id, v] : d.top[p + j]) {
                        std::fwrite(&id, 4, 1, ff);
                        std::fwrite(&v, 4, 1, ff);
                    }
                std::vector<std::uint16_t> bf(dr.head_in.size());
                for (std::size_t q = 0; q < bf.size(); ++q) {
                    std::uint32_t u;
                    std::memcpy(&u, &dr.head_in[q], 4);
                    bf[q] = static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16U) & 1U)) >> 16U);
                }
                std::fwrite(bf.data(), 2, bf.size(), ff);
            }
            if (p < d.prompt) continue;  // statistics on generated positions only
            ++st.windows;
            std::uint32_t a = 0;
            for (std::uint32_t j = 0; j < ds::block_size && p + 1U + j < d.tokens.size(); ++j) {
                ++st.reached[j];
                if (dr.tokens[j + 1U] != d.tokens[p + 1U + j]) break;
                ++st.hit[j];
                ++a;
            }
            st.accepted += a;
            const std::uint32_t want = d.tokens[p + 1U];
            const bool ok1 = dr.tokens[1] == want;
            // the target's own margin between its two largest logits at step p
            const auto& tp = d.top[p];
            const double margin = static_cast<double>(tp[0].second - tp[1].second);
            const std::size_t tb = margin < 0.5 ? 0 : margin < 1.0 ? 1 : margin < 2.0 ? 2 : margin < 4.0 ? 3 : 4;
            ++st.conf_seen[tb];
            st.conf_hit[tb] += ok1;
            const auto cb = ds::SurvivalController::bin(dr.confidence[0]);
            ++st.dconf_seen[cb];
            st.dconf_hit[cb] += ok1;
            if (!ds::contains(vocab, want)) { ++st.out_of_vocab; continue; }
            if (!top_rank) { st.rank_bins[0] += ok1; continue; }
            // depth-1 logits with the Markov bias of the input token, over the draft vocabulary
            const std::uint32_t V = vocab.size();
            std::vector<std::pair<float, std::uint32_t>> fin(V);
            for (std::uint32_t j = 0; j < V; ++j) {
                const std::uint32_t id = vocab.id(j);
                fin[j] = {dr.logits[j] + ds::markov_dot(head.bits.data() + std::size_t{id} * head.width, dr.markov.data()), id};
            }
            std::uint32_t rank = 0;
            float wv = 0.0F;
            for (const auto& [v, id] : fin)
                if (id == want) wv = v;
            for (const auto& [v, id] : fin) rank += (v > wv || (v == wv && id < want));
            {  // the head logits alone (no Markov bias): lowest id on ties, as the drafter
                std::uint32_t best = 0;
                for (std::uint32_t j = 1; j < V; ++j)
                    if (dr.logits[j] > dr.logits[best] || (dr.logits[j] == dr.logits[best] && vocab.id(j) < vocab.id(best))) best = j;
                st.nomarkov_hit += vocab.id(best) == want;
                ++st.ranked;
            }
            if (rank < 3U) ++st.rank_bins[rank];
            else if (rank < 8U) ++st.rank_bins[3];
        }
        if (ff != nullptr) std::fclose(ff);
        std::printf("%s (%s, prompt %zu, %zu generated)\n", prefix.c_str(), choice.label.c_str(), d.prompt, d.tokens.size() - d.prompt);
        st.print("  replay");
        if (st.windows != 0U) {
            prompt_s1.push_back(static_cast<double>(st.hit[0]) / st.reached[0]);
            prompt_acc.push_back(static_cast<double>(st.accepted) / st.windows);
        }
        total.add(st);
        std::fflush(stdout);
    }
    total.print("TOTAL");
    const auto summary = [](const char* label, std::vector<double> v) {
        if (v.empty()) return;
        std::sort(v.begin(), v.end());
        double m = 0.0, q = 0.0;
        for (const double x : v) m += x;
        m /= static_cast<double>(v.size());
        for (const double x : v) q += (x - m) * (x - m);
        const double sd = v.size() > 1U ? std::sqrt(q / static_cast<double>(v.size() - 1U)) : 0.0;
        std::printf("  per prompt %s: mean %.3f, sd %.3f, 95%% interval of the mean +-%.3f, min %.3f, median %.3f, max %.3f (%zu prompts)\n",
                    label, m, sd, 1.96 * sd / std::sqrt(static_cast<double>(v.size())), v.front(), v[v.size() / 2U], v.back(), v.size());
    };
    summary("S1", prompt_s1);
    summary("accepted per window", prompt_acc);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
