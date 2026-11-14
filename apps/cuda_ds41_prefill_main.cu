// strix-cuda-ds41-prefill: the deepseek41 CUDA prefill of one prompt on one device (cuda_ds41_prefill.hpp), timed.
//
//   strix-cuda-ds41-prefill MODEL.gguf [--tokens N | --prompt-ids FILE] [--positions N] [--repeat R] [--layers L]
//       [--device-gib G] [--host-gib G] [--host-commit-floor-gib F] [--host-pin-wait-s S] [--tiers DHF..] [--slots S] [--chunk C] [--moe-chunk M] [--window W]
//       [--no-cone | --replay-rows 0|256] [--engram file|zero] [--buffered] [--io-threads N] [--hashes] [--json OUT]
//       [--timing | --timing-layers] [--graphs [--graph-warmup N] | --graphs-ab] [--describe-only]
//       [--no-prefill-kernels] [--prefill-kernels-min-rows N] [--no-prefill-q8] [--prefill-q8-min-rows N] [--fast-dense] [--fast-moe] [--device D]
//       [--fast-attention [--exact-index]] [--fast-moe-signed-lut] [--fast-moe-wide-tiles] [--moe-routes-out FILE]
//       [--expert-residency] [--expert-rank FILE] [--xfer-group N] [--xfer-piece-mib N] [--auto-budget]
//       [--vram-cap-gib G] [--vram-reserve-gib G] [--no-gpu-probe] [--engram-upload mapped|stream] [--engram-cache-mib N]
//
// --tokens N: N tokens (the prompt file tiled, or a fixed pseudo-random sequence without one). Prints one JSON line per
// run: seconds (= TTFT of the prompt without tokenization), tok/s, the next token, Engram / expert I/O, the last row's
// stream FNV per layer with --hashes.
//
// Timing modes (device events, never a host synchronisation inside the run):
//   (none)           end-to-end numbers only: no events at all (the number to quote for tok/s).
//   --timing-layers  per-layer events only (5 per layer): "layers" = [layer, rows, total_ms, attention_ms, moe_ms,
//                    host_expert_wait_ms, device_expert_wait_ms]; overhead within noise.
//   --timing         per-layer events + an event before every op group: adds "ops" {op: device ms over every layer}
//                    (the per-op timeline; the extra events measured within the run-to-run noise end to end on the PRO,
//                    2 layers at 2K / 8K: compare with a run without timing).
// "expert_wait" (op) and device_expert_wait_ms are the device time the stream waits for the layer's expert slot copy;
// host_expert_wait_ms is the time the host blocked in acquire() (until the copy was enqueued), not the device wait.
//
// --graphs: after --graph-warmup N (default 1) eager runs, every layer runs as two captured CUDA graph segments
// replayed on later runs of the same token count (the Engram and expert-slot host waits stay between segments);
// per-op events are off in graph runs (--timing keeps the per-layer ones; attention_ms reads 0). --graphs-ab: eager
// on even runs, graphs on odd runs (an A/B in one process: compare the medians of runs >= 2).
// --describe-only: load, print the "loaded" line (device accounting: dense_gb, expert_gb, slots_gb, buffers_gb,
// state_gb, device_gb = their sum) and exit without a run.
// --replay-rows 256: approximate bounded decoder replay (default 0 = exact cone).
// Requires --fast-dense --fast-moe and all 40 layers; see CudaPrefillOptions.
//
// --batch N1,N2[,..]: closed layer-major batches (CudaPrefill::run_batch). Prompt k = N_k tokens of the prompt file
// tiled from offset k x 7919 (distinct prompts); one extra context of N_k rows per prompt but the longest
// (--positions >= the longest). Per-prompt fast-attention policy: on when --fast-attention and N_k >=
// --batch-fa-min-rows (default 8192, the service's). Each repeat runs every prompt alone (run(), context 0, unless
// --batch-no-solo), then the batch, and prints per prompt: solo / batch seconds, logits FNV, state digest and
// "state_equal" / "logits_equal" (the batch's serialized state and logits byte-compared with the solo run's).
#include "strix/device/cuda_enumerator.hpp"
#include "strix/cuda_ds41_prefill.hpp"
#include "strix/cuda_ds41_replay.hpp"
#include "strix/deepseek41_attention_state.hpp"

#include <algorithm>
#include <chrono>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace strix::ds41::cuda;

namespace {

std::uint64_t fnv(const std::vector<float>& v) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        for (int k = 0; k < 4; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
    }
    return h;
}

[[noreturn]] void usage(const char* why) {
    std::fprintf(stderr, "strix-cuda-ds41-prefill: %s\n", why);
    std::exit(2);
}

std::vector<std::uint32_t> read_ids(const std::string& path) {
    std::ifstream in(path);
    if (!in) usage("cannot read the prompt ids");
    std::vector<std::uint32_t> ids;
    std::string tok;
    while (in >> tok) {
        for (auto& ch : tok)
            if (ch == ',' || ch == '[' || ch == ']') ch = ' ';
        std::istringstream s(tok);
        std::uint64_t v = 0;
        while (s >> v) ids.push_back(static_cast<std::uint32_t>(v));
    }
    if (ids.empty()) usage("empty prompt ids");
    return ids;
}

}  // namespace

int main(int argc, char** argv) {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared card, restricts the process to the chosen one and prints the "device:" receipt line.
    strix::device::bind_cuda_main(argc, argv);
    if (argc < 2) usage("MODEL.gguf [options]");
    const std::string model = argv[1];
    CudaPrefillOptions o;
    std::uint32_t n_tokens = 2048, repeat = 1;
    std::string ids_file, json_out, tiers, hashes_out, state_in, state_out, logits_out;
    std::uint32_t check_extend = 0;
    bool positions_set = false, describe_only = false, graphs_ab = false, batch_solo = true;
    std::vector<std::vector<std::uint32_t>> batches;  // --batch sets (several: one load, every set per repeat)
    std::uint32_t batch_fa_min_rows = 8192;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage(("missing value of " + a).c_str());
            return argv[++i];
        };
        if (a == "--tokens") n_tokens = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--prompt-ids") ids_file = next();
        else if (a == "--pass-rows") {
            const auto value = next();
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), o.pass_rows);
            if (error != std::errc{} || end != value.data() + value.size() || o.pass_rows > 131072U)
                usage("--pass-rows must be an integer in 0..131072");
        }
        else if (a == "--positions" || a == "--context") { o.max_positions = static_cast<std::uint32_t>(std::stoul(next())); positions_set = true; }
        else if (a == "--repeat") repeat = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--device-gib") o.device_expert_gib = std::stod(next());
        else if (a == "--device") o.device = std::stoi(next());
        else if (a == "--host-gib") o.host_expert_gib = std::stod(next());
        else if (a == "--host-commit-floor-gib") o.host_commit_floor_gib = std::stod(next());
        else if (a == "--host-pin-wait-s") o.host_pin_wait_s = std::stod(next());
        else if (a == "--tiers") tiers = next();
        else if (a == "--host-alias") o.host_alias_layers = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--slots") o.slots = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--chunk") o.chunk_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--moe-chunk") o.moe_chunk_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--window") o.window_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--layers") o.end_layer = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--no-cone") o.cone = false;
        else if (a == "--replay-rows") {
            const auto v = next();
            if (v != "0" && v != "256") usage("--replay-rows must be 0 or 256");
            o.replay_rows = v == "256" ? 256U : 0U;
        }
        else if (a == "--engram") o.engram = next();
        else if (a == "--buffered") o.engram_direct = false;
        else if (a == "--io-threads") o.io_threads = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--engram-threads") o.engram_threads = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--hashes-out") { hashes_out = next(); o.stream_hashes = true; }
        else if (a == "--state-out") state_out = next();
        else if (a == "--state-in") state_in = next();
        else if (a == "--check-extend") check_extend = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--head-rows") o.head_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--logits-out") logits_out = next();
        else if (a == "--hashes") o.stream_hashes = true;
        else if (a == "--timing") o.timing = true;
        else if (a == "--timing-layers") { o.timing = true; o.timing_ops = false; }
        else if (a == "--graphs") o.graphs = true;
        else if (a == "--graphs-ab") { o.graphs = true; graphs_ab = true; }
        else if (a == "--graph-warmup") o.graph_warmup_runs = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--describe-only") describe_only = true;
        else if (a == "--no-prefill-kernels") o.prefill_kernels = false;
        else if (a == "--prefill-kernels-min-rows") o.prefill_kernels_min_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--no-prefill-q8") o.prefill_q8 = false;
        else if (a == "--fast-dense") o.fast_dense = true;
        else if (a == "--fast-moe") o.fast_moe = true;
        else if (a == "--fast-moe-signed-lut") o.fast_moe_signed_lut = true;
        else if (a == "--fast-moe-wide-tiles") o.fast_moe_wide_tiles = true;
        else if (a == "--moe-routes-out") o.moe_routes_out = next();
        else if (a == "--prefill-q8-min-rows") o.prefill_q8_min_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--expert-residency") o.expert_residency = true;
        else if (a == "--expert-rank") o.expert_rank = next();
        else if (a == "--xfer-group") o.xfer_group_experts = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--xfer-piece-mib") o.xfer_piece_mib = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--auto-budget") o.auto_budget = true;
        else if (a == "--vram-cap-gib") o.vram_cap_gib = std::stod(next());
        else if (a == "--vram-reserve-gib") o.vram_reserve_gib = std::stod(next());
        else if (a == "--no-gpu-probe") o.gpu_probe = false;
        else if (a == "--engram-upload") o.engram_upload = next();
        else if (a == "--engram-cache-mib") o.engram_cache_mib = std::stoul(next());
        else if (a == "--fast-attention") o.fast_attention = true;
        else if (a == "--exact-index") o.fast_index = false;
        else if (a == "--json") json_out = next();
        else if (a == "--batch") {
            std::string v = next();
            for (auto& ch : v)
                if (ch == ',') ch = ' ';
            std::istringstream is(v);
            std::uint32_t n = 0;
            batches.emplace_back();
            while (is >> n) batches.back().push_back(n);
            if (batches.back().empty()) usage("empty --batch");
        }
        else if (a == "--batch-fa-min-rows") batch_fa_min_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--batch-no-solo") batch_solo = false;
        else usage(("unknown option " + a).c_str());
    }
    if (!batches.empty()) {
        // contexts: the longest prompt of a set in context 0 (max_positions); extra context j holds the (j+2)-th longest
        // prompt of every set
        std::uint32_t mx = 0;
        for (const auto& b : batches) {
            std::vector<std::uint32_t> d(b);
            std::sort(d.rbegin(), d.rend());
            mx = std::max(mx, d[0]);
            if (o.batch_positions.size() < d.size() - 1U) o.batch_positions.resize(d.size() - 1U, 0U);
            for (std::size_t j = 1; j < d.size(); ++j) o.batch_positions[j - 1U] = std::max(o.batch_positions[j - 1U], d[j]);
        }
        n_tokens = mx;
    }
    if (!positions_set) o.max_positions = n_tokens;
    if ((!state_in.empty() || check_extend != 0U) && o.pass_rows == 0U) usage("state-in/check-extend require --pass-rows");
    if (!tiers.empty()) {
        o.tiers.assign(40, ExpertTier::file);
        for (std::size_t l = 0; l < tiers.size() && l < 40; ++l)
            o.tiers[l] = tiers[l] == 'D' ? ExpertTier::device : tiers[l] == 'H' ? ExpertTier::host : ExpertTier::file;
    }
    std::vector<std::uint32_t> tokens(n_tokens);
    if (!ids_file.empty()) {
        const auto ids = read_ids(ids_file);
        for (std::uint32_t i = 0; i < n_tokens; ++i) tokens[i] = ids[i % ids.size()];
    } else {
        std::uint64_t s = 0x9E3779B97F4A7C15ULL;
        for (auto& t : tokens) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            t = static_cast<std::uint32_t>((s >> 33U) % 120000U);
        }
    }
    try {
        validate_bounded_replay(o.replay_rows, o.cone, o.first_layer, o.end_layer,
                                o.head_rows, o.draft_rows, o.fast_dense, o.fast_moe);
        CudaPrefill p(model, o);
        std::printf("{\"event\":\"loaded\",\"describe\":%s}\n", p.describe().c_str());
        std::fflush(stdout);
        if (describe_only) return 0;
        std::ofstream js;
        if (!json_out.empty()) js.open(json_out, std::ios::app);
        if (!batches.empty()) {
            std::vector<std::uint32_t> base;
            if (!ids_file.empty()) base = read_ids(ids_file);
            for (std::uint32_t r = 0; r < repeat; ++r)
            for (const auto& batch : batches) {
            std::vector<std::vector<std::uint32_t>> prompts(batch.size());
            for (std::size_t k = 0; k < batch.size(); ++k) {
                prompts[k].resize(batch[k]);
                for (std::uint32_t i = 0; i < batch[k]; ++i)
                    prompts[k][i] = base.empty() ? static_cast<std::uint32_t>(((std::uint64_t{i} + 7919U * k) * 2654435761ULL >> 7U) % 120000U)
                                                 : base[(std::size_t{i} + 7919U * k) % base.size()];
            }
            const auto fa_of = [&](std::size_t k) { return o.fast_attention && batch[k] >= batch_fa_min_rows; };
            const auto hex = [](std::uint64_t v) {
                char b[17];
                std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(v));
                return std::string(b);
            };
            {
                std::vector<std::vector<std::uint8_t>> solo_state(batch.size());
                std::vector<std::vector<float>> solo_logits(batch.size());
                std::vector<std::vector<std::uint64_t>> solo_hashes(batch.size());
                std::vector<double> solo_s(batch.size(), 0.0);
                if (batch_solo)
                    for (std::size_t k = 0; k < batch.size(); ++k) {
                        p.set_fast_attention(fa_of(k));
                        const auto res = p.run(prompts[k]);
                        solo_s[k] = res.seconds;
                        solo_logits[k] = res.logits;
                        solo_hashes[k] = res.stream_hashes;
                        const auto st = p.export_state();
                        solo_state[k] = strix::deepseek41::serialize_attention_state(st);
                        std::ostringstream s;
                        s << "{\"event\":\"solo\",\"run\":" << r << ",\"item\":" << k << ",\"tokens\":" << batch[k] << ",\"fast_attention\":"
                          << (fa_of(k) ? "true" : "false") << ",\"seconds\":" << res.seconds << ",\"next_token\":" << res.next_token
                          << ",\"expert_gb_streamed\":" << double(res.expert_bytes_streamed) / 1e9 << ",\"expert_copy_busy_s\":" << res.expert_copy_busy_s
                          << ",\"expert_stall_ms\":" << res.expert_stall_ms << ",\"enqueue_s\":" << res.enqueue_seconds << ",\"engram_s\":" << res.engram_seconds
                          << ",\"expert_io_s\":" << res.expert_io_seconds << ",\"logits_fnv\":\"" << hex(fnv(res.logits)) << "\",\"digest\":\""
                          << hex(st.digest()) << "\",\"state_bytes\":" << solo_state[k].size() << "}";
                        std::printf("%s\n", s.str().c_str());
                        std::fflush(stdout);
                        if (js) js << s.str() << "\n";
                    }
                std::vector<CudaBatchItem> items(batch.size());
                for (std::size_t k = 0; k < batch.size(); ++k) items[k] = {prompts[k], fa_of(k)};
                const auto rs = p.run_batch(items);
                double serial = 0.0;
                for (const double v : solo_s) serial += v;
                std::ostringstream s;
                s << "{\"event\":\"batch\",\"run\":" << r << ",\"tokens\":[";
                for (std::size_t k = 0; k < batch.size(); ++k) s << (k ? "," : "") << batch[k];
                s << "],\"batch_seconds\":" << rs[0].batch_seconds << ",\"solo_serial_seconds\":" << serial
                  << ",\"expert_gb_streamed\":" << double(rs[0].expert_bytes_streamed) / 1e9 << ",\"expert_copy_busy_s\":" << rs[0].expert_copy_busy_s
                  << ",\"expert_stall_ms\":" << rs[0].expert_stall_ms << ",\"enqueue_s\":" << rs[0].enqueue_seconds << ",\"engram_s\":" << rs[0].engram_seconds
                  << ",\"expert_io_s\":" << rs[0].expert_io_seconds << ",\"items\":[";
                bool all_equal = true;
                for (std::size_t k = 0; k < batch.size(); ++k) {
                    const auto state = p.export_state(k);
                    const auto bytes = strix::deepseek41::serialize_attention_state(state);
                    const bool se = batch_solo && bytes == solo_state[k];
                    const bool le = batch_solo && rs[k].logits == solo_logits[k] && rs[k].stream_hashes == solo_hashes[k];
                    all_equal = all_equal && se && le;
                    s << (k ? "," : "") << "{\"tokens\":" << batch[k] << ",\"fast_attention\":" << (fa_of(k) ? "true" : "false")
                      << ",\"seconds\":" << rs[k].seconds << ",\"solo_seconds\":" << solo_s[k] << ",\"next_token\":" << rs[k].next_token
                      << ",\"row_layers\":" << rs[k].row_layers << ",\"logits_fnv\":\"" << hex(fnv(rs[k].logits)) << "\",\"digest\":\"" << hex(state.digest())
                      << "\",\"state_bytes\":" << bytes.size() << ",\"state_equal\":" << (se ? "true" : "false") << ",\"logits_equal\":" << (le ? "true" : "false") << "}";
                }
                s << "],\"identical\":" << (all_equal ? "true" : "false");
                if (o.timing) {
                    s << ",\"layers\":[";
                    for (std::size_t l = 0; l < rs[0].layers.size(); ++l) {
                        const auto& t = rs[0].layers[l];
                        s << (l ? "," : "") << "[" << l << "," << t.total_ms << "," << t.attention_ms << "," << t.moe_ms << "," << t.expert_wait_ms << "]";
                    }
                    s << "],\"ops\":{";
                    bool first = true;
                    for (const auto& [name, ms] : rs[0].ops) {
                        s << (first ? "" : ",") << "\"" << name << "\":" << ms;
                        first = false;
                    }
                    s << "}";
                }
                s << "}";
                std::printf("%s\n", s.str().c_str());
                std::fflush(stdout);
                if (js) js << s.str() << "\n";
            }
            }
            return 0;
        }
        for (std::uint32_t r = 0; r < repeat; ++r) {
            if (graphs_ab) p.set_graphs(r % 2U == 1U);  // eager on even runs, graphs on odd runs
            const auto res = [&] {
                if (state_in.empty()) return p.run(tokens);
                const auto state = strix::deepseek41::load_attention_state(state_in);
                if (state.position >= tokens.size() || state.tokens.size() != state.position)
                    throw std::invalid_argument("--state-in needs a valid prefix and a longer final prompt");
                if (!std::equal(state.tokens.begin(), state.tokens.end(), tokens.begin()))
                    throw std::invalid_argument("--state-in token history differs from prompt prefix");
                p.import_state(state);
                return p.extend(std::span<const std::uint32_t>(tokens).subspan(state.position));
            }();
            if (check_extend != 0U) {
                if (o.pass_rows == 0U || check_extend >= tokens.size()) throw std::invalid_argument("--check-extend needs pass_rows and an interior split");
                const auto expected = strix::deepseek41::serialize_attention_state(p.export_state());
                const auto prefix = std::span<const std::uint32_t>(tokens).first(check_extend);
                (void)p.run(prefix);
                const auto checkpoint = strix::deepseek41::deserialize_attention_state(
                    strix::deepseek41::serialize_attention_state(p.export_state()));
                p.import_state(checkpoint);
                const auto continued = p.extend(std::span<const std::uint32_t>(tokens).subspan(check_extend));
                const auto same_floats = [](const std::vector<float>& a, const std::vector<float>& b) {
                    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
                };
                if (expected != strix::deepseek41::serialize_attention_state(p.export_state()) ||
                    !same_floats(res.logits, continued.logits) || !same_floats(res.head_logits, continued.head_logits) ||
                    res.next_token != continued.next_token || res.stream_hashes != continued.stream_hashes)
                    throw std::runtime_error("fresh vs import+extend differs (state bytes/logits/token/stream)");
                std::printf("{\"event\":\"check_extend\",\"split\":%u,\"bytes_equal\":true}\n", check_extend);
            }
            std::ostringstream s;
            s << "{\"event\":\"run\",\"run\":" << r << ",\"tokens\":" << n_tokens << ",\"seconds\":" << res.seconds
              << ",\"enqueue_s\":" << res.enqueue_seconds << ",\"tok_s\":" << n_tokens / res.seconds << ",\"next_token\":" << res.next_token << ",\"engram_s\":" << res.engram_seconds
              << ",\"expert_io_s\":" << res.expert_io_seconds << ",\"expert_gb_streamed\":" << double(res.expert_bytes_streamed) / 1e9
              << ",\"row_layers\":" << res.row_layers << ",\"expert_copy_busy_s\":" << res.expert_copy_busy_s
              << ",\"xfer_gbps\":" << (res.expert_copy_busy_s > 0.0 ? double(res.expert_bytes_streamed) / res.expert_copy_busy_s / 1e9 : 0.0)
              << ",\"expert_stall_ms\":" << res.expert_stall_ms << ",\"engram_records\":" << res.engram_records
              << ",\"engram_pages\":" << res.engram_pages << ",\"engram_page_hits\":" << res.engram_page_hits << ",\"logits_fnv\":\"" << std::hex << fnv(res.logits) << std::dec << "\""
              << ",\"graphs\":" << (res.graphs ? "true" : "false") << ",\"graph_captures\":" << res.graph_captures;
            if (o.timing) {
                s << ",\"layers\":[";
                for (std::size_t l = 0; l < res.layers.size(); ++l) {
                    const auto& t = res.layers[l];
                    s << (l ? "," : "") << "[" << l << "," << t.rows << "," << t.total_ms << "," << t.attention_ms << "," << t.moe_ms << ","
                      << t.expert_wait_ms << "," << t.expert_wait_device_ms << "]";
                }
                s << "],\"ops\":{";
                bool first = true;
                for (const auto& [name, ms] : res.ops) {
                    s << (first ? "" : ",") << "\"" << name << "\":" << ms;
                    first = false;
                }
                s << "}";
            }
            if (o.stream_hashes) {
                s << ",\"stream_hashes\":[";
                for (std::size_t l = 0; l < res.stream_hashes.size(); ++l) s << (l ? "," : "") << "\"" << std::hex << res.stream_hashes[l] << std::dec << "\"";
                s << "]";
            }
            s << "}";
            std::printf("%s\n", s.str().c_str());
            std::fflush(stdout);
            if (js) js << s.str() << "\n";
            if (r + 1U == repeat && !hashes_out.empty()) {  // the format of strix-deepseek41-prefill --stream-hashes-out
                std::FILE* f = std::fopen(hashes_out.c_str(), "wb");
                if (f == nullptr) throw std::runtime_error("cannot write " + hashes_out);
                for (std::size_t l = 0; l < res.stream_hashes.size(); ++l)
                    std::fprintf(f, "stream@%zu %016llx\n", l, static_cast<unsigned long long>(res.stream_hashes[l]));
                for (const auto& [name, h] : p.state_hashes()) std::fprintf(f, "%s %016llx\n", name.c_str(), static_cast<unsigned long long>(h));
                std::fprintf(f, "logits %016llx\n", static_cast<unsigned long long>(fnv(res.logits)));
                std::fclose(f);
            }
            if (r + 1U == repeat && !logits_out.empty()) {  // raw FP32 [rows][vocab] of the last rows (or the last row)
                const auto& L = res.head_logits.empty() ? res.logits : res.head_logits;
                std::ofstream f(logits_out, std::ios::binary);
                f.write(reinterpret_cast<const char*>(L.data()), static_cast<std::streamsize>(L.size() * 4U));
            }
            if (r + 1U == repeat && !state_out.empty()) {
                const auto t0 = std::chrono::steady_clock::now();
                const auto state = p.export_state();
                const double ex = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                strix::deepseek41::save_attention_state(state_out, state);
                std::printf("{\"event\":\"state\",\"file\":\"%s\",\"bytes\":%llu,\"export_s\":%.4f,\"digest\":\"%016llx\"}\n", state_out.c_str(),
                            static_cast<unsigned long long>(state.bytes()), ex, static_cast<unsigned long long>(state.digest()));
                std::fflush(stdout);  // a driver hands the file on as soon as this line arrives (before the process ends)
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-cuda-ds41-prefill: %s\n", e.what());
        return 1;
    }
    return 0;
}
