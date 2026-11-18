#include "strix/hip_gdn_launcher.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "attn_core/recurrent_state.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using namespace strix;
namespace rstate = strix::attn_core::rstate;
constexpr std::uint32_t hidden = 2560, keys = 16, heads = 48, dim = 128, kernel = 4;
constexpr std::uint32_t key_width = keys * dim, value_width = heads * dim, conv_width = 2U * key_width + value_width;
constexpr std::size_t workspace_floats = conv_width + value_width + 2U * heads + value_width;
constexpr std::size_t state_floats = static_cast<std::size_t>(heads) * dim * dim;
void require(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}
struct Rng {
    std::uint64_t state;
    std::uint32_t next() { state = state * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<std::uint32_t>(state >> 33U); }
    float uniform(float lo, float hi) { return lo + (hi - lo) * static_cast<float>(next() & 0xffffffU) / 16777216.0F; }
};
std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}
struct Buffer { std::byte* address = nullptr; std::size_t bytes = 0; };
struct Owner {
    HipSetupQuiescenceGuard guard;
    hipStream_t stream = nullptr;
    std::vector<void*> allocations;
    std::vector<hipGraph_t> graphs;
    std::vector<hipGraphExec_t> execs;
    std::vector<Buffer> guarded;
    Owner() { guard.run([&] { check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create deferred GDN stream"); }); }
    ~Owner() {
        guard.recover_or_fail_stop();
        for (const auto exec : execs) if (hipGraphExecDestroy(exec) != hipSuccess) std::_Exit(3);
        for (const auto graph : graphs) if (hipGraphDestroy(graph) != hipSuccess) std::_Exit(3);
        for (auto* allocation : allocations) if (hipFree(allocation) != hipSuccess) std::_Exit(3);
        if (hipStreamDestroy(stream) != hipSuccess) std::_Exit(3);
    }
    Buffer allocate(std::size_t bytes) {
        void* p = nullptr;
        check(hipMalloc(&p, bytes + 512U), "allocate deferred GDN owner");
        try { allocations.push_back(p); }
        catch (...) { guard.recover_or_fail_stop(); (void)hipFree(p); throw; }
        check(hipMemset(p, 0xa5, bytes + 512U), "poison deferred GDN guards");
        guarded.push_back({static_cast<std::byte*>(p) + 256U, bytes});
        return guarded.back();
    }
    void finish() { check(hipStreamSynchronize(stream), "finish deferred GDN"); guard.require_original(); }
    void guards() const {
        std::vector<std::byte> bytes(256U);
        for (const auto& buffer : guarded)
            for (const auto* address : {buffer.address - 256U, buffer.address + buffer.bytes}) {
                check(hipMemcpy(bytes.data(), address, 256U, hipMemcpyDeviceToHost), "read deferred GDN guard");
                require(std::all_of(bytes.begin(), bytes.end(), [](auto b) { return b == std::byte{0xa5}; }),
                    "deferred GDN exterior guard changed");
            }
    }
};
std::vector<std::byte> read(Buffer buffer) {
    std::vector<std::byte> bytes(buffer.bytes);
    check(hipMemcpy(bytes.data(), buffer.address, bytes.size(), hipMemcpyDeviceToHost), "read deferred GDN bytes");
    return bytes;
}
template <class T> void write(Buffer buffer, const std::vector<T>& values) {
    require(values.size() * sizeof(T) == buffer.bytes, "deferred GDN upload extent differs");
    check(hipMemcpy(buffer.address, values.data(), buffer.bytes, hipMemcpyHostToDevice), "upload deferred GDN bytes");
}
std::vector<std::byte> q8_weights(std::uint32_t width, std::uint32_t outputs, Rng& rng) {
    const auto blocks = static_cast<std::size_t>(width / 32U) * outputs;
    std::vector<std::byte> result(blocks * 34U);
    constexpr std::array<std::uint16_t, 4> scales{0x1c00, 0x1e00, 0x9c00, 0x1a00};
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto scale = scales[rng.next() % scales.size()];
        result[block * 34U] = static_cast<std::byte>(scale & 0xffU);
        result[block * 34U + 1U] = static_cast<std::byte>(scale >> 8U);
        for (std::size_t lane = 0; lane < 32U; ++lane)
            result[block * 34U + 2U + lane] = static_cast<std::byte>(static_cast<std::uint8_t>(
                static_cast<std::int8_t>(static_cast<int>(rng.next() % 255U) - 127)));
    }
    return result;
}
template <class T> std::vector<T> uniform(std::size_t count, float lo, float hi, Rng& rng) {
    std::vector<T> result(count);
    for (auto& value : result) {
        if constexpr (sizeof(T) == 2U) value = bf16(rng.uniform(lo, hi)); else value = rng.uniform(lo, hi);
    }
    return result;
}
// Exact allocation-sized image: typed fixtures must not hide f32 reads or writes in an oversized tail.
std::vector<std::byte> state_image(const std::vector<float>& values, RecurrentStateType type) {
    std::vector<std::byte> image(static_cast<std::size_t>(recurrent_state_bytes(values.size(), type)), std::byte{0});
    if (type == RecurrentStateType::f32) { std::memcpy(image.data(), values.data(), image.size()); return image; }
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::uint16_t half = type == RecurrentStateType::f16 ? rstate::f16_from_f32_host(values[i]) : rstate::bf16_from_f32(values[i]);
        std::memcpy(image.data() + 2U * i, &half, 2U);
    }
    return image;
}
// Widening readback of such an image.
std::vector<float> state_values(const std::vector<std::byte>& image, std::size_t count, RecurrentStateType type) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (type == RecurrentStateType::f32) { std::memcpy(&values[i], image.data() + 4U * i, 4U); continue; }
        std::uint16_t half = 0; std::memcpy(&half, image.data() + 2U * i, 2U);
        values[i] = type == RecurrentStateType::f16 ? rstate::f32_from_f16_host(half) : rstate::f32_from_bf16(half);
    }
    return values;
}
struct Side { // One recurrent state with its rows.
    Buffer matrix, conv, retained;
    std::vector<Buffer> input, output, workspace;
};
struct Fixture {
    Owner owner;
    HipGdnLauncherConfig config;
    bool q8_gates, bf16_small;
    Buffer qkv, z, beta, alpha, out, conv_weights, dt, a, norm, control;
    Side window, serial;
    std::vector<float> state0; std::vector<std::uint16_t> history0;
    RecurrentStateType state_type;
    std::vector<std::byte> state_image0; // state0 as stored under state_type
    Fixture(GdnRecurrentStorage storage, HipGdnRecurrentLayout layout, bool q8, bool bf16_weights, unsigned seed,
        RecurrentStateType type = RecurrentStateType::f32)
        : q8_gates(q8), bf16_small(bf16_weights), state_type(type) {
        config.model.recurrent_state = type;
        config.virtual_q8_input_projections = config.prefetch_q8_input_projections = config.wave8_q8_input_projections = true;
        config.virtual_q8_output_projection = config.wave8_q8_output_projection = true;
        config.recurrent_storage = storage; config.recurrent_layout = layout;
        Rng rng{seed * 7919ULL + 17U};
        const auto upload = [&](Buffer& target, const auto& values) {
            target = owner.allocate(values.size() * sizeof(values[0])); write(target, values);
        };
        upload(qkv, q8_weights(hidden, conv_width, rng)); upload(z, q8_weights(hidden, value_width, rng));
        upload(out, q8_weights(value_width, hidden, rng));
        if (q8_gates) { upload(beta, q8_weights(hidden, heads, rng)); upload(alpha, q8_weights(hidden, heads, rng)); }
        else { upload(beta, uniform<float>(hidden * heads, -0.03F, 0.03F, rng)); upload(alpha, uniform<float>(hidden * heads, -0.03F, 0.03F, rng)); }
        if (bf16_small) {
            upload(conv_weights, uniform<std::uint16_t>(conv_width * kernel, -0.6F, 0.6F, rng));
            upload(dt, uniform<std::uint16_t>(heads, -1.0F, 1.0F, rng)); upload(a, uniform<std::uint16_t>(heads, -2.0F, -0.3F, rng));
            upload(norm, uniform<std::uint16_t>(dim, 0.5F, 1.5F, rng));
        } else {
            upload(conv_weights, uniform<float>(conv_width * kernel, -0.6F, 0.6F, rng));
            upload(dt, uniform<float>(heads, -1.0F, 1.0F, rng)); upload(a, uniform<float>(heads, -2.0F, -0.3F, rng));
            upload(norm, uniform<float>(dim, 0.5F, 1.5F, rng));
        }
        control = owner.allocate(sizeof(GdnWindowControl));
        state0 = uniform<float>(state_floats, -0.2F, 0.2F, rng);
        state_image0 = state_image(state0, type);
        history0 = uniform<std::uint16_t>(static_cast<std::size_t>(conv_width) * kernel, -1.0F, 1.0F, rng);
        for (auto* side : {&window, &serial}) {
            side->matrix = owner.allocate(state_image0.size());
            side->conv = owner.allocate(history0.size() * sizeof(std::uint16_t));
            side->retained = owner.allocate(static_cast<std::size_t>(gdn_window_retained_bytes(config.model)));
            for (unsigned row = 0; row < 16U; ++row) {
                side->input.push_back(owner.allocate(hidden * sizeof(std::uint16_t)));
                side->output.push_back(owner.allocate(hidden * sizeof(std::uint16_t)));
                side->workspace.push_back(owner.allocate(workspace_floats * sizeof(float)));
            }
        }
        // Rows 0..7 feed the first window, 8..15 the second; inputs are shared.
        for (unsigned row = 0; row < 16U; ++row) {
            const auto values = uniform<std::uint16_t>(hidden, -1.0F, 1.0F, rng);
            write(window.input[row], values); write(serial.input[row], values);
        }
    }
    GdnWindowLayer layer(const Side& side) const {
        GdnWindowLayer result;
        result.qkv = qkv.address; result.z = z.address; result.beta = beta.address; result.alpha = alpha.address;
        result.out = out.address; result.conv = conv_weights.address; result.dt = dt.address; result.a = a.address;
        result.norm = norm.address; result.gate_type = q8_gates ? 8U : 0U;
        result.conv_type = result.dt_type = result.a_type = result.norm_type = bf16_small ? 30U : 0U;
        result.matrix = reinterpret_cast<float*>(side.matrix.address);
        result.conv_state = reinterpret_cast<std::uint16_t*>(side.conv.address);
        result.retained = side.retained.address;
        return result;
    }
    std::vector<GdnWindowRow> rows(const Side& side, unsigned first, unsigned count) const {
        std::vector<GdnWindowRow> result;
        for (unsigned r = first; r < first + count; ++r)
            result.push_back({reinterpret_cast<const std::uint16_t*>(side.input[r].address),
                reinterpret_cast<std::uint16_t*>(side.output[r].address), reinterpret_cast<float*>(side.workspace[r].address)});
        return result;
    }
    void reset() {
        for (auto* side : {&window, &serial}) {
            write(side->matrix, state_image0); write(side->conv, history0);
            for (unsigned r = 0; r < 16U; ++r) {
                check(hipMemset(side->output[r].address, 0x5a, side->output[r].bytes), "poison deferred GDN output");
                check(hipMemset(side->workspace[r].address, 0x5a, side->workspace[r].bytes), "poison deferred GDN workspace");
            }
        }
        check(hipMemset(window.retained.address, 0x5a, window.retained.bytes), "poison deferred GDN retained rows");
    }
    void set_control(std::uint32_t pending, std::uint32_t slot) {
        const GdnWindowControl value{pending, slot};
        check(hipMemcpyAsync(control.address, &value, sizeof(value), hipMemcpyHostToDevice, owner.stream), "set deferred GDN control");
        owner.finish();
    }
    const GdnWindowControl* device_control() const { return reinterpret_cast<const GdnWindowControl*>(control.address); }
    void serial_rows(unsigned first, unsigned count) {
        const auto views = rows(serial, first, count);
        for (const auto& row : views) gdn_window_serial_row_probe(layer(serial), row, config, owner.stream);
        owner.finish();
    }
    void window_rows(unsigned first, unsigned count, bool capture) {
        const auto views = rows(window, first, count);
        if (!capture) { gdn_window_enqueue(layer(window), views, device_control(), config, owner.stream); owner.finish(); return; }
        hipGraph_t graph = nullptr; hipGraphExec_t exec = nullptr;
        check(hipStreamBeginCapture(owner.stream, hipStreamCaptureModeThreadLocal), "begin deferred GDN capture");
        gdn_window_enqueue(layer(window), views, device_control(), config, owner.stream);
        check(hipStreamEndCapture(owner.stream, &graph), "end deferred GDN capture"); owner.graphs.push_back(graph);
        check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate deferred GDN graph"); owner.execs.push_back(exec);
        check(hipGraphLaunch(exec, owner.stream), "launch deferred GDN graph"); owner.finish();
    }
    void commit() { gdn_window_commit_enqueue(layer(window), device_control(), config, owner.stream); owner.finish(); }
    void same_rows(unsigned first, unsigned count, const std::string& what) const {
        for (unsigned r = first; r < first + count; ++r) {
            require(read(window.workspace[r]) == read(serial.workspace[r]), what + ": workspace/core of row " + std::to_string(r) + " differs");
            require(read(window.output[r]) == read(serial.output[r]), what + ": output of row " + std::to_string(r) + " differs");
        }
    }
    void same_state(const std::string& what) const {
        require(read(window.matrix) == read(serial.matrix), what + ": recurrent matrix differs");
        require(read(window.conv) == read(serial.conv), what + ": conv history differs");
    }
    // Window 1 (rows 0..t-1) from the initial state, keep n; window 2 (rows
    // 8..8+u-1, u = t unless mixed sizes) applies those n first; keep m and commit.
    // Serial: rows 0..n-1, 8..8+u-1 are re-run from the committed prefix for every comparison.
    void scenario(unsigned t, unsigned n, unsigned m, bool capture, unsigned u = 0U) {
        if (!u) u = t;
        const auto label = "T=" + std::to_string(t) + (u != t ? "->" + std::to_string(u) : std::string{}) +
            " n_pending=" + std::to_string(n) + " n_keep=" + std::to_string(m);
        reset();
        set_control(0U, 1U);
        window_rows(0U, t, false);
        serial_rows(0U, t);
        same_rows(0U, t, label + " window 1");
        const auto bytes = [](const auto& values) {
            std::vector<std::byte> result(values.size() * sizeof(values[0]));
            std::memcpy(result.data(), values.data(), result.size()); return result;
        };
        require(read(window.matrix) == state_image0 && read(window.conv) == bytes(history0),
            label + ": window without pending rows wrote its state");
        require(read(serial.matrix) != state_image0, label + ": serial reference did not advance");
        // Rebuild the serial state at the committed prefix n.
        write(serial.matrix, state_image0); write(serial.conv, history0);
        serial_rows(0U, n);
        set_control(n, 0U);
        window_rows(8U, u, capture);
        same_state(label + " after pending commit");
        serial_rows(8U, u);
        same_rows(8U, u, label + " window 2");
        write(serial.matrix, state_image0); write(serial.conv, history0);
        serial_rows(0U, n); serial_rows(8U, m);
        set_control(m, 1U);
        commit();
        same_state(label + " after standalone commit");
        owner.guards();
    }
};
// Cohort: several sessions (own state, history, retained rows, control and inputs) in one
// gdn_window_cohort_enqueue against each session's own gdn_window_enqueue from the same state.
struct CohortSession {
    Side together, alone;
    Buffer control_together, control_alone;
    std::vector<float> state; std::vector<std::byte> image; std::vector<std::uint16_t> history;
};
void cohort_scenarios(Fixture& f, unsigned& cases) {
    std::vector<CohortSession> sessions(8);
    Rng rng{991U};
    for (auto& session : sessions) {
        session.state = uniform<float>(state_floats, -0.2F, 0.2F, rng);
        session.image = state_image(session.state, f.state_type);
        session.history = uniform<std::uint16_t>(static_cast<std::size_t>(conv_width) * kernel, -1.0F, 1.0F, rng);
        for (auto* side : {&session.together, &session.alone}) {
            side->matrix = f.owner.allocate(f.state_image0.size());
            side->conv = f.owner.allocate(session.history.size() * sizeof(std::uint16_t));
            side->retained = f.owner.allocate(static_cast<std::size_t>(gdn_window_retained_bytes(f.config.model)));
            for (unsigned row = 0; row < 16U; ++row) {
                side->input.push_back(f.owner.allocate(hidden * sizeof(std::uint16_t)));
                side->output.push_back(f.owner.allocate(hidden * sizeof(std::uint16_t)));
                side->workspace.push_back(f.owner.allocate(workspace_floats * sizeof(float)));
            }
        }
        session.control_together = f.owner.allocate(sizeof(GdnWindowControl));
        session.control_alone = f.owner.allocate(sizeof(GdnWindowControl));
        for (unsigned row = 0; row < 16U; ++row) {
            const auto values = uniform<std::uint16_t>(hidden, -1.0F, 1.0F, rng);
            write(session.together.input[row], values); write(session.alone.input[row], values);
        }
    }
    const auto control = [&](Buffer buffer, std::uint32_t pending, std::uint32_t slot) {
        const GdnWindowControl value{pending, slot};
        check(hipMemcpyAsync(buffer.address, &value, sizeof(value), hipMemcpyHostToDevice, f.owner.stream), "set cohort GDN control");
    };
    const auto device = [](Buffer buffer) { return reinterpret_cast<const GdnWindowControl*>(buffer.address); };
    struct Case { std::vector<unsigned> sizes; };
    // Up to 8 rows (one launch of the 1..8-row forms) and 16 / 32 rows (cohorts of several 8-row chunks:
    // input and output projections as one 16- or 32-row launch, the recurrence per session); both with the
    // Q8 and the dense A8 projections.
    const std::vector<Case> shapes{{{2U, 2U}}, {{4U, 4U}}, {{2U, 2U, 2U, 2U}}, {{2U, 2U, 4U}}, {{4U, 2U, 2U}}, {{2U, 6U}},
        {{1U, 7U}}, {{3U, 5U}}, {{1U, 1U}}, {{8U}}, {{8U, 8U}}, {{4U, 4U, 4U, 4U}}, {{8U, 4U, 2U, 2U}},
        {{2U, 2U, 2U, 2U, 2U, 2U, 2U, 2U}}, {{8U, 8U, 8U, 8U}}, {{8U, 4U, 4U, 8U, 2U, 2U, 2U, 2U}},
        {{4U, 4U, 2U}}, {{8U, 4U, 2U}}, {{8U, 8U, 4U, 2U, 2U}}, {{8U, 8U, 8U, 2U}},
        // ANYSEG: mixed row counts in a full 32-row cohort (8-row segments are
        // covered above), with distinct pending/retained histories and slot parities.
        {{1U, 2U, 3U, 4U, 5U, 6U, 7U, 4U}}, {{7U, 1U}}, {{5U, 1U, 1U, 1U}}, {{3U, 1U, 1U, 3U}}};
    const bool dense_before = f.config.dense_a8_projections;
    for (const bool dense : {false, true})
    for (const auto& shape : shapes) {
        unsigned total = 0;
        for (const auto size : shape.sizes) total += size;
        if (total > gdn_window_max_rows && !dense) {
            // Wider cohorts require dense A8: the Q8 FP32 forms are rejected before any launch.
            f.config.dense_a8_projections = false;
            std::vector<GdnWindowCohortSegment> segments;
            for (std::size_t k = 0; k < shape.sizes.size(); ++k)
                segments.push_back({f.layer(sessions[k].together), f.rows(sessions[k].together, 8U, shape.sizes[k]),
                    device(sessions[k].control_together)});
            bool thrown = false;
            try { gdn_window_cohort_enqueue(segments, f.config, f.owner.stream); } catch (const std::invalid_argument&) { thrown = true; }
            require(thrown, "GDN cohort of 16 or 32 rows without dense A8 was not rejected");
            continue;
        }
        f.config.dense_a8_projections = dense;
        const auto count = shape.sizes.size();
        for (unsigned variant = 0; variant < 3U; ++variant) {
            std::string label = dense ? "cohort dense A8" : "cohort";
            for (const auto size : shape.sizes) label += " " + std::to_string(size);
            label += " variant " + std::to_string(variant);
            std::vector<GdnWindowCohortSegment> segments;
            std::vector<std::vector<GdnWindowRow>> together_rows(count), alone_rows(count);
            std::vector<unsigned> first_rows(count), pending(count), keep(count);
            for (std::size_t k = 0; k < count; ++k) {
                auto& session = sessions[k];
                for (auto* side : {&session.together, &session.alone}) {
                    write(side->matrix, session.image); write(side->conv, session.history);
                    for (unsigned r = 0; r < 16U; ++r) {
                        check(hipMemset(side->output[r].address, 0x5a, side->output[r].bytes), "poison cohort GDN output");
                        check(hipMemset(side->workspace[r].address, 0x5a, side->workspace[r].bytes), "poison cohort GDN workspace");
                    }
                    check(hipMemset(side->retained.address, 0x5a, side->retained.bytes), "poison cohort GDN retained rows");
                }
                // A first window of every session alone leaves pending rows in slot 0.
                first_rows[k] = 1U + (k + variant) % 8U;
                pending[k] = (variant + k) % (first_rows[k] + 1U);
                keep[k] = 1U + (variant * 3U + k) % shape.sizes[k];
                control(session.control_together, 0U, 1U); control(session.control_alone, 0U, 1U);
                gdn_window_enqueue(f.layer(session.together), f.rows(session.together, 0U, first_rows[k]),
                    device(session.control_together), f.config, f.owner.stream);
                gdn_window_enqueue(f.layer(session.alone), f.rows(session.alone, 0U, first_rows[k]),
                    device(session.control_alone), f.config, f.owner.stream);
                control(session.control_together, pending[k], 0U); control(session.control_alone, pending[k], 0U);
                together_rows[k] = f.rows(session.together, 8U, shape.sizes[k]);
                alone_rows[k] = f.rows(session.alone, 8U, shape.sizes[k]);
                segments.push_back({f.layer(session.together), together_rows[k], device(session.control_together)});
            }
            f.owner.finish();
            gdn_window_cohort_enqueue(segments, f.config, f.owner.stream);
            for (std::size_t k = 0; k < count; ++k)
                gdn_window_enqueue(f.layer(sessions[k].alone), alone_rows[k], device(sessions[k].control_alone), f.config, f.owner.stream);
            f.owner.finish();
            for (std::size_t k = 0; k < count; ++k) {
                const auto& session = sessions[k];
                const auto what = label + " session " + std::to_string(k);
                for (unsigned r = 8U; r < 8U + shape.sizes[k]; ++r) {
                    require(read(session.together.workspace[r]) == read(session.alone.workspace[r]), what + ": workspace of row " + std::to_string(r) + " differs");
                    require(read(session.together.output[r]) == read(session.alone.output[r]), what + ": output of row " + std::to_string(r) + " differs");
                }
                require(read(session.together.matrix) == read(session.alone.matrix), what + ": recurrent matrix differs after pending rows");
                require(read(session.together.conv) == read(session.alone.conv), what + ": conv history differs after pending rows");
                require(read(session.together.retained) == read(session.alone.retained), what + ": retained rows differ");
            }
            for (std::size_t k = 0; k < count; ++k) {
                auto& session = sessions[k];
                control(session.control_together, keep[k], 1U); control(session.control_alone, keep[k], 1U);
                gdn_window_commit_enqueue(f.layer(session.together), device(session.control_together), f.config, f.owner.stream);
                gdn_window_commit_enqueue(f.layer(session.alone), device(session.control_alone), f.config, f.owner.stream);
            }
            f.owner.finish();
            for (std::size_t k = 0; k < count; ++k) {
                const auto what = label + " session " + std::to_string(k) + " after commit";
                require(read(sessions[k].together.matrix) == read(sessions[k].alone.matrix), what + ": recurrent matrix differs");
                require(read(sessions[k].together.conv) == read(sessions[k].alone.conv), what + ": conv history differs");
            }
            ++cases;
        }
    }
    f.config.dense_a8_projections = dense_before;
    f.owner.guards();
}
float elapsed(hipEvent_t start, hipEvent_t stop) { float ms = 0; check(hipEventElapsedTime(&ms, start, stop), "event time"); return ms; }
template <class F> float median_ms(Owner& owner, F&& run) {
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "event"); check(hipEventCreate(&stop), "event");
    for (unsigned i = 0; i < 5U; ++i) run();
    std::vector<float> samples;
    for (unsigned i = 0; i < 31U; ++i) {
        check(hipEventRecord(start, owner.stream), "record"); run(); check(hipEventRecord(stop, owner.stream), "record");
        check(hipEventSynchronize(stop), "sync"); samples.push_back(elapsed(start, stop));
    }
    (void)hipEventDestroy(start); (void)hipEventDestroy(stop);
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}
void benchmark() {
    for (const auto storage : {GdnRecurrentStorage::key_value, GdnRecurrentStorage::value_key}) {
        Fixture fixture(storage, HipGdnRecurrentLayout::transposed, false, false, 3U);
        fixture.reset();
        const auto serial_layer = fixture.layer(fixture.serial), window_layer = fixture.layer(fixture.window);
        const auto serial_row = fixture.rows(fixture.serial, 0U, 1U)[0];
        const float one = median_ms(fixture.owner, [&] { gdn_window_serial_row_probe(serial_layer, serial_row, fixture.config, fixture.owner.stream); });
        std::cout << (storage == GdnRecurrentStorage::key_value ? "key_value" : "value_key")
                  << " transposed: serial R1 layer " << one * 1000.0F << " us\n";
        for (const unsigned t : {1U, 2U, 4U, 8U}) {
            const auto views = fixture.rows(fixture.window, 0U, t);
            for (const unsigned pending : {0U, (t + 1U) / 2U}) {
                fixture.set_control(pending, 0U);
                const float window = median_ms(fixture.owner, [&] {
                    gdn_window_enqueue(window_layer, views, fixture.device_control(), fixture.config, fixture.owner.stream); });
                std::cout << "  T=" << t << " pending=" << pending << ": window " << window * 1000.0F << " us vs "
                          << t << "x serial " << one * static_cast<float>(t) * 1000.0F << " us (" << one * static_cast<float>(t) / window << "x)\n";
            }
        }
        fixture.set_control(4U, 0U);
        const float commit = median_ms(fixture.owner, [&] {
            gdn_window_commit_enqueue(window_layer, fixture.device_control(), fixture.config, fixture.owner.stream); });
        fixture.set_control(0U, 0U);
        const float empty = median_ms(fixture.owner, [&] {
            gdn_window_commit_enqueue(window_layer, fixture.device_control(), fixture.config, fixture.owner.stream); });
        std::cout << "  commit 4 rows " << commit * 1000.0F << " us, commit 0 rows " << empty * 1000.0F << " us\n";
    }
}
// `bench-cold`: the decode case, where the 36 layers' 113 MB of recurrent state stream through the 32 MB MALL
// between two windows of one layer. A 96 MB write before every sample evicts the layer's 3 MiB matrix; the
// window and commit times are swept over the pending rows (0 = no matrix store). Run it under a kernel trace
// for per-kernel times; the event times printed here include the projections.
void benchmark_cold() {
    Fixture fixture(GdnRecurrentStorage::key_value, HipGdnRecurrentLayout::transposed, false, false, 3U);
    fixture.reset();
    const auto window_layer = fixture.layer(fixture.window);
    void* flush = nullptr;
    constexpr std::size_t flush_bytes = 96ULL << 20U;
    check(hipMalloc(&flush, flush_bytes), "allocate MALL flush buffer");
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "event"); check(hipEventCreate(&stop), "event");
    const auto cold_ms = [&](auto&& run) {
        std::vector<float> samples;
        for (unsigned i = 0; i < 25U; ++i) {
            check(hipMemsetAsync(flush, static_cast<int>(i), flush_bytes, fixture.owner.stream), "MALL flush");
            check(hipEventRecord(start, fixture.owner.stream), "record"); run(); check(hipEventRecord(stop, fixture.owner.stream), "record");
            check(hipEventSynchronize(stop), "sync"); samples.push_back(elapsed(start, stop));
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2U] * 1000.0F;
    };
    for (const unsigned t : {2U, 4U, 8U}) {
        const auto views = fixture.rows(fixture.window, 0U, t);
        for (const unsigned pending : {0U, 1U, 2U, 4U, 8U}) {
            fixture.set_control(pending, 0U);
            std::cout << "cold T=" << t << " pending=" << pending << ": window "
                      << cold_ms([&] { gdn_window_enqueue(window_layer, views, fixture.device_control(), fixture.config, fixture.owner.stream); })
                      << " us\n";
        }
    }
    for (const unsigned pending : {0U, 1U, 4U, 8U}) {
        fixture.set_control(pending, 0U);
        std::cout << "cold commit pending=" << pending << ": "
                  << cold_ms([&] { gdn_window_commit_enqueue(window_layer, fixture.device_control(), fixture.config, fixture.owner.stream); })
                  << " us\n";
    }
    (void)hipEventDestroy(start); (void)hipEventDestroy(stop);
    check(hipFree(flush), "free MALL flush buffer");
}
// `bench-gates`: windows of T = 2, 4, 8 rows with nothing pending, warm (back to back) and cold (MALL evicted),
// 25 samples each (event times). Run it under a kernel trace with STRIX_GDN_GATES_ANYORDER=1
// and =0: the trace gives the Q8 input projection, the F32 gates and the next kernel's start per window.
void benchmark_gates() {
    Fixture fixture(GdnRecurrentStorage::key_value, HipGdnRecurrentLayout::transposed, false, false, 3U);
    fixture.reset();
    const auto window_layer = fixture.layer(fixture.window);
    void* flush = nullptr;
    constexpr std::size_t flush_bytes = 96ULL << 20U;
    check(hipMalloc(&flush, flush_bytes), "allocate MALL flush buffer");
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "event"); check(hipEventCreate(&stop), "event");
    fixture.set_control(0U, 0U);
    for (const bool cold : {false, true})
        for (const unsigned t : {2U, 4U, 8U}) {
            const auto views = fixture.rows(fixture.window, 0U, t);
            // Untimed warm-up of this specialization (first launch, code object load).
            gdn_window_enqueue(window_layer, views, fixture.device_control(), fixture.config, fixture.owner.stream);
            check(hipStreamSynchronize(fixture.owner.stream), "warm-up");
            std::vector<float> samples;
            for (unsigned i = 0; i < 25U; ++i) {
                if (cold) check(hipMemsetAsync(flush, static_cast<int>(i), flush_bytes, fixture.owner.stream), "MALL flush");
                check(hipEventRecord(start, fixture.owner.stream), "record");
                gdn_window_enqueue(window_layer, views, fixture.device_control(), fixture.config, fixture.owner.stream);
                check(hipEventRecord(stop, fixture.owner.stream), "record");
                check(hipEventSynchronize(stop), "sync"); samples.push_back(elapsed(start, stop));
            }
            std::sort(samples.begin(), samples.end());
            std::cout << (cold ? "cold" : "warm") << " T=" << t << ": window " << samples[samples.size() / 2U] * 1000.0F << " us\n";
        }
    (void)hipEventDestroy(start); (void)hipEventDestroy(stop);
    check(hipFree(flush), "free MALL flush buffer");
}
// `bench-state warm|cold`: recurrent-state traffic of the decode recurrences (incumbent storage), for a kernel
// trace. Cases run in order, 3 + 25 launches each, separated by a 60 ms host gap so a trace splits per case;
// cold flushes the MALL (96 MiB memset) before every sample. Prints "case <name> <event median us>".
void benchmark_state(bool cold, RecurrentStateType type = RecurrentStateType::f32) {
    Fixture f(GdnRecurrentStorage::key_value, HipGdnRecurrentLayout::baseline, true, true, 5U, type);
    f.reset();
    void* flush = nullptr;
    constexpr std::size_t flush_bytes = 96ULL << 20U;
    check(hipMalloc(&flush, flush_bytes), "allocate MALL flush buffer");
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "event"); check(hipEventCreate(&stop), "event");
    unsigned sample = 0;
    const auto run_case = [&](const std::string& name, auto&& run) {
        f.owner.finish();
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        for (unsigned i = 0; i < 3U; ++i) run();
        std::vector<float> samples;
        for (unsigned i = 0; i < 25U; ++i) {
            if (cold) check(hipMemsetAsync(flush, static_cast<int>(++sample & 0xffU), flush_bytes, f.owner.stream), "MALL flush");
            check(hipEventRecord(start, f.owner.stream), "record"); run(); check(hipEventRecord(stop, f.owner.stream), "record");
            check(hipEventSynchronize(stop), "sync"); samples.push_back(elapsed(start, stop));
        }
        std::sort(samples.begin(), samples.end());
        std::cout << "case " << name << " " << samples[samples.size() / 2U] * 1000.0F << " us\n";
    };
    const auto serial_layer = f.layer(f.serial), window_layer = f.layer(f.window);
    const auto serial_row = f.rows(f.serial, 0U, 1U)[0];
    run_case("r1", [&] { gdn_window_serial_row_probe(serial_layer, serial_row, f.config, f.owner.stream); });
    for (const unsigned t : {1U, 2U, 8U})
        for (const unsigned pending : {0U, t}) {
            f.set_control(pending, 0U);
            const auto views = f.rows(f.window, 0U, t);
            run_case("w" + std::to_string(t) + "p" + std::to_string(pending),
                [&] { gdn_window_enqueue(window_layer, views, f.device_control(), f.config, f.owner.stream); });
        }
    f.set_control(2U, 0U);
    run_case("commit2", [&] { gdn_window_commit_enqueue(window_layer, f.device_control(), f.config, f.owner.stream); });
    // Eight chats of one layer (C8 cohort): own state, conv history, retained rows and control each.
    std::vector<Side> chats(8);
    std::vector<Buffer> controls;
    for (auto& chat : chats) {
        chat.matrix = f.owner.allocate(f.state_image0.size()); write(chat.matrix, f.state_image0);
        chat.conv = f.owner.allocate(f.history0.size() * sizeof(std::uint16_t)); write(chat.conv, f.history0);
        chat.retained = f.owner.allocate(static_cast<std::size_t>(gdn_window_retained_bytes(f.config.model)));
        for (unsigned row = 0; row < 2U; ++row) {
            chat.input.push_back(f.window.input[row]);
            chat.output.push_back(f.owner.allocate(hidden * sizeof(std::uint16_t)));
            chat.workspace.push_back(f.owner.allocate(workspace_floats * sizeof(float)));
        }
        controls.push_back(f.owner.allocate(sizeof(GdnWindowControl)));
        check(hipMemset(controls.back().address, 0, sizeof(GdnWindowControl)), "zero cohort control");
    }
    for (const unsigned rows : {1U, 2U}) {
        std::vector<std::vector<GdnWindowRow>> views;
        for (const auto& chat : chats) views.push_back(f.rows(chat, 0U, rows));
        std::vector<GdnWindowCohortSegment> segments;
        for (std::size_t k = 0; k < chats.size(); ++k)
            segments.push_back({f.layer(chats[k]), views[k], reinterpret_cast<const GdnWindowControl*>(controls[k].address)});
        // 8 chats x 2 rows = 16 rows: the wide cohort forms need dense A8 (restored after the case).
        const bool dense_before = f.config.dense_a8_projections;
        if (rows == 2U) f.config.dense_a8_projections = true;
        run_case("c8r" + std::to_string(rows), [&] { gdn_window_cohort_enqueue(segments, f.config, f.owner.stream); });
        f.config.dense_a8_projections = dense_before;
    }
    f.owner.finish();
    (void)hipEventDestroy(start); (void)hipEventDestroy(stop);
    check(hipFree(flush), "free MALL flush buffer");
}
// "state16 codec": the device store / load of rstate::Codec<f16|bf16> against the host reference, bit for bit,
// over a sweep of float bit patterns (every upper half x lower halves around the rounding ties, NaN, inf, subnormals).
template <RecurrentStateType T>
__global__ void state16_codec_kernel(const float* in, std::uint16_t* narrowed, float* widened, std::size_t count) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto half = rstate::Codec<T>::store(in[i]);
    narrowed[i] = half; widened[i] = rstate::Codec<T>::load(half);
}
template <RecurrentStateType T> void state16_codec_case(Owner& owner, const std::vector<float>& values) {
    const auto count = values.size();
    auto in = owner.allocate(count * 4U), narrowed = owner.allocate(count * 2U), widened = owner.allocate(count * 4U);
    write(in, values);
    hipLaunchKernelGGL(state16_codec_kernel<T>, dim3(static_cast<unsigned>((count + 255U) / 256U)), dim3(256), 0, owner.stream,
        reinterpret_cast<const float*>(in.address), reinterpret_cast<std::uint16_t*>(narrowed.address),
        reinterpret_cast<float*>(widened.address), count);
    check(hipGetLastError(), "state16 codec launch"); owner.finish();
    const auto n = read(narrowed), w = read(widened);
    std::size_t bad = 0;
    for (std::size_t i = 0; i < count; ++i) {
        std::uint16_t got = 0; float wide = 0;
        std::memcpy(&got, n.data() + 2U * i, 2U); std::memcpy(&wide, w.data() + 4U * i, 4U);
        const std::uint16_t want = T == RecurrentStateType::f16 ? rstate::f16_from_f32_host(values[i]) : rstate::bf16_from_f32(values[i]);
        const float want_wide = T == RecurrentStateType::f16 ? rstate::f32_from_f16_host(want) : rstate::f32_from_bf16(want);
        if (got != want || std::bit_cast<std::uint32_t>(wide) != std::bit_cast<std::uint32_t>(want_wide)) {
            if (bad++ < 4U)
                std::cerr << "state16 codec " << recurrent_state_type_name(T) << " mismatch for 0x" << std::hex << std::bit_cast<std::uint32_t>(values[i])
                          << " device 0x" << got << " host 0x" << want << std::dec << '\n';
        }
    }
    require(bad == 0U, std::string("state16 codec ") + recurrent_state_type_name(T) + ": device differs from the host codec");
}
void state16_codec_tests(unsigned& cases) {
    std::vector<float> values;
    const std::uint32_t lows[] = {0x0000U, 0x0001U, 0x0fffU, 0x1000U, 0x1001U, 0x1fffU, 0x2000U, 0x3000U, 0x5555U, 0x7fffU, 0x8000U,
        0x8001U, 0xa000U, 0xaaaaU, 0xe000U, 0xffffU};
    for (std::uint32_t high = 0; high < 65536U; ++high)
        for (const auto low : lows) values.push_back(std::bit_cast<float>((high << 16U) | low));
    Rng rng{4242U};
    for (std::size_t i = 0; i < 65536U * 8U; ++i) values.push_back(std::bit_cast<float>(rng.next() ^ (rng.next() << 16U)));
    Owner owner;
    state16_codec_case<RecurrentStateType::f16>(owner, values);
    state16_codec_case<RecurrentStateType::bf16>(owner, values);
    owner.guards();
    std::cout << "state16 codec: " << values.size() << " patterns x {f16, bf16} bit-identical to the host codec\n";
    ++cases;
}
// Report only: 16 serial rows from the same inputs, 16-bit storage against FP32.
void state16_drift(RecurrentStateType type) {
    Fixture f32(GdnRecurrentStorage::key_value, HipGdnRecurrentLayout::baseline, false, false, 9U);
    Fixture f16(GdnRecurrentStorage::key_value, HipGdnRecurrentLayout::baseline, false, false, 9U, type);
    f32.reset(); f16.reset();
    f32.serial_rows(0U, 16U); f16.serial_rows(0U, 16U);
    const auto outputs = [](const Fixture& f) {
        std::vector<float> values;
        for (unsigned r = 0; r < 16U; ++r) {
            const auto bytes = read(f.serial.output[r]);
            for (std::size_t i = 0; i < hidden; ++i) {
                std::uint16_t half = 0; std::memcpy(&half, bytes.data() + 2U * i, 2U);
                values.push_back(std::bit_cast<float>(static_cast<std::uint32_t>(half) << 16U));
            }
        }
        return values;
    };
    const auto drift = [](const std::vector<float>& reference, const std::vector<float>& value) {
        double sum = 0, worst = 0;
        for (std::size_t i = 0; i < reference.size(); ++i) {
            sum += static_cast<double>(reference[i]) * reference[i];
            worst = std::max(worst, std::abs(static_cast<double>(value[i]) - reference[i]));
        }
        const double rms = std::sqrt(sum / static_cast<double>(reference.size()));
        return rms > 0 ? worst / rms : 0.0;
    };
    const auto state32 = state_values(read(f32.serial.matrix), state_floats, RecurrentStateType::f32);
    const auto state16 = state_values(read(f16.serial.matrix), state_floats, type);
    std::cout << "state16 drift " << recurrent_state_type_name(type) << " out " << drift(outputs(f32), outputs(f16))
              << " state " << drift(state32, state16) << '\n';
}
// 16-bit state is typed only on the baseline key-value organization: the others must refuse, not run FP32 code on 16-bit data.
void state16_refusals(unsigned& cases) {
    for (const auto type : {RecurrentStateType::f16, RecurrentStateType::bf16})
        for (const auto [storage, layout] : {std::pair{GdnRecurrentStorage::value_key, HipGdnRecurrentLayout::baseline},
                                             std::pair{GdnRecurrentStorage::key_value, HipGdnRecurrentLayout::transposed}}) {
            Fixture f(storage, layout, false, false, 11U, type);
            f.reset(); f.set_control(0U, 1U);
            bool thrown = false;
            try { f.window_rows(0U, 2U, false); } catch (const std::invalid_argument&) { thrown = true; }
            require(thrown, std::string("state16 ") + recurrent_state_type_name(type) + ": unsupported organization was not refused");
            ++cases;
        }
}
RecurrentStateType parse_state_type(std::string_view name) {
    if (name == "f32") return RecurrentStateType::f32;
    if (name == "f16") return RecurrentStateType::f16;
    if (name == "bf16") return RecurrentStateType::bf16;
    throw std::invalid_argument("state type must be f32, f16 or bf16");
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc > 3 && std::string_view(argv[1]) == "bench-state16") {
            benchmark_state(std::string_view(argv[3]) == "cold", parse_state_type(argv[2])); return 0;
        }
        if (argc > 2 && std::string_view(argv[1]) == "bench-state") { benchmark_state(std::string_view(argv[2]) == "cold"); return 0; }
        if (argc > 1 && std::string_view(argv[1]) == "bench") { benchmark(); return 0; }
        if (argc > 1 && std::string_view(argv[1]) == "bench-cold") { benchmark_cold(); return 0; }
        if (argc > 1 && std::string_view(argv[1]) == "bench-gates") { benchmark_gates(); return 0; }
        unsigned cases = 0;
        struct Variant { GdnRecurrentStorage storage; HipGdnRecurrentLayout layout; bool q8, bf16_small; };
        for (const auto variant : {Variant{GdnRecurrentStorage::key_value, HipGdnRecurrentLayout::baseline, false, false},
                                   Variant{GdnRecurrentStorage::key_value, HipGdnRecurrentLayout::transposed, false, false},
                                   Variant{GdnRecurrentStorage::value_key, HipGdnRecurrentLayout::baseline, false, false},
                                   Variant{GdnRecurrentStorage::value_key, HipGdnRecurrentLayout::transposed, true, true}}) {
            Fixture fixture(variant.storage, variant.layout, variant.q8, variant.bf16_small, cases + 1U);
            for (const unsigned t : {1U, 2U, 3U, 4U, 8U})
                for (unsigned n = 0; n <= t; ++n) { fixture.scenario(t, n, (t - n + 1U) % (t + 1U), n == 1U); ++cases; }
            // Owners of different sizes sharing one pending record (adaptive verifier).
            for (const auto [t, u] : {std::pair{2U, 4U}, std::pair{4U, 2U}, std::pair{8U, 2U}, std::pair{2U, 8U}, std::pair{4U, 8U}, std::pair{8U, 4U}})
                for (unsigned n = 0; n <= t; ++n) { fixture.scenario(t, n, (n + 1U) % (u + 1U), n == 2U, u); ++cases; }
            // Several sessions in one cohort window, each against its own window.
            cohort_scenarios(fixture, cases);
        }
        // 16-bit recurrent state (FAST storage): the same window-vs-serial and cohort-vs-alone bitwise equalities
        // under each type (the bits differ from the f32 run, the routes agree with each other), codec pins, drift.
        state16_codec_tests(cases);
        for (const auto type : {RecurrentStateType::f16, RecurrentStateType::bf16}) {
            for (const auto [q8, bf16_small] : {std::pair{false, false}, std::pair{true, true}}) {
                Fixture fixture(GdnRecurrentStorage::key_value, HipGdnRecurrentLayout::baseline, q8, bf16_small, cases + 1U, type);
                for (const unsigned t : {1U, 2U, 3U, 4U, 8U})
                    for (unsigned n = 0; n <= t; ++n) { fixture.scenario(t, n, (t - n + 1U) % (t + 1U), n == 1U); ++cases; }
                for (const auto [t, u] : {std::pair{2U, 4U}, std::pair{4U, 2U}, std::pair{8U, 2U}, std::pair{2U, 8U}, std::pair{4U, 8U}, std::pair{8U, 4U}})
                    for (unsigned n = 0; n <= t; ++n) { fixture.scenario(t, n, (n + 1U) % (u + 1U), n == 2U, u); ++cases; }
                cohort_scenarios(fixture, cases);
            }
            state16_drift(type);
        }
        state16_refusals(cases);
        std::cout << "deferred GDN window: " << cases << " scenarios bit-exact (outputs, workspaces, matrix, conv history)\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
