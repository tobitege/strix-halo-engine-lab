#include "strix/dispatch_resources.hpp"
#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_plan.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/hip_shared_expert_launcher.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strix;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}
struct View { std::uintptr_t address = 0; std::uint64_t bytes = 0; };
std::vector<std::byte> read(View view) {
    std::vector<std::byte> bytes(static_cast<std::size_t>(view.bytes));
    check(hipMemcpy(bytes.data(), reinterpret_cast<const void*>(view.address), bytes.size(), hipMemcpyDeviceToHost), "read guarded rows view");
    return bytes;
}
void write(View view, const std::vector<std::byte>& bytes) {
    require(view.bytes == bytes.size(), "rows upload extent differs");
    check(hipMemcpy(reinterpret_cast<void*>(view.address), bytes.data(), bytes.size(), hipMemcpyHostToDevice), "upload guarded rows view");
}
template<class T> std::vector<std::byte> bytes_of(const std::vector<T>& values) {
    std::vector<std::byte> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size()); return bytes;
}
struct Owner {
    HipSetupQuiescenceGuard guard;
    hipStream_t stream = nullptr;
    std::vector<void*> allocations;
    Owner() { guard.run([&] { check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create shared rows stream"); }); }
    ~Owner() {
        guard.recover_or_fail_stop();
        for (auto* p : allocations) if (hipFree(p) != hipSuccess) std::_Exit(3);
        if (hipStreamDestroy(stream) != hipSuccess) std::_Exit(3);
    }
    View allocate(std::uint64_t bytes) {
        require(bytes && bytes < (1ULL << 28U), "rows fixture allocation exceeds bound");
        void* pointer = nullptr;
        check(hipMalloc(&pointer, static_cast<std::size_t>(bytes + 512U)), "allocate shared rows owner");
        try { allocations.push_back(pointer); }
        catch (...) { guard.recover_or_fail_stop(); static_cast<void>(hipFree(pointer)); throw; }
        check(hipMemset(pointer, 0xa5, static_cast<std::size_t>(bytes + 512U)), "poison shared rows owner");
        return {reinterpret_cast<std::uintptr_t>(pointer) + 256U, bytes};
    }
    void finish() { check(hipStreamSynchronize(stream), "drain shared rows stream"); guard.require_original(); }
};
void guards(View view) {
    for (const auto address : {view.address - 256U, view.address + view.bytes}) {
        const auto bytes = read({address, 256U});
        require(std::all_of(bytes.begin(), bytes.end(), [](auto b) { return b == std::byte{0xa5}; }), "rows outer guard changed");
    }
}
std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}
struct Q8 { __half scale; std::int8_t codes[32]; };
static_assert(sizeof(Q8) == 34);
std::vector<std::byte> q8(unsigned width, unsigned outputs, unsigned salt) {
    std::vector<Q8> blocks(static_cast<std::size_t>(width / 32U) * outputs);
    constexpr std::array<std::uint16_t, 7> scale_bits{0x0000,0x8000,0x0001,0x0400,0x1800,0x1c00,0x2000};
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        const auto bits = scale_bits[(b + salt) % scale_bits.size()];
        std::memcpy(&blocks[b].scale, &bits, sizeof(bits));
        for (unsigned c = 0; c < 32; ++c)
            blocks[b].codes[c] = static_cast<std::int8_t>(static_cast<int>((b * 17U + c * 31U + salt) % 256U) - 128);
    }
    return bytes_of(blocks);
}
struct Fixture {
    Owner owner; // outlives every borrowing descriptor and outstanding enqueue
    ModelContract model = ModelContract::qwen38_flash_next();
    SerialExecutionPlan execution = build_serial_decode_execution_plan(model);
    SerialExecutionIoPlan io = build_serial_execution_io_plan(execution);
    ExecutionStep step;
    ExecutionArenaPlan serial;
    KnownWindowArena window;
    View scratch;
    ExecutionMemoryBinding memory;
    ModelManifest manifest;
    ModelRuntimeBinding weights;
    ExecutionTensorAccessPlan access;
    std::vector<View> weight_views;
    std::vector<std::vector<std::byte>> original_weights;
    std::vector<ExecutionRuntimeResources> resources;
    std::vector<ExecutionDispatchContext> contexts;
    std::vector<ExecutionBufferAddress> input, output, temporary;
    HipSharedExpertLauncherConfig config;
    std::string serial_id;
    ExecutionLauncher incumbent;
    std::vector<std::byte> seed;

    Fixture(unsigned rows, bool prefetch) {
        require(owner.guard.architecture() == "gfx1151", "shared rows fixture requires actual gfx1151");
        const auto found = std::find_if(execution.steps.begin(), execution.steps.end(),
            [](const auto& value) { return value.op == SerialOp::shared_expert && value.layer == 0U; });
        require(found != execution.steps.end(), "shared canonical step missing"); step = *found;
        const auto& row_io = io.steps[step.id];
        serial.alignment = 256;
        const auto allocation = [&](ExecutionBufferUse use, std::uint64_t bytes) {
            const auto offset = (serial.total_bytes + 255U) / 256U * 256U;
            serial.allocations.push_back({use, step.id, step.id, offset, bytes}); serial.total_bytes = offset + bytes;
        };
        allocation(row_io.reads[0], 5120); allocation(row_io.writes[0], 5120); allocation(row_io.temporaries[0], 5120);
        serial.naive_nonaliased_bytes = serial.total_bytes;
        window = rows >= 16U ? build_known_prefill_arena(serial, 256U, rows) : build_known_window_arena(serial, 256U, rows);
        scratch = owner.allocate(window.physical.total_bytes);
        memory = {0x5253484152454401ULL, scratch.address, scratch.bytes, 23};
        manifest.architecture = "qwen4_exp";
        add("blk.0.shared_expert.gate_proj.weight", 8, {2560,640}, q8(2560,640,1));
        add("blk.0.shared_expert.up_proj.weight", 8, {2560,640}, q8(2560,640,2));
        add("blk.0.shared_expert.down_proj.weight", 8, {640,2560}, q8(640,2560,3));
        std::vector<float> scalar(2560);
        for (std::size_t i = 0; i < scalar.size(); ++i) scalar[i] = static_cast<float>(static_cast<int>(i % 7U) - 3) * 0.0001F;
        add("blk.0.shared_expert_gate.weight", 0, {2560}, bytes_of(scalar));
        manifest.layout_identity = model_manifest_layout_identity(manifest);
        weights.manifest_layout_identity = manifest.layout_identity; weights.model_content_identity.bytes[0] = 1;
        access.manifest_layout_identity = manifest.layout_identity; access.plan_identity = 0x5253484152454402ULL;
        access.accesses.push_back({step.id, step.op, 0, {0,1,2,3}});
        config.hidden_size = 2560; config.intermediate_size = 640; config.prefetch_q8 = prefetch;
        if (prefetch) config.prefetch_eligible_layers = {0};
        kernels::KernelRegistry registry;
        ExecutionBindingRequest request;
        request.prefetch_q8_shared_expert = prefetch; request.shared_expert_prefetch_layers = config.prefetch_eligible_layers;
        register_hip_shared_expert_kernel_variant(model, request, config, registry);
        serial_id = registry.variants().front().id; incumbent = make_hip_shared_expert_launcher(serial_id, &config);
        contexts.resize(rows); resources.resize(rows); input.resize(rows); output.resize(rows); temporary.resize(rows);
        for (unsigned r = 0; r < rows; ++r) {
            resources[r] = {&memory, &manifest, &weights, &access};
            auto& c = contexts[r];
            c.execution_identity = memory.execution_identity; c.step_id = step.id; c.op = step.op; c.layer = step.layer;
            c.kernel_id = serial_id; c.reads = row_io.reads; c.writes = row_io.writes; c.temporaries = row_io.temporaries;
            c.arena = &window.row_view[r]; c.memory = &memory; c.resources = &resources[r];
            c.position = 17U + r; c.input_token_id = 101U + r;
            c.reads_state = step.reads_state; c.writes_state = step.writes_state;
            input[r] = resolve_dispatch_read(c, 0); output[r] = resolve_dispatch_write(c, 0); temporary[r] = resolve_dispatch_temporary(c, 0);
        }
    }
    void add(std::string name, unsigned type, std::vector<std::uint64_t> dims, std::vector<std::byte> data) {
        ManifestTensor t; t.name = std::move(name); t.layer = 0; t.role = TensorRole::dense_projection;
        t.ggml_type = type; t.dimensions = std::move(dims); t.absolute_offset = manifest.total_tensor_bytes;
        t.byte_size = data.size(); manifest.total_tensor_bytes += t.byte_size;
        const auto device = owner.allocate(data.size()); write(device, data);
        const auto index = static_cast<std::uint32_t>(weight_views.size());
        weight_views.push_back(device); original_weights.push_back(std::move(data)); manifest.tensors.push_back(std::move(t));
        weights.tensors.push_back({index, RuntimeTensorStorage::resident, device.address, device.bytes, 31, 0});
    }
    HipLaunchContext scope() const { HipLaunchContext s; s.stream = owner.stream; return s; }
    void prepare(unsigned kind, bool reverse) {
        seed.assign(static_cast<std::size_t>(scratch.bytes), std::byte{0xa5});
        constexpr std::array<std::uint16_t, 12> special{0,0x8000,1,0x8001,0x007f,0x807f,0x3f80,0xbf80,0x7f80,0xff80,0x7fc1,0xff83};
        for (std::size_t r = 0; r < contexts.size(); ++r) {
            const auto tag = reverse ? contexts.size() - 1U - r : r;
            std::vector<std::uint16_t> values(2560);
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = kind ? special[(i + tag * 5U) % special.size()] :
                    bf16(static_cast<float>(static_cast<int>((i * 29U + tag * 43U) % 97U) - 48) / static_cast<float>(97U + tag));
            std::memcpy(seed.data() + input[r].address - scratch.address, values.data(), input[r].bytes);
        }
    }
    std::vector<std::byte> run(bool wide) {
        write(scratch, seed);
        const auto current = scope();
        { const HipLaunchScope active(current);
          if (wide) {
              shared_expert_known_window_rows_preflight(contexts, config);
              require(read(scratch) == seed, "pure shared rows preflight changed payload");
              shared_expert_known_window_rows_launch(contexts, config);
          } else for (const auto& c : contexts) { incumbent.preflight(c, incumbent.user_data); incumbent.function(c, incumbent.user_data); } }
        owner.finish();
        const auto result = read(scratch);
        auto masked = result;
        for (std::size_t r = 0; r < contexts.size(); ++r)
            for (const auto view : {output[r], temporary[r]})
                std::copy_n(seed.begin() + static_cast<std::ptrdiff_t>(view.address - scratch.address),
                    static_cast<std::size_t>(view.bytes), masked.begin() + static_cast<std::ptrdiff_t>(view.address - scratch.address));
        require(masked == seed, "shared rows changed input or internal guard bytes");
        guards(scratch);
        for (std::size_t w = 0; w < weight_views.size(); ++w) {
            require(read(weight_views[w]) == original_weights[w], "shared rows changed a weight owner"); guards(weight_views[w]);
        }
        return result;
    }
    struct PreResult { std::vector<std::byte> arena; unsigned overflow; std::vector<std::uint16_t> a16; };
    // The FP32 gate|up / activation floats of the shared-pre rows have no reader (their down reads the FP16 preimage,
    // their gate is float 640): the producers may leave them unwritten. Zero them before comparing routes.
    std::vector<std::byte> mask_dead(std::vector<std::byte> arena, unsigned pre_rows) const {
        for (unsigned r = 0; r < pre_rows && r < contexts.size(); ++r) {
            const auto at = static_cast<std::size_t>(temporary[r].address - scratch.address);
            for (unsigned i = 0; i < 1280U; ++i)
                if (i != 640U) std::fill_n(arena.begin() + static_cast<std::ptrdiff_t>(at + i * sizeof(float)), sizeof(float), std::byte{0});
        }
        return arena;
    }
    // poison: the output and temporary views start from this byte (the rest of the arena from the seed), so a store
    // missing in both routes cannot hide behind a common initial value (compare two poisons, see written_equal).
    // Explicit producers ignore inherited form/pair settings: 0 pre-kernel, 1 paired form 4, 2 paired form 0.
    PreResult run_pre(bool pre, const float* injection, hipStream_t stream, bool admitted = true, int poison = -1, int pair = 1,
        SharedPreFallback fallback = SharedPreFallback::none) {
        auto start = seed;
        if (poison >= 0)
            for (std::size_t r = 0; r < contexts.size(); ++r)
                for (const auto view : {output[r], temporary[r]})
                    std::fill_n(start.begin() + static_cast<std::ptrdiff_t>(view.address - scratch.address), static_cast<std::size_t>(view.bytes),
                        static_cast<std::byte>(poison));
        write(scratch, start);
        const auto before = shared_expert_shared_pre_fixture_overflow_count();
        auto current = scope(); current.stream = stream;
        SharedPreFixtureRoute route;
        const unsigned requested_form = pair == 2 ? 0U : 4U;
        { const HipLaunchScope active(current);
          require(shared_expert_shared_pre_fixture_launch(contexts, config, pre, injection, pair, requested_form, &route) == (pre && admitted),
              "shared pre fixture did not take the requested admitted route"); }
        const auto expected_producer = !pre || !admitted ? SharedPreProducer::complete_activation
            : pair == 0 ? SharedPreProducer::pre_kernel : SharedPreProducer::paired;
        const auto expected_fallback = !pre ? SharedPreFallback::disabled
            : !admitted ? fallback : pair == 0 ? SharedPreFallback::pair_disabled : SharedPreFallback::none;
        require(route.producer == expected_producer, "shared pre producer was not exercised");
        require(route.fallback == expected_fallback, "shared pre fallback reason changed");
        require(route.form == (expected_producer == SharedPreProducer::paired ? requested_form : ~0U),
            "shared pre effective paired form was not exercised");
        check(hipStreamSynchronize(stream), "complete shared pre fixture");
        const auto delta = shared_expert_shared_pre_fixture_overflow_count() - before;
        auto result = read(scratch), masked = result;
        for (std::size_t r = 0; r < contexts.size(); ++r)
            for (const auto view : {output[r], temporary[r]}) {
                const auto at = static_cast<std::ptrdiff_t>(view.address - scratch.address);
                std::copy_n(seed.begin() + at, static_cast<std::size_t>(view.bytes), masked.begin() + at);
            }
        require(masked == start || poison >= 0, "shared pre changed input or internal guard bytes");
        if (poison >= 0) {
            for (std::size_t r = 0; r < contexts.size(); ++r)
                for (const auto view : {output[r], temporary[r]}) {
                    const auto at = static_cast<std::ptrdiff_t>(view.address - scratch.address);
                    std::copy_n(start.begin() + at, static_cast<std::size_t>(view.bytes), masked.begin() + at);
                }
            require(masked == start, "shared pre changed input or internal guard bytes");
        }
        guards(scratch);
        for (std::size_t w = 0; w < weight_views.size(); ++w) {
            require(read(weight_views[w]) == original_weights[w], "shared pre changed weight bytes"); guards(weight_views[w]);
        }
        std::vector<std::uint16_t> a16;
        if (pre && admitted)
            a16 = shared_expert_shared_pre_fixture_a16(stream, static_cast<std::uint32_t>(contexts.size() / 32U * 32U), 640U);
        return {std::move(result), delta, std::move(a16)};
    }
    // The window owner's detached form: a second stream and a private workspace at the arena's row
    // stride. Returns the arena run's bytes with the workspace rows read from the private copy; the
    // arena workspace keeps its seed.
    std::vector<std::byte> run_detached() {
        write(scratch, seed);
        const auto stride = temporary[1].address - temporary[0].address;
        const auto workspace = owner.allocate(stride * (contexts.size() - 1U) + temporary.back().bytes);
        hipStream_t side = nullptr;
        check(hipStreamCreateWithFlags(&side, hipStreamNonBlocking), "create detached shared stream");
        try {
            shared_expert_known_window_rows_launch_detached(contexts, config, side, workspace.address, workspace.bytes);
            check(hipStreamSynchronize(side), "complete detached shared rows");
        } catch (...) { static_cast<void>(hipStreamSynchronize(side)); static_cast<void>(hipStreamDestroy(side)); throw; }
        check(hipStreamDestroy(side), "destroy detached shared stream");
        auto result = read(scratch);
        for (std::size_t r = 0; r < contexts.size(); ++r) {
            const auto at = static_cast<std::ptrdiff_t>(temporary[r].address - scratch.address);
            require(std::equal(seed.begin() + at, seed.begin() + at + static_cast<std::ptrdiff_t>(temporary[r].bytes), result.begin() + at),
                "detached shared rows wrote the arena workspace");
            const auto copied = read({workspace.address + r * stride, temporary[r].bytes});
            std::copy(copied.begin(), copied.end(), result.begin() + at);
        }
        guards(scratch); guards(workspace);
        return result;
    }
    template<class Mutate, class Restore> void rejected(Mutate mutate, Restore restore, bool capture,
        std::string_view expected, bool expect_overflow = false) {
        write(scratch, seed); const auto prior_weights = original_weights;
        auto current = scope(); hipGraph_t graph = nullptr;
        bool rejected = false; std::string reason;
        mutate();
        { const HipLaunchScope active(current);
          if (capture) check(hipStreamBeginCapture(owner.stream, hipStreamCaptureModeThreadLocal), "begin wide shared rejection capture");
          try { shared_expert_known_window_rows_launch(contexts, config); }
          catch (const std::invalid_argument& error) { rejected = !expect_overflow; reason = error.what(); }
          catch (const std::overflow_error& error) { rejected = expect_overflow; reason = error.what(); }
          if (capture) check(hipStreamEndCapture(owner.stream, &graph), "end wide shared rejection capture"); }
        restore();
        if (graph) {
            std::size_t nodes = 0; check(hipGraphGetNodes(graph, nullptr, &nodes), "count wide shared rejected nodes");
            require(nodes == 0, "rejected wide shared admission captured nodes");
            check(hipGraphDestroy(graph), "destroy wide shared rejected graph");
        }
        owner.finish();
        const bool unchanged = read(scratch) == seed;
        if (!rejected || reason.find(expected) == std::string::npos || !unchanged)
            throw std::runtime_error("wide shared rejection expected [" + std::string(expected) +
                "] but received [" + reason + "]; rejected=" + std::to_string(rejected) +
                "; payload_unchanged=" + std::to_string(unchanged));
        for (std::size_t w = 0; w < weight_views.size(); ++w)
            require(read(weight_views[w]) == prior_weights[w], "wide shared rejection mutated weights");
    }
};
// Dedicated mode admits the short tiled prefixes too; thresholds can still be tested by the ordinary entry.
void fixture_environment(const char* name, const char* value) {
#if defined(_WIN32)
    require(_putenv_s(name, value) == 0, "set shared pre fixture environment");
#else
    require(setenv(name, value, 1) == 0, "set shared pre fixture environment");
#endif
}
bool fixture_flag(const char* name) { const char* v = std::getenv(name); return v && std::string_view(v) == "1"; }
std::vector<float> gate_up_values(unsigned rows, unsigned kind) {
    std::vector<float> values(static_cast<std::size_t>(rows) * 1280U);
    std::mt19937 rng(0x5132U);
    constexpr std::array<std::uint32_t, 4> zeros{0U, 0x80000000U, 0x80000000U, 0U};
    constexpr std::array<std::uint32_t, 12> subnormal{1U, 0x80000001U, 0x007fffffU, 0x807fffffU,
        0x30ffffffU, 0x31000000U, 0x31000001U, 0xb0ffffffU, 0xb1000000U, 0xb1000001U,
        0x31800000U, 0xb1800000U};  // gate 32 (act = 32 up exactly): FP32 subnormals, min FP16 subnormal 2^-24 +/- neighbours
    constexpr std::array<std::uint32_t, 6> zero_tie{0x307fffffU, 0x30800000U, 0x30800001U, 0xb07fffffU, 0xb0800000U, 0xb0800001U};  // act 2^-25 (FP16 zero tie) +/- 1 ulp
    // Each boundary, its two float neighbours, both signs; counting is >65504 even below the 65520 RNE boundary.
    const std::array<float, 12> boundary{std::nextafter(65504.0F, 0.0F), 65504.0F, std::nextafter(65504.0F, INFINITY),
        std::nextafter(65520.0F, 0.0F), 65520.0F, std::nextafter(65520.0F, INFINITY),
        -std::nextafter(65504.0F, 0.0F), -65504.0F, -std::nextafter(65504.0F, INFINITY),
        -std::nextafter(65520.0F, 0.0F), -65520.0F, -std::nextafter(65520.0F, INFINITY)};
    constexpr std::array<std::uint32_t, 8> nan{0x7fc00001U, 0xffc12345U, 0x7fffffffU, 0xffffffffU,
        0x7f800001U, 0xff812345U, 0x7fbfffffU, 0xffbfffffU};
    for (unsigned r = 0; r < rows; ++r) for (unsigned j = 0; j < 640U; ++j) {
        float gate = 32.0F, up = 0.0F;
        const auto index = static_cast<std::size_t>(r) * 1280U + j;
        const unsigned tag = r * 13U + j;
        if (kind <= 1U) {
            gate = static_cast<float>(static_cast<int>(rng() % 2049U) - 1024) / 128.0F;
            up = static_cast<float>(static_cast<int>(rng() % 2049U) - 1024) / 64.0F;
        } else if (kind == 2U) { gate = std::bit_cast<float>(zeros[tag % zeros.size()]); up = (tag & 1U) ? -1.0F : 1.0F; }
        else if (kind == 3U) up = std::bit_cast<float>(tag % 3U ? subnormal[tag % subnormal.size()] : zero_tie[(tag / 3U) % zero_tie.size()]);
        else if (kind == 4U) up = boundary[tag % boundary.size()] / 32.0F;
        else if (kind == 5U) {
            // Multiple bad elements per group, adjacent groups, and all-bad rows.
            up = (r % 3U == 0U || j % 16U < 10U) ? ((tag & 1U) ? -1.0e30F : 1.0e30F) : 1.0F;
        } else if (kind == 6U) {
            const float inf = std::bit_cast<float>((tag & 1U) ? 0xff800000U : 0x7f800000U);
            if (tag & 2U) gate = inf; else up = inf;
        } else if (kind == 7U) {
            const float payload = std::bit_cast<float>(nan[(tag / 2U) % nan.size()]);
            if (tag & 1U) gate = payload; else up = payload;
        } else {
            // Distinct NaNs in gate and up of the same feature: which payload survives depends on operand order.
            gate = std::bit_cast<float>(nan[tag % nan.size()]); up = std::bit_cast<float>(nan[(tag + 3U) % nan.size()]);
        }
        values[index] = gate; values[index + 640U] = up;
    }
    return values;
}
void shared_pre_bench(Fixture& f) {
    auto current = f.scope();
    const HipLaunchScope active(current);
    for (unsigned warm = 0; warm < 2U; ++warm) for (bool pre : {false, true})
        require(shared_expert_shared_pre_fixture_launch(f.contexts, f.config, pre) == pre, "bench route not admitted");
    f.owner.finish();
    hipEvent_t start = nullptr, stop = nullptr;
    check(hipEventCreate(&start), "create shared pre start");
    try {
        check(hipEventCreate(&stop), "create shared pre stop");
        // Median of 15 transactions after 3 warm-up launches of the same route (single samples follow clock steps).
        const auto sample = [&](bool pre) {
            std::vector<float> times;
            for (unsigned it = 0; it < 18U; ++it) {
                check(hipEventRecord(start, f.owner.stream), "record shared pre start");
                require(shared_expert_shared_pre_fixture_launch(f.contexts, f.config, pre) == pre, "bench route changed");
                check(hipEventRecord(stop, f.owner.stream), "record shared pre stop");
                check(hipEventSynchronize(stop), "complete shared pre sample");
                float ms = 0; check(hipEventElapsedTime(&ms, start, stop), "time shared pre transaction");
                if (it >= 3U) times.push_back(ms);
            }
            std::sort(times.begin(), times.end());
            std::cout << (pre ? " B=" : " A=") << times[times.size() / 2U] << "ms";
        };
        std::cout << "shared pre complete gate|up + activation + down rows 8192 A/A";
        sample(false); sample(false); std::cout << '\n';
        for (unsigned repeat = 0; repeat < 3U; ++repeat) {
            std::cout << "shared pre ABBA " << repeat;
            for (bool pre : {false, true, true, false}) sample(pre);
            std::cout << '\n';
        }
    } catch (...) {
        if (stop) static_cast<void>(hipEventDestroy(stop));
        static_cast<void>(hipEventDestroy(start)); throw;
    }
    check(hipEventDestroy(stop), "destroy shared pre stop"); check(hipEventDestroy(start), "destroy shared pre start");
}
void shared_pre_fixtures(bool bench) {
    fixture_environment("STRIX_QPF2K_DENSE", "1");
    fixture_environment("STRIX_QPF2K_DENSE_SITES", "48");
    fixture_environment("STRIX_QPF2K_DENSE_MIN_ROWS", "32");
    {
        Fixture short_rows(16U, false); short_rows.config.dense_f16_prefill = true; short_rows.prepare(0, false);
        const auto a = short_rows.run_pre(false, nullptr, short_rows.owner.stream, false);
        const auto b = short_rows.run_pre(true, nullptr, short_rows.owner.stream, false, -1, 1, SharedPreFallback::short_rows);
        require(a.arena == b.arena && a.overflow == b.overflow, "shared pre changed a short path");
    }
    for (unsigned rows : {32U, 40U, 48U, 56U, 64U, 256U, 288U, 512U, 1000U, 1024U, 8192U}) {
        Fixture f(rows, false); f.config.dense_f16_prefill = true; f.prepare(0, false);
        const auto injection = f.owner.allocate(static_cast<std::uint64_t>(rows) * 1280U * sizeof(float));
        hipStream_t side = nullptr;
        check(hipStreamCreateWithFlags(&side, hipStreamNonBlocking), "create shared pre side stream");
        try {
            for (unsigned kind = 0; kind < 9U; ++kind) {
                const auto values = gate_up_values(rows, kind);
                write(injection, bytes_of(values));
                const auto* device = kind ? reinterpret_cast<const float*>(injection.address) : nullptr;
                for (const auto stream : {f.owner.stream, side}) {
                    const unsigned pre_rows = rows / 32U * 32U;
                    const auto dead = [&](const std::vector<std::byte>& arena) { return f.mask_dead(arena, pre_rows); };
                    const auto a = f.run_pre(false, device, stream), b = f.run_pre(true, device, stream);
                    require(dead(a.arena) == dead(b.arena), "shared pre BF16 / complete FP32 temporary / guard bytes differ");
                    // Producers of the preimage: paired form 4, the activation pre-kernel, paired form 0. Same
                    // FP16 bytes, gate, outputs and counter.
                    for (const int pair : {0, 2}) {
                        const auto c = f.run_pre(true, device, stream, true, -1, pair);
                        require(dead(c.arena) == dead(b.arena), "paired gate|up changed output / gate / tail bytes");
                        if (c.a16 != b.a16) {
                            std::size_t n = 0, shown = 0;
                            for (std::size_t k = 0; k < b.a16.size() && k < c.a16.size(); ++k) {
                                if (b.a16[k] == c.a16[k]) continue;
                                if (shown++ < 6) {
                                    float act = 0.0F;
                                    std::memcpy(&act, a.arena.data() + static_cast<std::size_t>(f.temporary[k / 640U].address - f.scratch.address) + (k % 640U) * sizeof(float), sizeof(act));
                                    std::cerr << "a16 diff rows " << rows << " kind " << kind << " pair " << pair << " r " << k / 640U << " i " << k % 640U
                                              << std::hex << " prod 0x" << b.a16[k] << " route 0x" << c.a16[k] << " oracle 0x"
                                              << __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(act)) << std::dec << "\n";
                                }
                                ++n;
                            }
                            std::cerr << "a16 diffs " << n << " of " << b.a16.size() << " (sizes " << b.a16.size() << "/" << c.a16.size() << ")\n";
                        }
                        require(c.a16 == b.a16, "paired gate|up FP16 preimage bytes differ");
                        require(c.overflow == b.overflow, "paired gate|up overflow counter delta differs");
                    }
                    // Preimage oracle: the RNE FP16 of the default route's FP32 activation (NaN: any NaN).
                    require(b.a16.size() == static_cast<std::size_t>(pre_rows) * 640U, "shared pre preimage size");
                    for (unsigned r = 0; r < pre_rows; ++r) for (unsigned i = 0; i < 640U; ++i) {
                        float act = 0.0F;
                        std::memcpy(&act, a.arena.data() + static_cast<std::size_t>(f.temporary[r].address - f.scratch.address) + i * sizeof(float), sizeof(act));
                        const auto want = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(act));
                        const auto got = b.a16[static_cast<std::size_t>(r) * 640U + i];
                        const bool nan = (got & 0x7c00U) == 0x7c00U && (got & 0x3ffU);
                        require(std::isnan(act) ? nan : got == want, "shared pre preimage differs from the FP16 of the default activation");
                    }
                    if (stream == f.owner.stream) {
                        // Two poisons per route: written bytes (equal under both poisons) and their values must match, and
                        // every other output/temporary byte must keep its poison in both routes.
                        const auto a1 = dead(f.run_pre(false, device, stream, true, 0xa5).arena), a2 = dead(f.run_pre(false, device, stream, true, 0x5a).arena);
                        const auto b1 = dead(f.run_pre(true, device, stream, true, 0x5a).arena), b2 = dead(f.run_pre(true, device, stream, true, 0xa5).arena);
                        std::size_t written = 0;
                        for (std::size_t i = 0; i < a1.size(); ++i) {
                            const bool wa = a1[i] == a2[i], wb = b1[i] == b2[i];
                            require(wa == wb, "shared pre wrote a different byte set than the default");
                            require(!wa || a1[i] == b1[i], "shared pre written byte differs (distinct poison)");
                            written += wa;
                        }
                        require(written > 0, "shared pre poison comparison saw no written bytes");
                    }
                    require(a.overflow == b.overflow, "shared pre complete overflow counter delta differs");
                    unsigned expected = 0U;
                    for (unsigned r = 0; r < rows / 32U * 32U; ++r) for (unsigned g = 0; g < 640U; g += 8U) {
                        bool bad = false;
                        for (unsigned j = 0; j < 8U; ++j) {
                            float act = 0.0F;
                            const auto at = static_cast<std::size_t>(f.temporary[r].address - f.scratch.address) + (g + j) * sizeof(float);
                            std::memcpy(&act, a.arena.data() + at, sizeof(act));
                            bad |= std::isfinite(act) && !(std::fabs(act) <= 65504.0F);
                        }
                        expected += bad ? 1U : 0U;
                    }
                    require(a.overflow == expected, "shared pre counter differs from aligned-eight workspace oracle");
                    if (kind == 5U) require(a.overflow != 0U, "finite overflow fixture did not exercise counter");
                    require(read(injection) == bytes_of(values), "shared pre modified injected gate/up payloads"); guards(injection);
                    // Queued reuse with no host observation between producers/down consumers.
                    const auto queued_before = shared_expert_shared_pre_fixture_overflow_count();
                    auto current = f.scope(); current.stream = stream;
                    { const HipLaunchScope active(current);
                      for (bool pre : {true, false, true})
                          require(shared_expert_shared_pre_fixture_launch(f.contexts, f.config, pre, device) == pre,
                              "queued shared pre route changed"); }
                    check(hipStreamSynchronize(stream), "complete queued shared pre reuse");
                    require(dead(read(f.scratch)) == dead(a.arena), "queued shared pre bytes differ");
                    require(shared_expert_shared_pre_fixture_overflow_count() - queued_before == 3U * a.overflow,
                        "queued shared pre counter deltas differ");
                }
            }
            // Owner suppression must defeat even the explicit opt-in seam, without changing fallback bytes.
            const bool prior_baseline = qpf2k::baseline; qpf2k::baseline = true;
            try {
                const auto a = f.run_pre(false, nullptr, f.owner.stream, false);
                const auto b = f.run_pre(true, nullptr, f.owner.stream, false, -1, 1, SharedPreFallback::suppressed);
                require(a.arena == b.arena && a.overflow == b.overflow, "suppressed shared pre changed fallback");
            } catch (...) { qpf2k::baseline = prior_baseline; throw; }
            qpf2k::baseline = prior_baseline;
            if (bench && rows == 8192U) shared_pre_bench(f);
        } catch (...) { static_cast<void>(hipStreamSynchronize(side)); static_cast<void>(hipStreamDestroy(side)); throw; }
        check(hipStreamDestroy(side), "destroy shared pre side stream");
        std::cout << "shared pre byte fixtures rows " << rows << " passed\n";
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool fixtures_only = fixture_flag("STRIX_SHARED_PRE_FIXTURES_ONLY");
        const bool bench = fixture_flag("STRIX_SHARED_PRE_BENCH");
        if (fixtures_only || bench) { shared_pre_fixtures(bench); return 0; }
        require(argc >= 2 && argc <= 3, "usage: shared-known-window-rows ROWS [PREFETCH0OR1]");
        const auto rows = static_cast<unsigned>(std::stoul(argv[1]));
        require(rows == 4 || rows == 8, "fixture requires R4 or R8");
        require(argc == 2 || std::string_view(argv[2]) == "0" || std::string_view(argv[2]) == "1",
            "fixture prefetch argument must be zero or one");
        const bool prefetch = argc == 3 && std::string_view(argv[2]) == "1";
        Fixture f(rows, prefetch);
        for (unsigned kind = 0; kind < 2; ++kind) for (const bool reverse : {false, true}) {
            f.prepare(kind, reverse); const auto reference = f.run(false);
            for (unsigned repeat = 0; repeat < 3; ++repeat)
                require(f.run(true) == reference, "wide shared result differs from complete incumbent arena bits");
            require(f.run_detached() == reference, "detached wide shared result differs from complete incumbent arena bits");
        }
        f.prepare(0, false);
        for (const bool capture : {false, true}) {
            const auto original = f.contexts;
            f.rejected([&] { ++f.contexts.back().position; }, [&] { f.contexts = original; }, capture, "R2 row identity/position differs");
            f.rejected([&] { f.contexts.back().execution_identity ^= 1; }, [&] { f.contexts = original; }, capture, "R2 row identity/position differs");
            f.rejected([&] { f.contexts.back().temporaries = {}; }, [&] { f.contexts = original; }, capture, "R2 requires matching C1 row IO");
            auto copied_memory = f.memory;
            f.rejected([&] { f.contexts.back().memory = &copied_memory; }, [&] { f.contexts = original; }, capture, "R2 scratch aliases another memory binding");
            const auto offset = f.window.row_view.back().allocations[2].offset;
            f.rejected([&] { f.window.row_view.back().allocations[2].offset = f.window.row_view.front().allocations[2].offset; },
                [&] { f.window.row_view.back().allocations[2].offset = offset; }, capture, "R2 row extent/order differs");
            auto copied_weights = f.weights;
            f.rejected([&] { f.resources.back().model_weights = &copied_weights; }, [&] { f.resources.back().model_weights = &f.weights; }, capture, "R2 weight/access owners differ");
            const auto type = f.manifest.tensors[0].ggml_type;
            f.rejected([&] { f.manifest.tensors[0].ggml_type = 20; }, [&] { f.manifest.tensors[0].ggml_type = type; }, capture, "R2 tensor type/layer/owner differs");
            f.rejected([&] { f.contexts[rows - 2U].position += 2; f.contexts.back().position += 2; },
                [&] { f.contexts = original; }, capture, "rows identity/position/IO differs");
            f.rejected([&] { f.contexts.front().position = UINT64_MAX; }, [&] { f.contexts = original; },
                capture, "rows positions overflow");
            f.rejected([&] { f.resources[rows - 2U].model_weights = &copied_weights; f.resources.back().model_weights = &copied_weights; },
                [&] { f.resources[rows - 2U].model_weights = &f.weights; f.resources.back().model_weights = &f.weights; },
                capture, "rows retained resource owners differ");
            const auto row_views = f.window.row_view;
            f.rejected([&] {
                const auto stride = f.input[1].address - f.input[0].address;
                for (unsigned r = 0; r < rows; ++r)
                    f.window.row_view[r].allocations[1].offset = f.window.row_view[r].allocations[0].offset + 2U * stride;
            }, [&] { f.window.row_view = row_views;
                for (unsigned r = 0; r < rows; ++r) f.contexts[r].arena = &f.window.row_view[r]; },
                capture, "rows cross-row writable/input alias");
            f.rejected([&] {
                for (unsigned r = 0; r < rows; ++r) f.window.row_view[r].allocations[0].offset += f.scratch.bytes + 256U;
            }, [&] { f.window.row_view = row_views;
                for (unsigned r = 0; r < rows; ++r) f.contexts[r].arena = &f.window.row_view[r]; },
                capture, "execution buffer exceeds bound arena allocation");
            const auto arena_bytes = f.memory.arena_bytes;
            f.rejected([&] { f.memory.arena_bytes = UINT64_MAX; }, [&] { f.memory.arena_bytes = arena_bytes; },
                capture, "execution arena address range overflows uintptr_t", true);
            f.rejected([&] { f.contexts.resize(3); }, [&] { f.contexts = original; }, capture, "rows count must be R4 or R8");
        }
        f.rejected([] {}, [] {}, true, "rows cannot run during capture"); // Valid metadata still cannot capture.
        const auto reference = f.run(false);
        require(f.run(true) == reference, "wide shared retry after rejected metadata is not reusable");
        std::cout << "shared known-window rows source fixture R" << rows << " passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
