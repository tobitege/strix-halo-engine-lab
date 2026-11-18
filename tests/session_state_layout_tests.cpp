#include "rotary_mock.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/rotary_descriptor.hpp"

#include "strix/memory_plan.hpp"

#include <iostream>
#include <array>
#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

void attention_kv_storage_binds_layout() {
    auto model = strix::ModelContract::qwen38_flash_next();
    strix::SerialSessionStateLayoutRequest request;
    request.context_tokens = 4096;
    const auto bf16 = strix::build_serial_session_state_layout(model, request);
    model.attention_kv_storage = strix::AttentionKvStorage::int8_token;
    const auto int8 = strix::build_serial_session_state_layout(model, request);
    const auto* bf16_kv = strix::find_session_state_allocation(bf16, strix::SessionStateComponent::attention_kv);
    const auto* int8_kv = strix::find_session_state_allocation(int8, strix::SessionStateComponent::attention_kv);
    expect(bf16_kv && int8_kv && bf16_kv->bytes == 4096ULL * 24576ULL && int8_kv->bytes == 4096ULL * 12480ULL,
           "kv-int8-tok-v1 sizes the KV allocation");
    expect(bf16.layout_identity != int8.layout_identity, "KV storage enters the session layout identity");
    strix::validate_serial_session_state_layout_identity(model, int8);
}

void recurrent_state_type_binds_layout() {
    auto model = strix::ModelContract::qwen38_flash_next();
    model.recurrent_state = strix::RecurrentStateType::f32;
    strix::SerialSessionStateLayoutRequest request;
    request.context_tokens = 4096;
    const auto f32 = strix::build_serial_session_state_layout(model, request);
    const auto* f32_matrix = strix::find_session_state_allocation(f32, strix::SessionStateComponent::recurrent_matrix);
    expect(f32_matrix && f32_matrix->bytes == 108ULL * 1024ULL * 1024ULL, "f32 recurrent matrix layout unchanged");
    std::uint64_t previous_identity = f32.layout_identity;
    for (const auto type : {strix::RecurrentStateType::f16, strix::RecurrentStateType::bf16}) {
        model.recurrent_state = type;
        const auto typed = strix::build_serial_session_state_layout(model, request);
        const auto* matrix = strix::find_session_state_allocation(typed, strix::SessionStateComponent::recurrent_matrix);
        expect(matrix && f32_matrix && matrix->bytes * 2 == f32_matrix->bytes, "16-bit recurrent matrix is half of f32");
        expect(typed.device_total_bytes + f32_matrix->bytes / 2 == f32.device_total_bytes &&
                   typed.host_total_bytes == f32.host_total_bytes,
               "only the recurrent matrix changes size");
        for (const auto component : {strix::SessionStateComponent::attention_kv, strix::SessionStateComponent::gdn_convolution}) {
            const auto* a = strix::find_session_state_allocation(typed, component);
            const auto* b = strix::find_session_state_allocation(f32, component);
            expect(a && b && a->bytes == b->bytes, "other components keep their f32 size");
        }
        expect(typed.layout_identity != previous_identity, "recurrent state type enters the layout identity");
        previous_identity = typed.layout_identity;
        strix::validate_serial_session_state_layout_identity(model, typed);
        bool refused = false;
        try { strix::validate_serial_session_state_layout_identity(strix::ModelContract{}, typed); }
        catch (const std::exception&) { refused = true; }
        expect(refused, "layout identity of a 16-bit type is not valid for an f32 model");
    }
    expect(strix::gdn_matrix_state_bytes_per_layer(strix::ModelContract{}) == 3ULL * 1024ULL * 1024ULL,
           "default (f32) per-layer matrix bytes");
}

int main() {
    recurrent_state_type_binds_layout();
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::SerialSessionStateLayoutRequest request;
    request.context_tokens = 32'768;
    request.device_alignment = 256;
    request.host_alignment = 64;

    const auto layout =
        strix::build_serial_session_state_layout(
            model, request);

    auto rotary_model = model;
    auto previous = layout.layout_identity;
    for (auto factor : {2U, 4U}) {
        rotary_model.qwen_rotary_table = mock_rotary(factor);
        const auto other = strix::build_serial_session_state_layout(rotary_model, request);
        expect(other.layout_identity != previous && other.layout_identity != layout.layout_identity &&
            other.device_total_bytes == layout.device_total_bytes,
            "same capacity/bytes distinguish native/2/4 state envelopes");
        bool refused = false;
        try { strix::validate_serial_session_state_layout_identity(rotary_model, layout); }
        catch (const std::invalid_argument&) { refused = true; }
        expect(refused, "incompatible layout refuses without an allocator or importer");
        previous = other.layout_identity;
    }

    // Captured by running the unmodified identity function at 0d792e11.
    expect(layout.layout_identity == 16699253671626688724ULL &&
               strix::append_rotary_identity(layout.layout_identity, strix::RotaryDescriptor::native()) ==
                   16699253671626688724ULL,
           "native session identity retains its original eight bytes with no rotary payload");
    expect(layout.layout_identity != 0,
           "session-state layout has a stable non-zero identity");
    expect(layout.allocations.size() == 7,
           "serial state layout exposes seven explicit components");

    const auto* kv =
        strix::find_session_state_allocation(
            layout,
            strix::SessionStateComponent::attention_kv);
    expect(kv != nullptr &&
               kv->bytes ==
                   strix::base_kv_bytes_per_token(model) *
                       request.context_tokens &&
               kv->storage ==
                   strix::SessionStateStorage::device,
           "attention KV receives exact context-sized device storage");

    const auto* ple =
        strix::find_session_state_allocation(
            layout,
            strix::SessionStateComponent::ple_convolution);
    expect(ple != nullptr &&
               ple->bytes == 180ULL * 1024ULL &&
               !ple->provisional,
           "PLE convolution state is an exact device allocation");

    const auto* sparse =
        strix::find_session_state_allocation(
            layout,
            strix::SessionStateComponent::sparse_index);
    expect(sparse != nullptr &&
               !sparse->provisional &&
               sparse->bytes ==
                   strix::provisional_indexer_bytes_per_token(model) *
                       request.context_tokens,
           "QSA raw-key cache is an exact context-sized device allocation");

    const auto* ngram =
        strix::find_session_state_allocation(
            layout,
            strix::SessionStateComponent::ngram_history);
    const auto* position =
        strix::find_session_state_allocation(
            layout,
            strix::SessionStateComponent::position_history);
    expect(ngram != nullptr &&
               position != nullptr &&
               ngram->storage ==
                   strix::SessionStateStorage::host &&
               position->storage ==
                   strix::SessionStateStorage::host,
           "small resume histories remain explicit host state");

    strix::MemoryPlanRequest memory_request;
    memory_request.context_tokens =
        request.context_tokens;
    memory_request.sessions = 1;
    memory_request.include_mtp_kv = false;
    memory_request.include_indexer = true;
    const auto memory =
        strix::build_memory_plan(
            model, memory_request);

    const auto exact_unaligned =
        memory.base_kv +
        memory.indexer +
        memory.gdn_state +
        memory.ple_conv_state +
        memory.ngram_history +
        memory.position_history;

    expect(layout.device_total_bytes +
               layout.host_total_bytes >=
               exact_unaligned,
           "aligned session layout covers memory-plan state bytes");
    expect(layout.device_total_bytes %
                   request.device_alignment ==
               0 &&
               layout.host_total_bytes %
                   request.host_alignment ==
               0,
           "session state pools honor independent host/device alignment");

    auto no_sparse = request;
    no_sparse.include_sparse_index_candidate = false;
    const auto without_sparse =
        strix::build_serial_session_state_layout(
            model, no_sparse);
    expect(
        strix::find_session_state_allocation(
            without_sparse,
            strix::SessionStateComponent::sparse_index) ==
            nullptr,
        "session layout can omit QSA raw-key state");
    expect(without_sparse.layout_identity !=
               layout.layout_identity,
           "sparse-state policy changes layout identity");

    auto invalid = request;
    invalid.context_tokens =
        model.max_position_embeddings + 1ULL;
    bool context_rejected = false;
    try {
        static_cast<void>(
            strix::build_serial_session_state_layout(
                model, invalid));
    } catch (const std::invalid_argument&) {
        context_rejected = true;
    }
    expect(context_rejected,
           "session state layout rejects context beyond model maximum");

    auto vk_request = request;
    vk_request.recurrent_storage = strix::GdnRecurrentStorage::value_key;
    const auto vk = strix::build_serial_session_state_layout(model, vk_request);
    expect(vk.layout_identity != layout.layout_identity && vk.device_total_bytes == layout.device_total_bytes &&
        vk.host_total_bytes == layout.host_total_bytes && vk.allocations.size() == layout.allocations.size(),
        "physical recurrent layout changes identity without changing any pool capacity");
    auto explicit_baseline = request; explicit_baseline.recurrent_storage = strix::GdnRecurrentStorage::key_value;
    expect(strix::build_serial_session_state_layout(model, explicit_baseline).layout_identity == layout.layout_identity,
        "explicit incumbent storage retains the default layout identity");
    auto forged = layout; forged.recurrent_storage = strix::GdnRecurrentStorage::value_key;
    bool rejected = false;
    try { strix::validate_serial_session_state_layout_identity(model, forged); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "forged same-ID physical layout metadata is rejected");
    auto invalid_storage_request = request;
    invalid_storage_request.recurrent_storage = static_cast<strix::GdnRecurrentStorage>(99U);
    rejected = false;
    try { static_cast<void>(strix::build_serial_session_state_layout(model, invalid_storage_request)); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "layout builder rejects unsupported physical storage enum");

    // Non-square dimensions and two heads distinguish h/k/v addressing. All
    // copies are integer bits, including NaN payloads and signed zero.
    auto source = std::make_shared<std::array<std::uint32_t, 12>>(
        std::array<std::uint32_t, 12>{0x7fc12345U,0x80000000U,0U,0x7f800000U,0xff800000U,1U,
            0x3f800000U,0xbf800000U,0x00400000U,0xffc76543U,0x00800000U,0x7f7fffffU});
    auto destination = std::make_shared<std::array<std::uint32_t, 12>>();
    auto restored = std::make_shared<std::array<std::uint32_t, 12>>();
    const auto view = [](const auto& owner) {
        return strix::GdnRecurrentHostView{std::span<std::uint32_t>(*owner), owner,
            reinterpret_cast<std::uintptr_t>(owner->data()), sizeof(*owner), 7};
    };
    unsigned fences = 0;
    strix::convert_gdn_recurrent_host_storage(view(source), strix::GdnRecurrentStorage::key_value,
        view(destination), strix::GdnRecurrentStorage::value_key, 2, 2, 3, [&] { ++fences; });
    expect((*destination)[0] == (*source)[0] && (*destination)[1] == (*source)[3] &&
        (*destination)[6] == (*source)[6] && (*destination)[9] == (*source)[10],
        "once conversion maps non-square logical coordinates across separate heads");
    strix::convert_gdn_recurrent_host_storage(view(destination), strix::GdnRecurrentStorage::value_key,
        view(restored), strix::GdnRecurrentStorage::key_value, 2, 2, 3, [&] { ++fences; });
    expect(*restored == *source && fences == 2, "cross-layout roundtrip preserves every FP32 bit and fences each conversion");
    for (unsigned fault = 0; fault < 8; ++fault) {
        auto bad_source = view(source); auto bad_destination = view(destination);
        auto kind = strix::GdnRecurrentStorage::value_key;
        if (fault == 0) bad_source.owner.reset();
        if (fault == 1) bad_source.allocation_generation = 0;
        if (fault == 2) bad_source.owner_bytes -= 1;
        if (fault == 3) bad_destination.words = bad_destination.words.first(11);
        if (fault == 4) bad_destination = bad_source;
        if (fault == 5) kind = static_cast<strix::GdnRecurrentStorage>(99);
        if (fault == 6) bad_source.owner_base = std::numeric_limits<std::uintptr_t>::max() - 1;
        const auto before = *destination; const auto prior_fences = fences;
        rejected = false;
        try { strix::convert_gdn_recurrent_host_storage(bad_source, strix::GdnRecurrentStorage::key_value,
            bad_destination, kind, 2, 2, 3, [&] { ++fences; if (fault == 7) throw std::runtime_error("fence failure"); }); }
        catch (const std::exception&) { rejected = true; }
        expect(rejected && *destination == before && fences == prior_fences + (fault == 7 ? 1U : 0U),
            "conversion rejects malformed owner/extent/layout/alias or failed fence before modifying payload");
    }

    attention_kv_storage_binds_layout();
    if (failures == 0) {
        std::cout << "session state layout tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
