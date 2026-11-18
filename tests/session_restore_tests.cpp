#include "strix/session_restore.hpp"

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::PersistentState all_state() {
    using strix::PersistentState;
    return PersistentState::attention_kv |
           PersistentState::recurrent_matrix |
           PersistentState::convolution_history |
           PersistentState::sparse_index |
           PersistentState::ngram_history |
           PersistentState::position_history;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::SerialSessionStateLayoutRequest layout_request;
    layout_request.context_tokens = 4096;
    const auto layout =
        strix::build_serial_session_state_layout(
            model,
            layout_request);

    strix::SerialSessionSnapshotRequest request;
    request.compiled_execution_identity =
        0x1234ULL;
    request.runtime_compatibility_identity =
        0x11112222ULL;
    request.next_position = 64;
    request.committed_tokens = 64;
    request.state.valid = all_state();
    for (auto& generation :
         request.state.generations) {
        generation = 3;
    }

    const auto snapshot =
        strix::build_serial_session_snapshot_plan(
            model,
            layout,
            request);
    const auto restore =
        strix::build_serial_session_restore_plan(
            model,
            layout,
            request.runtime_compatibility_identity,
            snapshot);

    expect(restore.plan_identity != 0 &&
               restore.ranges.size() ==
                   snapshot.ranges.size(),
           "restore plan reverses every compact snapshot range");
    expect(restore.next_position ==
               snapshot.next_position &&
               restore.state.generations ==
                   snapshot.state.generations,
           "restore plan preserves committed session metadata");

    auto vk_request = layout_request; vk_request.recurrent_storage = strix::GdnRecurrentStorage::value_key;
    const auto vk_layout = strix::build_serial_session_state_layout(model, vk_request);
    const auto vk_snapshot = strix::build_serial_session_snapshot_plan(model, vk_layout, request);
    const auto vk_restore = strix::build_serial_session_restore_plan(model, vk_layout,
        request.runtime_compatibility_identity, vk_snapshot);
    expect(vk_restore.recurrent_storage == strix::GdnRecurrentStorage::value_key &&
        vk_restore.plan_identity != restore.plan_identity,
        "same-layout candidate restore retains physical storage identity");
    bool storage_rejected = false;
    try { static_cast<void>(strix::build_serial_session_restore_plan(model, layout,
        request.runtime_compatibility_identity, vk_snapshot)); }
    catch (const std::invalid_argument&) { storage_rejected = true; }
    expect(storage_rejected, "restore cannot silently reinterpret a cross-layout payload");
    auto forged_snapshot = snapshot; forged_snapshot.recurrent_storage = strix::GdnRecurrentStorage::value_key;
    storage_rejected = false;
    try { static_cast<void>(strix::build_serial_session_restore_plan(model, layout,
        request.runtime_compatibility_identity, forged_snapshot)); }
    catch (const std::invalid_argument&) { storage_rejected = true; }
    expect(storage_rejected, "restore rejects same-ID forged physical storage metadata");

    // recurrent state contract: a record written under one type is never restored under another
    for (const auto type : {strix::RecurrentStateType::f16, strix::RecurrentStateType::bf16}) {
        auto typed = model; typed.recurrent_state = type;
        const auto typed_layout = strix::build_serial_session_state_layout(typed, layout_request);
        const auto typed_snapshot = strix::build_serial_session_snapshot_plan(typed, typed_layout, request);
        static_cast<void>(strix::build_serial_session_restore_plan(typed, typed_layout,
            request.runtime_compatibility_identity, typed_snapshot));  // same type restores
        for (const auto* other : {&snapshot, &typed_snapshot}) {
            bool refused = false;
            try { static_cast<void>(strix::build_serial_session_restore_plan(
                other == &snapshot ? typed : model, other == &snapshot ? typed_layout : layout,
                request.runtime_compatibility_identity, other == &snapshot ? snapshot : typed_snapshot)); }
            catch (const std::invalid_argument&) { refused = true; }
            expect(refused, "restore refuses a snapshot written under a different recurrent state type");
        }
        auto other_type = model; other_type.recurrent_state = type == strix::RecurrentStateType::f16 ?
            strix::RecurrentStateType::bf16 : strix::RecurrentStateType::f16;
        const auto other_layout = strix::build_serial_session_state_layout(other_type, layout_request);
        bool refused = false;
        try { static_cast<void>(strix::build_serial_session_restore_plan(other_type, other_layout,
            request.runtime_compatibility_identity, typed_snapshot)); }
        catch (const std::invalid_argument&) { refused = true; }
        expect(refused, "restore refuses f16 <-> bf16 records of identical size");
    }

    for (std::size_t index = 0;
         index < restore.ranges.size();
         ++index) {
        const auto& source =
            snapshot.ranges[index];
        const auto& destination =
            restore.ranges[index];
        expect(destination.component ==
                   source.component &&
                   destination.storage ==
                       source.storage &&
                   destination.snapshot_offset ==
                       source.destination_offset &&
                   destination.bytes ==
                       source.bytes,
               "restore range reads from compact snapshot payload");
        const auto* allocation =
            strix::find_session_state_allocation(
                layout,
                destination.component);
        expect(allocation != nullptr &&
                   destination.destination_offset ==
                       allocation->offset,
               "restore range targets current session-state layout");
    }


    auto aligned_request = request;
    aligned_request.alignment = 256;
    const auto aligned_snapshot =
        strix::build_serial_session_snapshot_plan(
            model,
            layout,
            aligned_request);
    const auto aligned_restore =
        strix::build_serial_session_restore_plan(
            model,
            layout,
            request.runtime_compatibility_identity,
            aligned_snapshot);
    expect(aligned_snapshot.alignment == 256 &&
               aligned_restore.ranges.size() ==
                   aligned_snapshot.ranges.size(),
           "non-default snapshot alignment round-trips through restore");

    bool compatibility_rejected = false;
    try {
        static_cast<void>(
            strix::build_serial_session_restore_plan(
                model,
                layout,
                request.runtime_compatibility_identity + 1,
                snapshot));
    } catch (const std::invalid_argument&) {
        compatibility_rejected = true;
    }
    expect(compatibility_rejected,
           "restore rejects different runtime compatibility identity");

    auto corrupted = snapshot;
    corrupted.ranges[0].bytes -= 1;
    bool corruption_rejected = false;
    try {
        static_cast<void>(
            strix::build_serial_session_restore_plan(
                model,
                layout,
                request.runtime_compatibility_identity,
                corrupted));
    } catch (const std::invalid_argument&) {
        corruption_rejected = true;
    }
    expect(corruption_rejected,
           "restore rejects mutated snapshot range even if header identity is unchanged");

    auto compact_request = aligned_request;
    compact_request.policy = strix::SessionSnapshotPolicy::occupied_kv_v1;
    const auto compact = strix::build_serial_session_snapshot_plan(model, layout, compact_request);
    const auto compact_restore = strix::build_serial_session_restore_plan(model, layout,
        request.runtime_compatibility_identity, compact);
    expect(compact_restore.policy == compact.policy && compact_restore.snapshot_alignment == compact.alignment &&
        compact_restore.ranges.size() == compact.ranges.size(), "compact restore retains version, alignment and full layer census");
    for (std::size_t i = 0; i < compact.ranges.size(); ++i)
        expect(compact_restore.ranges[i].destination_offset == compact.ranges[i].source_offset &&
            compact_restore.ranges[i].snapshot_offset == compact.ranges[i].destination_offset,
            "compact restore writes each saved layer at its original physical stride");

    if (failures == 0) {
        std::cout << "session restore tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
