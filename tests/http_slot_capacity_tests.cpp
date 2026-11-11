#include "../apps/http_slot_capacity.hpp"
#include "../apps/hip_decode_options.hpp"
#include <iostream>
using namespace strix::decode_cli;
int main() {
    int failures = 0;
    const auto expect = [&](bool ok) { if (!ok) ++failures; };
    for (const auto n : {1U, 2U, 8U, 9U, 16U}) {
        const auto bytes = 100U + (n - 1U) * 10U + n * 2U;
        expect(resolve_http_slot_capacity(10, 2, bytes, bytes, 100).slots == n);
        expect(resolve_http_slot_capacity(10, 2, bytes - 1U, bytes, 100).slots == (n - 1U));
    }
    expect(resolve_http_slot_capacity(10, 2, 1000, 200, 100).slots == 9U);
    expect(resolve_http_slot_capacity(10, 2, 1000, 1000, 100, 4).slots == 4U);
    expect(resolve_http_slot_capacity(10, 2, {}, 1000, 100).slots == 0U);
    expect(resolve_http_slot_capacity(10, 2, 1000, {}, 100).slots == 0U);
    expect(resolve_http_slot_capacity(UINT64_MAX, 1, UINT64_MAX, UINT64_MAX, 0).slots == 0U);
    expect(resolve_http_slot_capacity(1, 0, UINT64_MAX, UINT64_MAX, 0).slots == UINT32_MAX);
    expect(resolve_http_slot_capacity(10, 2, 1000, 1000, 100, 1000).slots == 75U);
    expect(resolve_http_slot_capacity(10, 2, 1000, 1000, 100, 75).slots == 75U);
    expect(resolve_http_slot_capacity(10, 2, 1, 1, 100).slots == 0U);
    expect(resolve_http_slot_capacity(10,2,101,1000,100).slots==0U); // host only
    expect(resolve_http_slot_capacity(10,2,1000,101,100,1).slots==0U); // first slot cap cannot force fit
    const HttpServingMemory bridge{30,10,5,2};
    expect(resolve_http_bridge_capacity(10,bridge,134,1000,100,100).slots==0U);
    expect(resolve_http_bridge_capacity(10,bridge,1000,111,100,100).slots==0U);
    expect(resolve_http_bridge_capacity(10,bridge,135,112,100,100).slots==1U);
    // Same boundary transition as the captured runtime callback: pre-bridge
    // sizing, shared residency, four eager slots, then successive lazy slots.
    for (const bool copied : {false,true}) for (const bool cache : {false,true}) {
        const HttpServingMemory cost{copied?3U:2U,2,1,1};
        const std::uint64_t fixed=4, cache_cap=cache?3:0, staging_cap=cache?2:0;
        const auto initial_headroom=http_shared_growth_headroom(fixed,cache_cap,0,staging_cap,0);
        const auto initial_host=initial_headroom+cost.shared_host+6+5;
        const auto initial_device=initial_headroom+cost.shared_device+6+5;
        expect(resolve_http_bridge_capacity(1,cost,initial_host,initial_device,
            initial_headroom,initial_headroom).slots==6);
        HttpBridgeGrowthObservation observed{initial_host,initial_device,0,0};
        unsigned observations=0;
        const auto growth=make_http_bridge_growth_check(1,cost,fixed,cache_cap,staging_cap,[&] {
            ++observations;
            return observed;
        });
        // Shared assets now resident; each eager slot consumes P, extra stacks R.
        auto host=initial_host-cost.shared_host-4-3;
        auto device=initial_device-cost.shared_device-4-3;
        // Try empty, partially filled and fully resident cache/capture payloads.
        for (const auto fraction : {0U,1U,2U}) {
            const auto cache_resident=cache_cap*fraction/2;
            const auto staging_resident=staging_cap*fraction/2;
            observed={host-cache_resident-staging_resident,device-cache_resident-staging_resident,
                cache_resident,staging_resident};
            const auto remaining=http_shared_growth_headroom(fixed,cache_cap,cache_resident,
                staging_cap,staging_resident);
            for (unsigned built=4;built<6;++built) {
                expect(growth());
                *observed.host-=2; *observed.device-=2;
            }
            expect(observed.host==remaining && observed.device==remaining);
            expect(!growth());
            expect(!http_bridge_growth_fits(1,cost,remaining+1,remaining+2,remaining));
            expect(!http_bridge_growth_fits(1,cost,remaining+2,remaining+1,remaining));
        }
        expect(observations==9);
    }
    expect(http_shared_growth_headroom(4,3,99,2,99)==4);
    expect(http_eager_slots(25) == 4U);
    expect(http_eager_slots(2) == 2U);
    expect(http_eager_slots(25, 0) == 25U);
    expect(http_eager_slots(25, 1) == 1U);
    for (const bool queued : {false, true})
        for (const bool memory : {false, true})
            for (const auto active : {3U, 4U})
                for (const auto capacity : {4U, 25U})
                    expect(http_slot_should_grow(queued, active, 4, capacity, memory) ==
                        (queued && memory && active == 4U && capacity == 25U));
    for (const bool queued : {false, true})
        for (const auto active : {3U, 4U})
            expect(http_slot_should_grow(queued, active, 4, 25, true, true) == (active == 4U));
    expect(!http_slot_should_grow(false, 4, 4, 4, true, true));
    expect(!http_slot_should_grow(false, 4, 4, 25, false, true));
    expect(http_slot_memory_fits(10, 2, 112, 112, 100));
    expect(!http_slot_memory_fits(10, 2, 111, 112, 100));
    expect(!http_slot_memory_fits(10, 2, 112, {}, 100));
    expect(!http_slot_memory_fits(10, 2, {}, 112, 100));
    expect(!http_slot_memory_fits(10, 2, 112, 111, 100));
    expect(http_slot_memory_fits(10, 0, 110, 110, 100));
    expect(!http_slot_memory_fits(10, 0, 109, 110, 100));
    expect(!http_slot_memory_fits(UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX, 1));
    const auto parse = [](std::vector<std::string> args) {
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        return parse_options(static_cast<int>(argv.size()), argv.data());
    };
    expect(http_slot_retained_bytes(10,20,30,40,50)==150);
    bool retained_overflow=false;
    try { (void)http_slot_retained_bytes(UINT64_MAX,1,0,0,0); } catch(const std::overflow_error&) { retained_overflow=true; }
    expect(retained_overflow);
    auto deploy_auto=parse({"decoder","model.gguf","1","--deploy","auto"});
    expect(!deploy_auto.http_batch_explicit && !deploy_auto.deployment_spec->slots);
    auto deploy_single=parse({"decoder","model.gguf","1","--http-batch","0","--deploy","auto"});
    expect(deploy_single.http_batch_explicit && !deploy_single.http_batch_cap && !deploy_single.deployment_spec->slots);
    auto parsed = parse({"decoder", "model.gguf", "1"});
    expect(!parsed.http_batch_explicit);
    parsed = parse({"decoder", "model.gguf", "1", "--http-batch", "0"});
    expect(parsed.http_batch_explicit && !parsed.http_batch_cap);
    parsed = parse({"decoder", "model.gguf", "1", "--http-batch", "16"});
    expect(parsed.http_batch_explicit && parsed.http_batch_cap == 16U);
    Options o;
    o.http_serve = true;
    o.mtp.window_prefix = o.mtp.draft_chain = true;
    o.mtp.window_rows_set = {2U, 4U, 8U};
    resolve_http_route(o); expect(o.http_batch == 1U && !o.http_batch_explicit);
    o.http_batch_explicit = true; o.http_batch_cap = 9;
    resolve_http_route(o); expect(o.http_batch == 9U);
    o.http_batch_cap = 0; resolve_http_route(o); expect(o.http_batch == 0U);
    o.http_batch_explicit = false; o.mtp.copy_wide_rows = {16};
    resolve_http_route(o); expect(o.http_batch == 0U);
    o.http_batch_explicit = true; o.http_batch_cap = 2;
    bool rejected = false;
    try { resolve_http_route(o); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected);
    o.http_batch_explicit = false; o.mtp.copy_wide_rows.clear(); o.mtp.draft_chain = false;
    resolve_http_route(o); expect(o.http_batch == 0U);
    o.mtp.draft_chain = true; o.rtx_state_import = true;
    resolve_http_route(o); expect(o.http_batch == 0U && o.http_slot_reason == "single-only state import");
    o.rtx_state_import = false; o.http_sampling = "sample";
    o.http_batch_explicit = true; o.http_batch_cap = 1U;
    resolve_http_route(o); expect(o.http_batch == 1U);
    std::cout << "HTTP slot tests: failures=" << failures << '\n';
    return failures ? 1 : 0;
}
