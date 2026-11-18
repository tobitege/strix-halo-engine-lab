#pragma once
// Automatic exchange-row admission (max-rows auto: the 8192-row candidate, 4096 fallback) shared by strix-glm5-rank and
// strix-glm5-serve, so both entry points size the exchange and the prefill chunk the same way for the same memory:
// a dry run of the load and prefill admissions at 8192 rows over every capacity-layout candidate, one vote per probe
// with the peer, then the entry point's row agreement (the CLI's decode handshake, the server's row merge).
#include "glm_rank_defaults.hpp"
#include "glm/engine/glm_mem.hpp"
#include "glm/prefill/glm_pf_capacity_policy.hpp"
#include "glm/prefill/glm_pf_load_admission.hpp"
#include "glm/prefill/glm_pf_path.hpp"
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix::glm::rank_defaults {

struct AutoRowsPlan {
    unsigned rank = 0, world = 1;
    std::uint64_t capacity = 0;
    bool mtp = false;
    bool reference = false;  // prefill_mode == "ref"
    strix::glm::pf::TierB tier = strix::glm::pf::TierB::gemv;
};

// Row probing deliberately restores the layout. Always give Engine::load both
// callbacks so it can vote a common candidate and coordinate install retries.
struct PrefillLoad {
    AutoRowsPlan plan;
    std::uint64_t margin = 0, admitted_path = 0;
    bool before_load = false, plan_only = false;

    template<class Engine, class Options, class SelectRows>
    auto load(Engine& eng, Options& opt, SelectRows&& select_rows) {
        namespace pf = strix::glm::pf;
        namespace mem = strix::glm::engine::mem;
        std::uint64_t first_forward = 0;
        const auto make = [&](bool sharded) {
            return pf::make_prefill_path(64 / plan.world, 64 / plan.world, plan.capacity, opt.max_rows,
                                         plan.tier, margin, sharded, first_forward);
        };
        return eng.load(std::forward<SelectRows>(select_rows), [&](const mem::LoadAdmission& admission) {
            first_forward = admission.first_forward_reserve;
            // Provisional auto admission still has the 8192-row plan. Rechecks
            // see the agreed rows and actual communication pools.
            if (!plan.reference && (!opt.auto_rows || admission.recheck)) {
                const auto b = pf::prefill_budget(64 / plan.world, 64 / plan.world, plan.capacity,
                                                  opt.max_rows, plan.mtp, plan.tier, admission.kv_shard);
                admitted_path = b.path;
                const auto comm_bytes = plan.world > 1 && !admission.recheck
                    ? std::uint64_t{opt.max_rows} * (3 * 4096 * 4 + 16) + 16384 : 0;
                pf::admit_prefill_load(admission, b.path, b.workspace, margin, 3 * comm_bytes);
                std::fprintf(stderr, "rank %u prefill admission: path %.3f GiB, workspace %.3f GiB, install/staging %.3f GiB, late safety %.3f GiB, first-forward reserve %.3f GiB, layout %s%s, grouped-moe legacy-wmma=%u exact=%u, attention-scratch-rows=%u\n",
                             plan.rank, mem::gib(b.path), mem::gib(b.workspace), mem::gib(admission.prefill_install),
                             mem::gib(margin), mem::gib(first_forward),
                             admission.kv_shard ? "sharded" : "replicated", admission.candidate ? " (candidate)" : "",
                             unsigned(b.moe_route.legacy_wmma), unsigned(b.moe_route.exact), b.attention_rows);
            }
            if (!admission.candidate && before_load && !plan_only && (!opt.auto_rows || admission.recheck)) {
                eng.set_prefill(make(admission.kv_shard));
                std::printf("rank %u prefill %s set before load\n", plan.rank,
                            plan.tier == pf::TierB::gemv ? "gemv" : "wmma");
            }
        }, [&](bool sharded) -> std::shared_ptr<strix::glm::engine::PrefillPath> {
            if (plan.reference || before_load || plan_only) return {};
            return make(sharded);
        });
    }
};

// Called from the engine's load-admission callback when max_rows_auto(e). vote(local) exchanges one fit vote with the
// peer (0 refused, 1 admitted, 2 install error) and returns the peer's vote (== local on one rank); agree(rows) runs the
// entry point's row agreement after the probe and returns the agreed rows. Updates e.max_rows and returns the rows.
template<class Vote, class Agree>
unsigned admit_auto_rows(Effective& e, const strix::glm::engine::mem::LoadAdmission& admission, const AutoRowsPlan& p,
                         std::uint64_t prefill_margin, Vote&& vote, Agree&& agree) {
    namespace pf = strix::glm::pf;
    namespace mem = strix::glm::engine::mem;
    const unsigned heads = 64 / p.world;
    auto b = pf::prefill_budget(heads, heads, p.capacity, kAutoMaxRows, p.mtp, p.tier, admission.kv_shard);
    const auto comm_bytes = p.world > 1 && !admission.recheck ? std::uint64_t{kAutoMaxRows} * (3 * 4096 * 4 + 16) + 16384 : 0;
    select_max_rows(e, [&] {
        if (p.reference) {
            auto terms = admission.terms;
            terms.push_back({"communication growth and chunk staging", 3 * comm_bytes, 0});
            return mem::fits(terms, admission.snapshot, admission.margins, admission.device_free);
        }
        // Preserve the preferred chunk size if ANY scratch candidate admits locally. The engine later votes the SAME
        // layout on both ranks before installation, including at fallback rows.
        const auto saved = pf::capacity_layout();
        std::exception_ptr install_error;
        std::uint32_t fits = 0;
        try {
            for (const auto layout : pf::capacity_candidates(pf::capacity_policy(), admission.kv_shard)) {
                try {
                    pf::capacity_layout() = layout;
                    b = pf::prefill_budget(heads, heads, p.capacity, kAutoMaxRows, p.mtp, p.tier, admission.kv_shard);
                    pf::capacity_layout() = saved;
                    pf::admit_prefill_load(admission, b.path, b.workspace, prefill_margin, 3 * comm_bytes);
                    fits = 1;
                    break;
                } catch (const std::runtime_error&) {
                    // A local refusal is only a row-size probe here.
                }
            }
        } catch (...) {
            install_error = std::current_exception();
            fits = 2;
        }
        pf::capacity_layout() = saved;
        // One vote per probe, even if ranks admit different candidates.
        const std::uint32_t peer_fits = vote(fits);
        if (peer_fits > 2U) throw std::runtime_error("glm capacity: invalid row probe vote");
        if (install_error) std::rethrow_exception(install_error);
        if (peer_fits == 2U) throw std::runtime_error("peer install failed");
        return fits == 1U;
    });
    std::fprintf(stderr, "rank %u max-rows admission: %s; candidate %s, path %.2f GiB, workspace %.2f GiB, install %.2f GiB\n",
                 p.rank, mem::describe(admission.snapshot).c_str(), e.max_rows.value.c_str(),
                 mem::gib(b.path), mem::gib(b.workspace), mem::gib(admission.prefill_install));
    agree_max_rows(e, agree(static_cast<unsigned>(std::stoul(e.max_rows.value))));
    return static_cast<unsigned>(std::stoul(e.max_rows.value));
}

}  // namespace strix::glm::rank_defaults
