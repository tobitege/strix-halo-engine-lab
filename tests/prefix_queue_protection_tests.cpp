// Queue protection of prefix states (PrefixStateCache::protect_longest + serve::QueuedPrefixProtection), the
// governed automatic budget rule (serve::automatic_cache_budget / fit_cache_late) and the pending Qwen cache
// qualification rows (batch 8/16, context 32,768). Host-only.
#include "qwen_cache_admission.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/prefix_state_cache.hpp"
#include "strix/serve/cache_budget_rule.hpp"
#include "strix/serve/prefix_cache_queue_binding.hpp"
#include "strix/serve/queued_prefix_protection.hpp"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Tokens = std::vector<std::uint32_t>;
using Checkpoint = std::shared_ptr<const strix::PrefixCacheCheckpoint>;
constexpr std::uint32_t rows = 16;
constexpr std::uint64_t blocks = 17;          // a 17-block history per chat ("17K" scaled down)
constexpr std::uint64_t segment_bytes = 1024;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
strix::PrefixCacheBuffer heap(std::uint64_t bytes) {
    std::shared_ptr<std::byte> data(new std::byte[static_cast<std::size_t>(bytes ? bytes : 1)], std::default_delete<std::byte[]>());
    return {std::move(data), bytes};
}
strix::ModelContentIdentity identity_of(const char* text) {
    return strix::sha256_bytes(std::as_bytes(std::span<const char>(text, std::strlen(text))));
}
Tokens history(std::uint32_t chat) {
    Tokens t(blocks * rows);
    for (std::size_t i = 0; i < t.size(); ++i) t[i] = chat * 100000U + static_cast<std::uint32_t>(i) + 1U;
    return t;
}
// The full-history checkpoint of one chat (no shared segments between chats).
Checkpoint state(strix::PrefixStateCache& cache, const strix::ModelContentIdentity& id, const Tokens& tokens) {
    std::vector<std::shared_ptr<const strix::PrefixCacheSegment>> segments;
    auto previous = id;
    for (std::uint64_t k = 0; k < tokens.size() / rows; ++k) {
        const std::span<const std::uint32_t> block(tokens.data() + k * rows, rows);
        auto segment = std::make_shared<strix::PrefixCacheSegment>();
        segment->identity = id; segment->parent = previous;
        segment->digest = strix::prefix_cache_block_digest(previous, block);
        segment->block_index = k; segment->block_rows = rows;
        segment->tokens.assign(block.begin(), block.end());
        segment->payload = cache.allocate(segment_bytes);
        previous = segment->digest;
        segments.push_back(segment);
    }
    auto c = std::make_shared<strix::PrefixCacheCheckpoint>();
    c->identity = id; c->digest = previous; c->block_rows = rows;
    c->position = tokens.size(); c->segments = segments;
    c->fixed = cache.allocate(24); c->metadata.assign(3, std::byte{1});
    return c;
}
Tokens followup(const Tokens& h) {
    auto t = h;
    for (std::uint32_t i = 0; i < 20; ++i) t.push_back(900000000U + i);  // short new question
    return t;
}

// 16 chats x 17 blocks, 16 follow-ups queued at once, 8 slots, capture pressure from 16 other chats.
// Returns the number of waiting follow-ups whose state was lost before their restore.
std::size_t burst(bool protect, bool cancel_one) {
    const auto id = identity_of("qcprot-model");
    const std::uint64_t per_state = blocks * segment_bytes + 24;
    auto cache = std::make_shared<strix::PrefixStateCache>(strix::PrefixStateCachePolicy{per_state * 17 + 4096, {}, 0}, heap);
    std::vector<Tokens> chats;
    for (std::uint32_t c = 0; c < 16; ++c) {
        chats.push_back(history(c));
        require(cache->insert(state(*cache, id, chats.back())), "insert chat state");
    }
    strix::serve::GenerationQueue queue(64);
    std::vector<std::shared_ptr<strix::serve::GenerationTicket>> tickets;
    for (const auto& h : chats) {
        strix::serve::GenerationJob job;
        job.prompt = followup(h);
        tickets.push_back(queue.submit(std::move(job)));
        require(tickets.back() != nullptr, "submit");
    }
    strix::serve::QueuedPrefixProtectionPolicy policy;
    policy.enabled = protect;  // max_prepare 4: four scans to track 16 waiting prompts (bounded per scan)
    strix::serve::QueuedPrefixProtection protector(policy,
        strix::serve::prefix_cache_queue_binding(cache, [id] { return id; }, rows));
    for (int i = 0; i < 4; ++i) protector.scan(queue);
    if (protect) {
        require(protector.stats().held == 16 && protector.stats().prepared == 16, "16 waiting states protected after 4 scans");
        require(cache->stats().queue_protected_entries == 16, "cache gauge: 16 protected entries");
    }
    if (cancel_one) {
        tickets[15]->cancel();
        protector.scan(queue);
        require(protector.stats().held == 15 && protector.stats().released_left == 1, "cancellation releases its protection");
        require(cache->stats().queue_protected_entries == 15, "cache gauge after cancellation");
    }
    // 8 slots: admit 8, each restores (lookup after admission) then releases its reference.
    std::size_t lost = 0;
    std::vector<strix::serve::TakenJob> running;
    const auto admit = [&] {
        auto taken = queue.take(false);
        require(taken.has_value(), "take");
        auto handle = protector.take(taken->ticket);
        require(!protect || handle, "admitted job carries its protection");
        const auto& p = taken->job.prompt;
        const auto hit = cache->find_longest(id, p, rows, p.size() - 1);
        if (!hit || hit->position != blocks * rows) ++lost;
        handle.reset();  // restore done
        running.push_back(std::move(*taken));
    };
    for (int s = 0; s < 8; ++s) admit();
    // Capture pressure while 8 (or 7) follow-ups wait: running jobs and 16 other chats capture new states.
    for (std::uint32_t c = 100; c < 116; ++c) {
        protector.scan(queue);
        static_cast<void>(cache->insert(state(*cache, id, history(c))));
    }
    const auto waiting = cancel_one ? 7 : 8;
    for (int s = 0; s < waiting; ++s) { protector.scan(queue); admit(); }
    protector.scan(queue);
    if (protect) {
        require(cache->stats().queue_protected_entries == 0 && cache->stats().queue_protected_bytes == 0, "no protection leaks");
        require(protector.stats().tracked == 0 && protector.stats().held == 0, "protector tracks nothing at the end");
        require(cache->stats().queue_protected_erased == 0, "no protected entry erased");
    }
    return lost;
}

void queue_protection() {
    require(burst(true, false) == 0, "protection on: zero protected states lost");
    require(burst(true, true) == 0, "protection on with a cancellation: zero lost");
    const auto lost = burst(false, false);
    require(lost > 0, "control: without protection the pressure loses waiting states");
    std::cout << "queue protection: on 0 lost, off " << lost << " of 8 waiting lost\n";
}

void cache_edges() {
    const auto id = identity_of("qcprot-edges");
    const std::uint64_t per_state = blocks * segment_bytes + 24;
    auto cache = std::make_shared<strix::PrefixStateCache>(strix::PrefixStateCachePolicy{per_state * 2 + 1024, {}, 0}, heap);
    const auto a = history(1), b = history(2);
    require(cache->insert(state(*cache, id, a)), "insert a");
    const auto fa = followup(a);
    auto ticket = cache->prepare_lookup(id, fa, rows, fa.size() - 1);
    // Cap below one state: refused, never forced.
    require(!cache->protect_longest(ticket, per_state / 2), "cap refuses");
    require(cache->stats().queue_protection_cap_skips == 1, "cap skip counted");
    // A miss counts and holds nothing.
    const auto fb = followup(b);
    require(!cache->protect_longest(cache->prepare_lookup(id, fb, rows, fb.size() - 1)), "miss");
    require(cache->stats().queue_protection_misses == 1, "miss counted");
    {
        auto p1 = cache->protect_longest(ticket);
        auto p2 = cache->protect_longest(ticket);
        require(p1 && p2 && p1.position() == blocks * rows, "two references on one entry");
        require(cache->stats().queue_protected_entries == 1, "one entry, two references");
        // Everything else under pressure: the protected entry survives, the capture of b is still admitted.
        require(cache->insert(state(*cache, id, b)), "insert b");
        require(cache->insert(state(*cache, id, history(3))), "insert c evicts b, not a");
        require(cache->find_longest(id, fa, rows, fa.size() - 1) != nullptr, "protected a survived");
        p1.release();
        require(cache->stats().queue_protected_entries == 1, "still referenced");
        auto moved = std::move(p2);
        require(!p2 && moved, "move transfers the reference");
    }
    require(cache->stats().queue_protected_entries == 0 && cache->stats().queue_protected_bytes == 0, "released");
    // A handle may outlive its store.
    auto late = cache->protect_longest(cache->prepare_lookup(id, fa, rows, fa.size() - 1));
    require(static_cast<bool>(late), "late handle");
    cache.reset();
    late.release();
    // Disabled protector or empty binding: nothing happens.
    strix::serve::QueuedPrefixProtection off({}, {});
    strix::serve::GenerationQueue q;
    off.scan(q);
    require(!off.enabled() && off.stats().scans == 0, "disabled protector is inert");
    require(!strix::serve::queued_prefix_protection_lever.default_on, "queue protection default OFF until measured");
}

void budget_rule() {
    using strix::serve::CacheBudgetPolicy;
    using strix::serve::CacheBudgetRule;
    using strix::serve::automatic_cache_budget;
    constexpr std::uint64_t gib = std::uint64_t{1} << 30, mib = std::uint64_t{1} << 20;
    // Legacy rule is the incumbent expression, bit for bit.
    for (std::uint64_t A : {6 * gib, 20 * gib, 64 * gib, 100 * gib})
        for (std::uint64_t R : {4 * gib, 14 * gib + 256 * mib})
            if (A >= R) require(automatic_cache_budget({}, A, R) == std::min({4 * gib, A / 16, A - R}), "legacy == min(4 GiB, A/16, A-R)");
    require(!strix::serve::cache_budget_share_lever.default_on, "share rule default OFF until measured");
    // Late fit, 16 streams at 32K: U 128 GiB, L 40 GiB, A 80 GiB, F 8 GiB, E 1 GiB, S 1 GiB.
    strix::decode_cli::CacheLateFit fit;
    fit.observation_id = 1; fit.limits_consistent = true; fit.aliasing_known = true;
    fit.U = 128 * gib; fit.L = 40 * gib; fit.A = 80 * gib;
    fit.F = 8 * gib; fit.E = 1 * gib; fit.S = 1 * gib;
    fit.checkpoint = 597 * mib; fit.device_unpaid = 0; fit.device_free = 8 * gib;
    fit.context = fit.context_off = 32768; fit.slots = fit.slots_off = 16;
    const auto reserve = *strix::decode_cli::cache_fit_reserve(fit);
    require(reserve == 14 * gib + 256 * mib, "reserve = 4 GiB + F + E + 256 MiB + S");
    const auto legacy = strix::decode_cli::fit_cache_late(fit);
    require(legacy.budget == 4 * gib && legacy.budget / (597 * mib) == 6, "legacy keeps six 597 MiB states");
    fit.budget_policy = CacheBudgetPolicy{CacheBudgetRule::share, 50, 12 * gib};
    const auto share = strix::decode_cli::fit_cache_late(fit);
    require(share.budget == 12 * gib, "share rule at a 12 GiB cap");
    require(16 * 597 * mib <= share.budget, "16 x 597 MiB fit at 12 GiB");
    require(share.budget <= *fit.A - reserve && *fit.L + reserve + share.budget <= *fit.U, "reserves respected");
    fit.budget_policy.cap_bytes = gib;
    require(strix::decode_cli::fit_cache_late(fit).budget == 4 * gib, "share cap is subordinate to the legacy floor");
    // Without a cap the share is half of A - R; at 100 % exactly A - R; never below the legacy minimum.
    fit.budget_policy.cap_bytes = 0;
    const auto half = strix::decode_cli::fit_cache_late(fit).budget;
    require(half == ((*fit.A - reserve) / 2) / 4096 * 4096, "half of what is left after the streams");
    fit.budget_policy.share_pct = 100;
    require(strix::decode_cli::fit_cache_late(fit).budget == (*fit.A - reserve) / 4096 * 4096, "100 % = A - R");
    fit.budget_policy.share_pct = 1;
    require(strix::decode_cli::fit_cache_late(fit).budget == 4 * gib, "never below the legacy minimum");
    // Tight memory: the share is bounded by A - R and the fit still passes.
    fit.A = 20 * gib; fit.budget_policy.share_pct = 50;
    const auto tight = strix::decode_cli::fit_cache_late(fit);
    require(tight.budget != 0 && tight.budget <= *fit.A - reserve, "tight: bounded by A - R");
    fit.A = reserve - 1;
    require(strix::decode_cli::fit_cache_late(fit).budget == 0, "below the reserve: declined as before");
}

void legacy_protected_budget(bool promotion_only = false) {
    const auto id = identity_of("legacy-protected-budget");
    const auto directory = std::filesystem::current_path() / "legacy-protected-budget";
    std::filesystem::create_directories(directory);
    strix::PrefixStateCachePolicy policy{(blocks * segment_bytes + 24) * 2 + 1024, directory, 0};
    policy.legacy = true;
    policy.retention = strix::PrefixCacheRetention::lru;
    const auto a = history(801), b = history(802), c = history(803);
    {
        strix::PrefixStateCache writer(policy, heap);
        require(writer.insert(state(writer, id, c)), "seed disk checkpoint");
        writer.flush();
        require(writer.stats().disk_writes != 0, "disk checkpoint written");
    }
    {
        strix::PrefixStateCache cache(policy, heap);
        require(cache.insert(state(cache, id, a)) && cache.insert(state(cache, id, b)), "fill legacy LRU");
        auto ha = cache.protect_longest(cache.prepare_lookup(id, a, rows, a.size()));
        auto hb = cache.protect_longest(cache.prepare_lookup(id, b, rows, b.size()));
        require(ha && hb, "protect all legacy victims");
        const auto retained = cache.stats().host_bytes;
        if (!promotion_only) require(!cache.insert(state(cache, id, c)), "protected legacy capture refused");
        require(cache.stats().host_bytes == retained && retained <= policy.host_byte_budget, "legacy capture stays within budget");
        const auto loaded = cache.find_longest(id, c, rows, c.size());
        require(loaded && cache.stats().disk_hits == 1, "legacy disk hit still returned");
        require(cache.stats().host_checkpoints == 2 && cache.stats().host_bytes == retained, "disk promotion cannot exceed budget");
    }
    std::filesystem::remove_all(directory);
}

void queue_prepare_host_only() {
    const auto id = identity_of("host-only-preparation");
    const auto directory = std::filesystem::current_path() / "queue-prepare-host-only";
    auto cache = std::make_shared<strix::PrefixStateCache>(strix::PrefixStateCachePolicy{1U << 20, directory, 0}, heap);
    auto tokens = history(901);
    const auto expected = tokens;
    require(cache->insert(state(*cache, id, tokens)), "host checkpoint with disk tier");
    cache->flush();
    // A probing preparation always populates disk, even for absent files.
    const auto probed = cache->prepare_lookup(id, tokens, rows, tokens.size());
    require(!probed.disk.empty(), "control probes the configured disk tier");
    auto binding = strix::serve::prefix_cache_queue_binding(cache, [id] { return id; }, rows, 0,
        [](std::uint64_t n) { return n; });
    auto prepared = binding.prepare(tokens);
    const auto ticket = std::static_pointer_cast<const strix::PrefixLookupTicket>(prepared);
    require(ticket && ticket->disk.empty() && ticket->disk_epoch == 0, "queue preparation performs no disk probes");
    tokens.clear();
    require(ticket->owned && *ticket->owned == expected, "host-only ticket owns its prompt");
    require(static_cast<bool>(binding.protect(prepared)), "host-only ticket protects host checkpoint");
    cache.reset();
    std::filesystem::remove_all(directory);
}

void retry_past_head_misses() {
    const auto id = identity_of("retry-fairness");
    auto cache = std::make_shared<strix::PrefixStateCache>(strix::PrefixStateCachePolicy{1U << 20, {}, 0}, heap);
    require(cache->insert(state(*cache, id, history(16))), "cache only the seventeenth history");
    strix::serve::GenerationQueue queue;
    for (std::uint32_t i = 0; i < 17; ++i) {
        strix::serve::GenerationJob job;
        job.prompt = followup(history(i));
        require(static_cast<bool>(queue.submit(std::move(job))), "enqueue fairness job");
    }
    strix::serve::QueuedPrefixProtectionPolicy policy;
    policy.enabled = true;
    strix::serve::QueuedPrefixProtection protection(policy,
        strix::serve::prefix_cache_queue_binding(cache, [id] { return id; }, rows));
    for (unsigned scan = 0; scan < 8; ++scan) {
        const auto before = protection.stats().attempts;
        protection.scan(queue);
        require(protection.stats().attempts - before <= policy.max_protect, "retry scan remains bounded");
    }
    require(protection.stats().prepared == 17 && protection.stats().held == 1, "later hit is not starved by persistent head misses");
}

void qualification_rows() {
    using namespace strix::decode_cli;
    const auto find = [](std::span<const CacheQualification> table, std::string_view evidence) -> const CacheQualification* {
        for (const auto& row : table) if (row.evidence_id == evidence) return &row;
        return nullptr;
    };
    std::size_t pending = 0;
    for (const auto evidence : {"QPC-SINGLE-32K-v1", "QPC-B8-v1", "QPC-B16-v1"}) {
        const auto* row = find(automatic_prefix_auto_route_qualifications, evidence);
        require(row && row->receipt_required && row->receipt.empty() && !row->qualified(), "pending prefix row unqualified");
        require(row->context_max == 32768, "pending rows cover context 32,768");
        auto flipped = *row;
        flipped.gpu_qualified = flipped.cost_qualified = flipped.resource_qualified = true;
        require(!flipped.qualified(), "flags alone never qualify a receipt-gated row");
        flipped.receipt = "node-receipt";
        require(flipped.qualified(), "flags + receipt qualify");
        ++pending;
    }
    for (const auto evidence : {"QBC-B8-v1", "QBC-B16-v1"}) {
        const auto* row = find(automatic_prefix_batch_cert_route_qualifications, evidence);
        require(row && row->use == CacheUse::batch_l0 && row->receipt_required && !row->qualified(), "pending cert row unqualified");
        ++pending;
    }
    require(pending == 5, "five pending rows");
    // Batch 16 at 32K: the AUTO gate finds a candidate row but no qualified one (empty identity never matches).
    const auto& b16 = *find(automatic_prefix_auto_route_qualifications, "QPC-B16-v1");
    require(!b16.matches(CacheRunner::batch_mtp, 16, 32768, 2048, single_t2048_default_identity), "empty identity never matches");
    // The incumbent single row is unchanged and still the only qualified one.
    std::size_t qualified = 0;
    for (const auto& row : automatic_prefix_auto_route_qualifications) qualified += row.qualified() ? 1U : 0U;
    require(qualified == 1 && automatic_prefix_auto_route_qualifications[0].context_max == 16384, "only single-t2048-default qualified");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2) {
            const std::string test = argv[1];
            if (test == "legacy-capture") legacy_protected_budget();
            else if (test == "legacy-promotion") legacy_protected_budget(true);
            else if (test == "host-only") queue_prepare_host_only();
            else if (test == "fair-retry") retry_past_head_misses();
            else throw std::runtime_error("unknown test case");
            return 0;
        }
        queue_protection();
        cache_edges();
        legacy_protected_budget();
        queue_prepare_host_only();
        retry_past_head_misses();
        budget_rule();
        qualification_rows();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
    std::cout << "prefix queue protection tests: OK\n";
    return 0;
}
