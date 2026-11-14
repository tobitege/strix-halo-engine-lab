// A prepared lookup (prepare_lookup + find_longest(ticket)) returns what the one-step find_longest returns:
// a recorded sequence of lookups is run one-step against one cache and prepared against an identically
// filled cache, and every hit (digest, position, payload bytes) and every counter must agree.
#include "strix/model_content_hash.hpp"
#include "strix/prefix_state_cache.hpp"

#include <chrono>
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
std::byte pattern(std::uint64_t a, std::uint64_t i) { return static_cast<std::byte>((a * 131U + i * 7U) & 255U); }

// Checkpoints of `tokens` at the given block counts, sharing segments as an engine does.
std::vector<Checkpoint> build(strix::PrefixStateCache& cache, const strix::ModelContentIdentity& identity, const Tokens& tokens,
                              std::uint32_t rows, const std::vector<std::uint64_t>& at_blocks, std::uint64_t segment_bytes) {
    std::vector<Checkpoint> out;
    std::vector<std::shared_ptr<const strix::PrefixCacheSegment>> segments;
    auto previous = identity;
    for (std::uint64_t k = 0; k < tokens.size() / rows; ++k) {
        const std::span<const std::uint32_t> block(tokens.data() + k * rows, rows);
        auto segment = std::make_shared<strix::PrefixCacheSegment>();
        segment->identity = identity;
        segment->parent = previous;
        segment->digest = strix::prefix_cache_block_digest(previous, block);
        segment->block_index = k;
        segment->block_rows = rows;
        segment->tokens.assign(block.begin(), block.end());
        segment->payload = cache.allocate(segment_bytes);
        for (std::uint64_t i = 0; i < segment_bytes; ++i) segment->payload.data.get()[i] = pattern(k + tokens[k * rows], i);
        previous = segment->digest;
        segments.push_back(segment);
        bool wanted = false;
        for (const auto blocks : at_blocks) wanted = wanted || blocks == k + 1;
        if (!wanted) continue;
        auto checkpoint = std::make_shared<strix::PrefixCacheCheckpoint>();
        checkpoint->identity = identity;
        checkpoint->digest = previous;
        checkpoint->block_rows = rows;
        checkpoint->position = (k + 1) * rows;
        checkpoint->segments = segments;
        checkpoint->fixed = cache.allocate(24);
        for (std::uint64_t i = 0; i < 24; ++i) checkpoint->fixed.data.get()[i] = pattern(1000 + checkpoint->position, i);
        checkpoint->metadata.assign(3, static_cast<std::byte>(k));
        out.push_back(checkpoint);
    }
    return out;
}
bool same(const Checkpoint& a, const Checkpoint& b) {
    if (!a || !b) return !a && !b;
    if (a->digest != b->digest || a->position != b->position || a->identity != b->identity || a->block_rows != b->block_rows ||
        a->segments.size() != b->segments.size() || a->fixed.bytes != b->fixed.bytes || a->metadata != b->metadata)
        return false;
    if (std::memcmp(a->fixed.data.get(), b->fixed.data.get(), static_cast<std::size_t>(a->fixed.bytes)) != 0) return false;
    for (std::size_t i = 0; i < a->segments.size(); ++i) {
        const auto& x = *a->segments[i];
        const auto& y = *b->segments[i];
        if (x.digest != y.digest || x.tokens != y.tokens || x.payload.bytes != y.payload.bytes ||
            std::memcmp(x.payload.data.get(), y.payload.data.get(), static_cast<std::size_t>(x.payload.bytes)) != 0)
            return false;
    }
    return true;
}
std::string counters(const strix::PrefixStateCacheStats& s) {
    std::string out;
    for (const auto value : {s.lookups, s.plan_misses, s.host_hits, s.disk_hits, s.misses, s.evicted, s.evictions_invalid, s.disk_reads,
                             s.disk_read_failures, s.lookup_tokens, s.hit_tokens, s.restored_tokens, s.host_checkpoints, s.inserted})
        out += std::to_string(value) + ' ';
    return out;
}
Tokens prompt_of(std::size_t count, std::uint32_t seed) {
    Tokens tokens(count);
    std::uint32_t state = seed * 2654435761U + 1U;
    for (auto& token : tokens) {
        state = state * 1664525U + 1013904223U;
        token = state >> 12U;
    }
    return tokens;
}

struct Lookup {
    const char* name;
    strix::ModelContentIdentity identity;
    Tokens prompt;
    std::uint64_t limit;
    bool promote;
    std::string conversation;
    int eligible; // 0: none, 1: accepts, 2: declines checkpoints at position 2*rows
};

void recorded_lookups(const std::filesystem::path& root) {
    constexpr std::uint32_t rows = 4;
    const auto id = identity_of("identity A"), other = identity_of("identity B");
    const Tokens base = prompt_of(42, 1); // ten full blocks + 2
    Tokens branch = base;                 // same first five blocks, then different
    for (std::size_t i = 22; i < branch.size(); ++i) branch[i] ^= 0x5555U;
    Tokens early = base;                  // differs inside the first block
    early[1] ^= 1U;
    const Tokens longer = [&] { Tokens t = base; for (std::uint32_t i = 0; i < 30; ++i) t.push_back(7000U + i); return t; }();

    // Two directories with the same checkpoint files: blocks 2, 5 and 9 of the base prompt, block 7 of the branch.
    const auto fill = [&](const std::filesystem::path& dir) {
        std::filesystem::create_directories(dir);
        strix::PrefixStateCache writer({1ULL << 20, dir, 0}, heap);
        for (const auto& checkpoint : build(writer, id, base, rows, {2, 5, 9}, 64)) require(writer.insert(checkpoint), "insert base");
        for (const auto& checkpoint : build(writer, id, branch, rows, {7}, 64)) require(writer.insert(checkpoint), "insert branch");
        writer.flush();
        require(writer.stats().disk_write_failures == 0 && writer.stats().disk_writes == 4, "disk writes");
    };
    fill(root / "one-step");
    fill(root / "prepared");

    const std::vector<Lookup> recorded = {
        {"cold miss, other identity", other, base, 41, true, "", 0},
        {"miss inside the first block", id, early, 41, true, "", 0},
        {"short limit below the first checkpoint", id, base, 7, true, "", 0},
        {"disk hit without promotion", id, base, 23, false, "", 0},
        {"disk hit at block 5, promoted", id, base, 23, true, "", 0},
        {"host hit at block 5", id, base, 23, true, "chat-1", 0},
        {"longest: block 9 from disk", id, base, 41, true, "chat-1", 0},
        {"host hit at block 9", id, base, 41, true, "", 0},
        {"longer prompt still ends at block 9", id, longer, 71, true, "", 0},
        {"branch: block 7 from disk", id, branch, 41, true, "chat-2", 0},
        {"branch limited to the shared prefix", id, branch, 21, true, "", 0},
        {"predicate accepts host checkpoints", id, base, 41, true, "", 1},
        {"predicate declines block 2, nothing shorter", id, base, 11, true, "", 2},
        {"block 2 without predicate", id, base, 11, true, "", 0},
        {"repeat miss", id, early, 41, false, "", 0},
        {"limit at an exact boundary", id, base, 36, true, "", 0},
        {"limit one below a boundary", id, base, 35, true, "", 0},
        {"empty prompt", id, Tokens{}, 0, true, "", 0},
    };
    for (const bool host_only : {false, true}) {
        // Fresh readers: every hit starts on disk and moves to the host tier as the sequence goes on. The
        // host-only pass has no disk tier and is filled directly.
        strix::PrefixStateCache one({1ULL << 20, host_only ? std::filesystem::path{} : root / "one-step", 0}, heap);
        strix::PrefixStateCache two({1ULL << 20, host_only ? std::filesystem::path{} : root / "prepared", 0}, heap);
        if (host_only)
            for (auto* cache : {&one, &two}) {
                for (const auto& checkpoint : build(*cache, id, base, rows, {2, 5, 9}, 64)) require(cache->insert(checkpoint), "host insert");
                for (const auto& checkpoint : build(*cache, id, branch, rows, {7}, 64)) require(cache->insert(checkpoint), "host insert");
            }
        std::size_t hits = 0;
        for (const auto& lookup : recorded) {
            const auto predicate = [&](const strix::PrefixCacheCheckpoint& c) { return lookup.eligible == 1 || c.position != 2 * rows; };
            const std::function<bool(const strix::PrefixCacheCheckpoint&)> eligible =
                lookup.eligible ? std::function<bool(const strix::PrefixCacheCheckpoint&)>(predicate) : nullptr;
            const auto expected = one.find_longest(lookup.identity, lookup.prompt, rows, lookup.limit, lookup.promote, lookup.conversation, eligible);
            Tokens scratch = lookup.prompt; // the ticket owns its tokens: the caller's buffer may go away
            auto ticket = two.prepare_lookup(lookup.identity, scratch, rows, lookup.limit);
            for (auto& token : scratch) token = ~token;
            scratch.clear();
            scratch.shrink_to_fit();
            const auto copy = ticket; // copies share the token storage
            ticket = {};
            const auto actual = two.find_longest(copy, lookup.promote, lookup.conversation, eligible);
            require(same(expected, actual), std::string("prepared lookup differs: ") + lookup.name);
            require(counters(one.stats()) == counters(two.stats()), std::string("counters differ after: ") + lookup.name);
            hits += expected ? 1U : 0U;
        }
        require(hits >= 9, "the recorded sequence must exercise hits");
        std::cout << "recorded lookups " << recorded.size() << (host_only ? " host-only" : " disk+host") << " hits " << hits
                  << " counters " << counters(two.stats()) << '\n';
    }
    {
        // A ticket prepared before this cache publishes a checkpoint is not trusted for the disk tier: the
        // lookup probes the files itself and finds what the one-step lookup finds.
        const auto dir = root / "stale";
        std::filesystem::create_directories(dir);
        strix::PrefixStateCache cache({1ULL << 20, dir, 0}, heap);
        const auto before = cache.prepare_lookup(id, base, rows, 41);
        require(before.disk.size() == 10 && !cache.find_longest(before), "nothing stored yet");
        {
            strix::PrefixStateCache other_writer({1ULL << 20, dir, 0}, heap);
            for (const auto& checkpoint : build(other_writer, id, base, rows, {5}, 64)) require(other_writer.insert(checkpoint), "insert");
            other_writer.flush();
        }
        // Written by another cache object: this cache's epoch did not move, so the old ticket answers as of
        // the time it was prepared, and a new ticket sees the file.
        require(!cache.find_longest(before), "ticket answers as of its preparation");
        const auto fresh = cache.find_longest(cache.prepare_lookup(id, base, rows, 41), false);
        require(fresh && fresh->position == 20, "new ticket sees the published checkpoint");
        const auto absent = cache.prepare_lookup(id, branch, rows, 41);
        for (const auto& checkpoint : build(cache, id, branch, rows, {7}, 64)) require(cache.insert(checkpoint), "insert");
        cache.flush(); // this cache publishes: tickets prepared before are no longer trusted for the disk tier
        const auto found = cache.find_longest(absent);
        require(found && found->position == 28, "ticket prepared before a publication of this cache");
    }
}

// --bench: lookup cost for long prompts on a miss and on a hit, one-step against prepared. Not a test.
int bench(const std::filesystem::path& root) {
    using Clock = std::chrono::steady_clock;
    const auto us = [](Clock::duration d) { return std::chrono::duration<double, std::micro>(d).count(); };
    constexpr std::uint32_t rows = 256;
    constexpr int rounds = 200;
    const auto id = identity_of("bench identity");
    for (const bool disk : {false, true}) {
        for (const std::size_t tokens : {std::size_t{8192}, std::size_t{32768}, std::size_t{131072}}) {
            const auto dir = root / ("bench-" + std::to_string(tokens));
            if (disk) std::filesystem::create_directories(dir);
            strix::PrefixStateCache cache({64ULL << 20, disk ? dir : std::filesystem::path{}, 0}, heap);
            const Tokens stored = prompt_of(tokens, 3);
            for (const auto& checkpoint : build(cache, id, stored, rows, {tokens / rows / 2}, 16)) require(cache.insert(checkpoint), "insert");
            cache.flush();
            Tokens miss = stored;
            miss[0] ^= 1U; // nothing matches: every block is probed
            for (const bool hit : {false, true}) {
                const Tokens& prompt = hit ? stored : miss;
                double one_step = 0, prepare = 0, find = 0;
                for (int round = 0; round < rounds; ++round) {
                    auto begin = Clock::now();
                    const auto a = cache.find_longest(id, prompt, rows, prompt.size(), false);
                    one_step += us(Clock::now() - begin);
                    begin = Clock::now();
                    const auto ticket = cache.prepare_lookup(id, prompt, rows, prompt.size());
                    prepare += us(Clock::now() - begin);
                    begin = Clock::now();
                    const auto b = cache.find_longest(ticket, false);
                    find += us(Clock::now() - begin);
                    require(same(a, b) && static_cast<bool>(a) == hit, "bench lookups disagree");
                }
                std::cout << "lookup bench " << (disk ? "disk+host" : "host-only") << " tokens " << tokens << " blocks " << tokens / rows
                          << (hit ? " hit " : " miss") << " | one_step_us " << one_step / rounds << " | prepared: before_admission_us "
                          << prepare / rounds << " on_path_us " << find / rounds << '\n';
            }
        }
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    const auto root = std::filesystem::temp_directory_path() /
                      ("strix-prefix-ticket-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    int status = 0;
    try {
        std::filesystem::create_directories(root);
        if (argc >= 2 && std::string(argv[1]) == "--bench") status = bench(root);
        else {
            recorded_lookups(root);
            std::cout << "prefix lookup ticket tests passed\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "prefix lookup ticket tests failed: " << error.what() << '\n';
        status = 1;
    }
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
    return status;
}
