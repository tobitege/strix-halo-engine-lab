// Standalone white-box host test: compile this TU instead of bpe_tokenizer.cpp.
// A miss observer measures reuse and synchronizes cold misses without timing assertions.
#include <atomic>
#include <barrier>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

namespace {
std::atomic<unsigned> misses{0};
bool prefill = false;
bool collisions = false;
std::barrier<>* rendezvous = nullptr;
template<class Table> void cache_miss(const std::shared_ptr<Table>& table);
}
#define STRIX_BPE_CACHE_TEST_MISS(table) cache_miss(table)
#include "../src/serve/bpe_tokenizer.cpp"
#undef STRIX_BPE_CACHE_TEST_MISS

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<class Table> void cache_miss(const std::shared_ptr<Table>& table) {
    const unsigned miss = misses.fetch_add(1) + 1U;
    if (prefill) {
        prefill = false;
        for (std::size_t i = 0; i < Table::limit - 1U; ++i) {
            const auto piece = std::to_string(i);
            require(table->insert(std::hash<std::string_view>{}(piece), piece, {}), "prefill failed");
        }
    }
    if (collisions) {
        collisions = false;
        Table collision_table;
        std::vector<std::future<void>> jobs;
        for (std::uint32_t i = 0; i < 8U; ++i) jobs.push_back(std::async(std::launch::async, [&, i] {
            // Same complete hash, including wraparound from the last slot. Duplicate publishers race.
            for (std::uint32_t j = 0; j < 8U; ++j) {
                const auto id = (i + j) % 8U;
                require(collision_table.insert(Table::slots - 1U, std::to_string(id), {id}), "collision insert");
            }
        }));
        for (auto& job : jobs) job.get();
        require(collision_table.size.load() == 8U, "duplicate collision entries");
        for (std::uint32_t i = 0; i < 8U; ++i) {
            const auto* entry = collision_table.find(Table::slots - 1U, std::to_string(i));
            require(entry && entry->ids == std::vector<std::uint32_t>{i}, "collision changed IDs");
        }
        require(!collision_table.find(Table::slots - 1U, "absent"), "collision false hit");
    }
    // The first four misses must retain the old table before any can replace it. Count per case:
    // std::async may reuse worker threads, so thread-local state would survive between cases.
    if (rendezvous && miss <= 4U) rendezvous->arrive_and_wait();
}

auto tokenizer() {
    std::vector<std::string> tokens;
    for (char c = 'a'; c <= 'z'; ++c) tokens.emplace_back(1U, c);
    tokens.emplace_back("\xC4\xA0"); // byte-level space
    tokens.emplace_back("aa");
    return strix::serve::BpeTokenizer::from_parts(std::move(tokens), std::vector<std::int32_t>(28U, 1),
                                                  std::vector<std::string>{"a a"}, "qwen35");
}

std::vector<std::uint32_t> expected() {
    std::vector<std::uint32_t> ids;
    for (int i = 0; i < 16; ++i) ids.insert(ids.end(), {26U, 27U, 27U});
    return ids;
}

std::string repeated_piece() {
    std::string text;
    for (int i = 0; i < 16; ++i) text += " aaaa";
    return text;
}

void rollover_reuses_piece() {
    const auto bpe = tokenizer();
    prefill = true;
    (void)bpe->encode(" seed", false); // populate exactly limit entries, without replacing the table
    misses = 0;
    require(bpe->encode(repeated_piece(), false) == expected(), "rollover IDs");
    require(misses.load() == 1U, "repeated pieces were recomputed after rollover");
    (void)bpe->encode(repeated_piece(), false);
    require(misses.load() == 1U, "new segment did not reuse rollover insertion");
}

void concurrent_reuse(bool full) {
    const auto bpe = tokenizer();
    if (full) {
        prefill = true;
        (void)bpe->encode(" seed", false);
    }
    misses = 0;
    std::barrier barrier(4);
    rendezvous = &barrier;
    std::vector<std::future<std::vector<std::uint32_t>>> jobs;
    for (int i = 0; i < 4; ++i)
        jobs.push_back(std::async(std::launch::async, [&] { return bpe->encode(repeated_piece(), false); }));
    for (auto& job : jobs) require(job.get() == expected(), "concurrent IDs");
    rendezvous = nullptr;
    require(misses.load() == 4U, "concurrent segments did not reuse their first miss");
    (void)bpe->encode(repeated_piece(), false);
    require(misses.load() == 4U, "current table was not populated");
}
}

int main() {
    try {
        collisions = true;
        (void)tokenizer()->encode(" aaaa", false);
        rollover_reuses_piece();
        concurrent_reuse(false);
        concurrent_reuse(true);
    } catch (const std::exception& error) {
        std::cerr << "BPE cache tests failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "BPE cache tests passed\n";
}
