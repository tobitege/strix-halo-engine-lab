#pragma once
// CPU-only envelope for the existing Engine export_rows/import_rows
// representation.
#include <cstdint>
#include "strix/file_snapshot.hpp"
#include "strix/session_state_cache.hpp"
#include "strix/session/prefix_cache.hpp"
#include <filesystem>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <vector>
namespace strix::minimax::session {
using Bytes = std::vector<std::uint8_t>;
using Exchange = std::function<Bytes(std::span<const std::uint8_t>)>;
std::string sha(std::span<const std::uint8_t>);
std::string compat_record(std::map<std::string, std::string>);
bool compatibility_environment_variable(const std::string &);
enum class AssetLoader { Pack, Draft };
std::string compatibility(std::map<std::string, std::string> effective,
                          const std::filesystem::path &model,
                          const std::filesystem::path &executable,
                          AssetLoader loader = AssetLoader::Pack);
std::string compatibility(std::map<std::string, std::string> effective,
                          const LoaderSnapshot& snapshot,
                          const std::filesystem::path& executable);
// Hash-only CPU path shared by the rank CLI and focused tests. Verifies two
// complete passes, including executable mappings and supplied staged inputs.
struct PrewarmReceipt {
  double wall_seconds = 0, second_seconds = 0;
  std::uint64_t hits = 0;
};
PrewarmReceipt prewarm(const std::filesystem::path &model,
                      const std::filesystem::path &draft,
                      const std::filesystem::path &draft_exl3,
                      const std::filesystem::path &executable,
                      const std::vector<std::string> &extra_files = {});
void validate_id(const std::string &);
using Frontier = ::strix::SessionFrontier;
using Section = ::strix::SessionSection;
inline Frontier initial_frontier() {
  Frontier f; f.pending = 0; f.lineage = "minimax-single-row-committed-v1"; return f;
}
// LEGACY CODEC/accessor adapter only. All state is in the shared base envelope.
// References bind to this object's base on copy/move; no second history/compat store.
struct Checkpoint : ::strix::SessionCheckpoint {
  std::string& compat = compatibility.record;
  std::vector<std::uint32_t>& history = frontier.tokens;
  Checkpoint() { compatibility.family = "minimax"; world = 2; frontier = initial_frontier(); }
  Checkpoint(const Checkpoint& c) : ::strix::SessionCheckpoint(c) {}
  Checkpoint(Checkpoint&& c) noexcept : ::strix::SessionCheckpoint(std::move(c)) {}
  Checkpoint& operator=(const Checkpoint& c) { ::strix::SessionCheckpoint::operator=(c); return *this; }
  Checkpoint& operator=(Checkpoint&& c) noexcept { ::strix::SessionCheckpoint::operator=(std::move(c)); return *this; }
};
// Section ordering follows the existing engine cache-row byte ordering.
std::vector<Section> split_state(std::span<const std::uint8_t>,
                                 std::uint64_t positions);
Bytes join_state(const std::vector<Section> &);
Bytes encode(const Checkpoint &);
Checkpoint decode(std::span<const std::uint8_t>, const std::string &compat,
                  std::uint32_t rank, std::uint32_t world,
                  std::uint64_t max_bytes);
std::uint64_t available_memory();
bool memory_admitted(std::uint64_t parked, std::uint64_t payload,
                     std::uint64_t budget, std::uint64_t available);
std::string state_digest(
    const Checkpoint
        &); // excludes id/generation; includes frontier/history/sections
void atomic_write(const std::filesystem::path &, std::span<const std::uint8_t>);
Bytes read(const std::filesystem::path &, std::uint64_t max_bytes);
void agree(bool success, const Exchange &, const std::string &phase);
struct Summary {
  std::string id;
  Frontier frontier;
  std::uint64_t bytes = 0;
};
struct Stats {
  std::uint64_t bytes = 0, entries = 0, evictions = 0, disk_bytes = 0;
};
class Cache {
public:
  Cache(std::string compat, std::uint32_t rank, std::uint32_t world,
        std::uint64_t budget, std::filesystem::path directory,
        Exchange exchange, std::uint64_t max_parked = 8,
        std::uint64_t disk_budget = UINT64_MAX,
        std::shared_ptr<PrefixStateCache> owner = {});
  void prepare_capture(
      std::uint64_t payload_bytes); // evict LRU before reserving peak
  void admit(std::uint64_t payload_bytes)
      const;                // BEFORE allocating capture/restore buffers
  void prepare_resume(const std::string &id,
                      std::uint64_t payload_bytes); // evict other LRU entries before a RAM restore peak
  void capture(Checkpoint); // RAM checkpoint; LRU, no mutation when a single
                            // entry exceeds budget
  void checkpoint(
      Checkpoint); // rank file durable, receipts exchanged, group manifest last
  Checkpoint resume(const std::string &, bool disk);
  void erase(const std::string &);
  std::vector<std::string> list() const;
  Stats stats() const;
  std::vector<Summary> describe(bool include_disk = true) const;
  bool in_ram(const std::string& id) const { return ram_.find(id) != nullptr; }
  bool on_disk(const std::string& id) const;
  const std::shared_ptr<PrefixStateCache>& owner() const noexcept { return ram_.owner(); }

private:
  std::string compat_;
  std::uint32_t rank_, world_;
  std::uint64_t max_parked_, disk_budget_;
  std::uint64_t budget_, evictions_ = 0;
  std::filesystem::path directory_;
  Exchange exchange_;
  strix::session::NamedStateCache ram_;
  PrefixStateCache::Reservation working_;
  void identify(Checkpoint &);
  void erase_locked(const std::string &);
};
} // namespace strix::minimax::session
