#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix::session {

// Device-neutral owner table; Page may own a backend allocation/descriptor.
// Allocation and cloning stay in backend callbacks. No byte layout, kernels or
// numerical decisions live here. All accesses are externally serialized and
// quiescent; a backend must finish readers/writers before capture or teardown.
// Callbacks must not retain mutable aliases, mutate their const source, or expose
// aliases to page subresources. A clone owns physically independent writable
// storage, not merely a copy of a descriptor pointing at the same allocation.
template<class Page> class KvPages {
public:
  using Clone = std::function<std::unique_ptr<Page>(const Page&)>;
  class Snapshot {
  public:
    [[nodiscard]] std::size_t size() const noexcept { return pages_.size(); }
    [[nodiscard]] const Page& read(std::size_t i) const { return *pages_.at(i); }
    [[nodiscard]] const void* owner(std::size_t i) const { return pages_.at(i).get(); }
  private:
    std::vector<std::shared_ptr<Page>> pages_;
    friend class KvPages;
  };

  explicit KvPages(Clone clone) : clone_(std::move(clone)) {
    if (!clone_) throw std::invalid_argument("page clone callback required");
  }
  KvPages(const Snapshot& snapshot, Clone clone) : KvPages(std::move(clone)) {
    pages_ = snapshot.pages_; // owners only; zero prefix bytes cloned
  }
  [[nodiscard]] Snapshot capture() const {
    Snapshot snapshot;
    snapshot.pages_ = pages_;
    return snapshot;
  }
  [[nodiscard]] std::size_t size() const noexcept { return pages_.size(); }
  [[nodiscard]] const Page& read(std::size_t i) const { return *pages_.at(i); }
  [[nodiscard]] const void* owner(std::size_t i) const { return pages_.at(i).get(); }
  // Transfer exclusive ownership; no externally retained writable alias allowed.
  void append(std::unique_ptr<Page> page) {
    if (!page) throw std::invalid_argument("null page");
    pages_.emplace_back(std::move(page));
  }
  // A live snapshot ALWAYS counts as an owner, even after source/siblings die.
  // No mutable shared_ptr escapes. First write detaches exactly this page; later
  // lane-local writes reuse it until another snapshot/fork shares it again.
  // Clone/allocation failure leaves owners unchanged. A throwing writer may
  // partially modify this lane's detached page: Engine must roll back/invalidate.
  template<class Writer> void write(std::size_t i, Writer&& writer) {
    auto& page = pages_.at(i);
    if (page.use_count() != 1) {
      auto detached = clone_(*page);
      if (!detached) throw std::runtime_error("page clone failed");
      std::shared_ptr<Page> owner(std::move(detached));
      page.swap(owner);
    }
    std::forward<Writer>(writer)(*page);
  }
private:
  Clone clone_;
  std::vector<std::shared_ptr<Page>> pages_;
};

} // namespace strix::session
