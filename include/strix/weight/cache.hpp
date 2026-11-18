#pragma once
#include "resolve.hpp"
#include <functional>
#include <mutex>
namespace strix::weight {
struct Retirement {
    std::vector<void*> events;
    // Streams whose last-use event could not be recorded. They must be drained
    // (or fenced successfully) before reclamation, including failed conversion.
    std::vector<void*> unresolved_streams;
    // Release only after actual reclamation; quarantine retains this charge.
    std::shared_ptr<void> reservation;
};
struct RuntimeCallbacks {
    std::function<Status(std::uint64_t, void*&)> allocate;
    std::function<void(void*)> free;
    std::function<Status(ExecutionContext&, void*&)> record_event;
    std::function<Status(ExecutionContext&, void*)> wait_event;
    std::function<bool(void*)> event_complete;
    std::function<void(void*)> destroy_event;
    std::function<Status(const WeightView&, const Plan&, void*, ExecutionContext&)> convert;
    // Transfers allocation, fences and budget ownership. Must not throw.
    // Callbacks can block under cache locks: non-reentrant; never call
    // back into this cache. Streams must outlive retirement of unresolved uses.
    std::function<void(void*, Retirement, ExecutionContext)> retire;
    std::function<Status(const Plan&, const WeightView&, const KernelArguments&, void*, void*, std::size_t, ExecutionContext&)> enqueue;
};
struct ExecutionContext {
    Backend backend = Backend::cpu;
    void* stream = nullptr;
    std::uint32_t device_index = 0;
    std::shared_ptr<const RuntimeCallbacks> callbacks;
};
class ImageCache {
public:
    struct State;
    explicit ImageCache(std::uint64_t budget);
    ~ImageCache();
    ImageCache(const ImageCache&) = delete;
    ImageCache& operator=(const ImageCache&) = delete;
    void set_budget(std::uint64_t);
    std::uint64_t evict_completed(std::uint64_t requested);
    std::uint64_t bytes_reserved() const;
private:
    std::shared_ptr<State> state_;
    friend Status acquire_image(ImageCache&, const ImageKey&, const Plan&, const WeightView&, ExecutionContext&, ImageLease&);
};
class ImageLease {
public:
    struct Entry;
    ImageLease() = default;
    ~ImageLease();
    ImageLease(ImageLease&&) noexcept;
    ImageLease& operator=(ImageLease&&) noexcept;
    ImageLease(const ImageLease&) = delete;
    ImageLease& operator=(const ImageLease&) = delete;
    void* data() const noexcept;
    explicit operator bool() const noexcept;
private:
    std::shared_ptr<Entry> entry_;
    ExecutionContext context_;
    bool used_ = false;
    friend Status acquire_image(ImageCache&, const ImageKey&, const Plan&, const WeightView&, ExecutionContext&, ImageLease&);
    friend Status record_last_use(ImageLease&, ExecutionContext&);
};
Status make_image_key(const WeightView&, const Plan&, std::string arch, std::string kernel_abi, ImageKey&);
Status sidecar_identity(const WeightView&, Digest256&);
ExecutionContext cpu_execution_context();
} // namespace strix::weight
