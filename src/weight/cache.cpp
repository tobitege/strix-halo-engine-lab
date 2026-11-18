#include <strix/weight/cache.hpp>
#include <strix/weight/source.hpp>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <map>
namespace strix::weight {
namespace {
std::string key_bytes(const ImageKey& k) {
    std::string s;
    auto number = [&](std::uint64_t n) { for (int i = 0; i < 8; ++i) { s.push_back(static_cast<char>(n & 255)); n >>= 8; } };
    auto string = [&](std::string_view v) { number(v.size()); s.append(v); };
    auto digest = [&](const Digest256& d) { s.append(reinterpret_cast<const char*>(d.bytes.data()), d.bytes.size()); };
    digest(k.source); digest(k.sidecars); digest(k.metadata); number(k.tensor); number(k.payload.offset); number(k.payload.bytes);
    for (auto n : k.byte_strides) number(n);
    number(k.has_byte_strides); for (auto n : k.shape.dims) number(n); number(k.shape.rank);
    for (Id id : {k.storage, k.image, k.arithmetic}) { string(id.name); number(id.revision); }
    number(static_cast<unsigned>(k.backend)); string(k.arch); string(k.kernel_abi);
    return s;
}
bool callbacks_valid(const ExecutionContext& c) {
    return c.callbacks && c.callbacks->allocate && c.callbacks->free && c.callbacks->record_event &&
        c.callbacks->wait_event && c.callbacks->event_complete && c.callbacks->destroy_event && c.callbacks->convert && c.callbacks->retire;
}
}
struct ImageLease::Entry {
    struct Charge {
        std::shared_ptr<std::atomic<std::uint64_t>> reserved;
        std::uint64_t bytes = 0;
        ~Charge() { if (reserved) reserved->fetch_sub(bytes); }
    };
    struct Charges { std::shared_ptr<Charge> image, workspace; };
    std::shared_ptr<Charges> charges;
    std::mutex mutex;
    std::condition_variable ready;
    bool building = true, poisoned = false;
    Status status{Error::ok, {}};
    void* allocation = nullptr;
    std::shared_ptr<const RuntimeCallbacks> callbacks;
    ExecutionContext context;
    std::vector<void*> events;
    std::vector<void*> unresolved_streams;
    void* readiness = nullptr;
    bool has_readiness = false;
    ~Entry() {
        if (allocation) callbacks->retire(allocation, {std::move(events), std::move(unresolved_streams), std::move(charges)}, context);
        else if (callbacks) for (auto event : events) callbacks->destroy_event(event);
    }
    bool completed() {
        std::lock_guard lock(mutex);
        release_workspace();
        return !building && !poisoned && std::all_of(events.begin(), events.end(), [&](void* e) { return callbacks->event_complete(e); });
    }
    void release_workspace() {
        if (has_readiness && callbacks->event_complete(readiness)) charges->workspace.reset();
    }
};
struct ImageCache::State {
    mutable std::mutex mutex;
    std::uint64_t budget = 0;
    std::shared_ptr<std::atomic<std::uint64_t>> reserved = std::make_shared<std::atomic<std::uint64_t>>(0);
    bool bound = false;
    Backend backend{};
    std::uint32_t device_index = 0;
    std::map<std::string, std::shared_ptr<ImageLease::Entry>> entries;
    std::uint64_t evict(std::uint64_t wanted) {
        std::uint64_t freed = 0;
        for (auto it = entries.begin(); it != entries.end() && freed < wanted;) {
            if (it->second.use_count() == 1 && it->second->completed()) {
                const auto before = reserved->load(); it = entries.erase(it);
                freed += before - reserved->load();
            } else ++it;
        }
        return freed;
    }
};
ImageCache::ImageCache(std::uint64_t budget) : state_(std::make_shared<State>()) { state_->budget = budget; }
ImageCache::~ImageCache() = default;
void ImageCache::set_budget(std::uint64_t budget) {
    std::lock_guard lock(state_->mutex); state_->budget = budget;
    const auto reserved = state_->reserved->load(); if (reserved > budget) state_->evict(reserved - budget);
}
std::uint64_t ImageCache::evict_completed(std::uint64_t requested) { std::lock_guard lock(state_->mutex); return state_->evict(requested); }
std::uint64_t ImageCache::bytes_reserved() const { return state_->reserved->load(); }
ImageLease::~ImageLease() { if (entry_ && !used_) { auto ignored = record_last_use(*this, context_); (void)ignored; } }
ImageLease::ImageLease(ImageLease&& other) noexcept : entry_(std::move(other.entry_)), context_(std::move(other.context_)), used_(other.used_) {}
ImageLease& ImageLease::operator=(ImageLease&& other) noexcept {
    if (this != &other) { ImageLease old(std::move(*this)); entry_ = std::move(other.entry_); context_ = std::move(other.context_); used_ = other.used_; }
    return *this;
}
void* ImageLease::data() const noexcept { return entry_ ? entry_->allocation : nullptr; }
ImageLease::operator bool() const noexcept { return entry_ != nullptr; }
Status record_last_use(ImageLease& lease, ExecutionContext& context) {
    if (!lease.entry_ || context.backend != lease.context_.backend || context.device_index != lease.context_.device_index ||
        context.callbacks != lease.context_.callbacks) return {Error::unsupported_backend, "lease/context backend, device or callbacks differ"};
    auto& entry = *lease.entry_;
    std::lock_guard lock(entry.mutex);
    void* event = nullptr;
    Status status{Error::device_failure, "last-use event callback failed"};
    try { status = context.callbacks->record_event(context, event); } catch (...) {}
    if (!status) {
        entry.poisoned = true;
        if (std::find(entry.unresolved_streams.begin(), entry.unresolved_streams.end(), context.stream) == entry.unresolved_streams.end())
            entry.unresolved_streams.push_back(context.stream);
        return status;
    }
    entry.release_workspace();
    for (auto it = entry.events.begin(); it != entry.events.end();) {
        if (context.callbacks->event_complete(*it)) { if (entry.has_readiness && *it == entry.readiness) entry.has_readiness = false; context.callbacks->destroy_event(*it); it = entry.events.erase(it); }
        else ++it;
    }
    entry.events.push_back(event); lease.used_ = true;
    return {Error::ok, {}};
}
Status acquire_image(ImageCache& cache, const ImageKey& key, const Plan& plan, const WeightView& view,
                     ExecutionContext& context, ImageLease& result) {
    if (!callbacks_valid(context)) return {Error::unsupported_backend, "backend image runtime is not registered"};
    if (key.backend != context.backend || plan.selected.backend != context.backend)
        return {Error::unsupported_backend, "image key, plan and execution backend differ"};
    if (!(key.storage == view.storage) || !(key.storage == plan.selected.storage) || !(key.image == plan.selected.image) ||
        !(key.arithmetic == plan.selected.arithmetic) || key.tensor != view.tensor || key.shape.dims != view.shape.dims ||
        key.shape.rank != view.shape.rank || key.payload.offset != view.payload.range.offset || key.payload.bytes != view.payload.range.bytes ||
        key.byte_strides != view.byte_strides || key.has_byte_strides != view.has_byte_strides)
        return {Error::malformed, "image key does not describe tensor/plan"};
    if (!view.payload.source) return {Error::stale_source, "source is absent"};
    ImageKey expected{};
    auto identity_status = make_image_key(view, plan, key.arch, key.kernel_abi, expected);
    if (!identity_status) return identity_status;
    if (!(expected == key)) return {Error::stale_source, "source, sidecar or metadata identity differs"};
    const auto bytes = plan.selected.resources.image_bytes;
    const auto temporary = plan.selected.resources.workspace_bytes;
    if (bytes > std::numeric_limits<std::uint64_t>::max() - temporary) return {Error::memory_budget, "reservation overflow"};
    const auto total = bytes + temporary;
    std::shared_ptr<ImageLease::Entry> entry;
    bool creator = false;
    const auto serialized = key_bytes(key);
    {
        std::lock_guard lock(cache.state_->mutex);
        if (cache.state_->bound && (cache.state_->backend != context.backend || cache.state_->device_index != context.device_index))
            return {Error::unsupported_backend, "image cache belongs to a different backend/device"};
        cache.state_->bound = true; cache.state_->backend = context.backend; cache.state_->device_index = context.device_index;
        auto it = cache.state_->entries.find(serialized);
        if (it != cache.state_->entries.end()) entry = it->second;
        else {
            if (total > cache.state_->budget) return {Error::memory_budget, "image exceeds cache budget"};
            const auto reserved = cache.state_->reserved->load();
            if (reserved > cache.state_->budget - total) cache.state_->evict(reserved - (cache.state_->budget - total));
            if (cache.state_->reserved->load() > cache.state_->budget - total) return {Error::memory_budget, "in-use or retiring images prevent reservation"};
            entry = std::make_shared<ImageLease::Entry>(); entry->callbacks = context.callbacks; entry->context = context;
            entry->charges = std::make_shared<ImageLease::Entry::Charges>();
            entry->charges->image = std::make_shared<ImageLease::Entry::Charge>();
            entry->charges->image->reserved = cache.state_->reserved; entry->charges->image->bytes = 0;
            entry->charges->workspace = std::make_shared<ImageLease::Entry::Charge>();
            entry->charges->workspace->reserved = cache.state_->reserved; entry->charges->workspace->bytes = 0;
            cache.state_->entries.emplace(serialized, entry); cache.state_->reserved->fetch_add(total);
            entry->charges->image->bytes = bytes; entry->charges->workspace->bytes = temporary; creator = true;
        }
    }
    if (creator) {
        Status status{Error::device_failure, "image runtime threw an exception"};
        try {
        status = context.callbacks->allocate(bytes, entry->allocation);
        if (status && bytes && !entry->allocation) status = {Error::device_failure, "allocator returned null image"};
        if (status) status = context.callbacks->convert(view, plan, entry->allocation, context);
        if (status) {
            void* event = nullptr; status = context.callbacks->record_event(context, event);
            if (status) { entry->events.push_back(event); entry->readiness = event; entry->has_readiness = true; }
        }
        } catch (...) { status = {Error::device_failure, "image allocation/conversion/event failure"}; }
        {
            std::lock_guard lock(entry->mutex); entry->status = status; entry->building = false;
            if (!status && entry->allocation) entry->unresolved_streams.push_back(context.stream);
            entry->release_workspace();
        }
        {
            std::lock_guard lock(cache.state_->mutex);
            if (!status) cache.state_->entries.erase(serialized);
        }
        entry->ready.notify_all();
    }
    {
        std::unique_lock lock(entry->mutex); entry->ready.wait(lock, [&] { return !entry->building; });
        if (!entry->status) return entry->status;
        if (entry->callbacks != context.callbacks) return {Error::unsupported_backend, "cache runtime differs"};
        if (entry->has_readiness) { auto status = context.callbacks->wait_event(context, entry->readiness); if (!status) return status; }
    }
    ImageLease lease; lease.entry_ = std::move(entry); lease.context_ = context; result = std::move(lease);
    return {Error::ok, {}};
}
Status enqueue(const Plan& plan, const WeightView& view, const KernelArguments& args, void* output, void* workspace,
               std::size_t workspace_bytes, ExecutionContext& context) {
    if (context.backend != plan.selected.backend) return {Error::unsupported_backend, "enqueue context backend differs"};
    if (!(view.storage == plan.selected.storage)) return {Error::unknown_format, "enqueue storage differs"};
    if (workspace_bytes < plan.workspace_bytes || (plan.workspace_bytes && !workspace)) return {Error::memory_budget, "workspace missing or too short"};
    if (!args.data || !args.bytes || !args.schema || !args.abi_revision || !output) return {Error::malformed, "invalid typed argument envelope"};
    if (!context.callbacks || !context.callbacks->enqueue) return {Error::unsupported_arithmetic, "backend has no registered launcher"};
    return context.callbacks->enqueue(plan, view, args, output, workspace, workspace_bytes, context);
}
}
