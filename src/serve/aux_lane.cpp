#include "strix/serve/aux_lane.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>

namespace strix::serve::aux {
namespace {

double ms_between(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

std::string json_string(std::string_view text) {
    std::string out = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char escaped[8];
            std::snprintf(escaped, sizeof escaped, "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
            out += escaped;
        } else {
            out += c;
        }
    }
    return out + "\"";
}

std::string number(double value) {
    char text[40];
    std::snprintf(text, sizeof text, "%.3f", value);
    return text;
}

struct Pending {
    AuxJob job;
    std::promise<AuxResult> promise;
    std::vector<std::size_t> order;
    std::size_t next = 0;
    Clock::time_point queued;
    std::uint32_t fallbacks = 0;
    std::string errors;
    bool inner = false; // one attempt made for run_now, which keeps the job-level counters itself
};

} // namespace

std::string_view kind_name(JobKind kind) noexcept {
    switch (kind) {
    case JobKind::score: return "score";
    case JobKind::vision: return "vision";
    case JobKind::embed: return "embed";
    }
    return "unknown";
}

std::string_view unit_name(UnitKind unit) noexcept {
    switch (unit) {
    case UnitKind::gpu: return "gpu";
    case UnitKind::npu: return "npu";
    case UnitKind::cpu: return "cpu";
    }
    return "unknown";
}

UnitKind parse_unit(std::string_view text) {
    if (text == "gpu") return UnitKind::gpu;
    if (text == "npu") return UnitKind::npu;
    if (text == "cpu") return UnitKind::cpu;
    throw std::invalid_argument("aux unit: expected gpu, npu or cpu, got " + std::string(text));
}

Config with_environment(Config base) {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    const char* value = std::getenv("STRIX_AUX_UNIT");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    if (!value || !*value) return base;
    const std::string_view text(value);
    if (text == "auto") {
        base.force.reset();
        return base;
    }
    try {
        base.force = parse_unit(text);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("STRIX_AUX_UNIT: expected auto, gpu, npu or cpu, got " + std::string(text));
    }
    return base;
}

std::vector<std::size_t> plan(const Config& config, std::span<const Executor> executors, JobKind kind,
                              std::string_view identity, bool gpu_busy) {
    std::vector<UnitKind> units;
    if (config.force) {
        units.push_back(*config.force);
    } else {
        for (const auto unit : config.order[static_cast<std::size_t>(kind)])
            if (std::find(units.begin(), units.end(), unit) == units.end()) units.push_back(unit);
        // R-aux decides GPU against NPU whenever both durations were measured; otherwise the configured order stands.
        const auto& rates = config.rates[static_cast<std::size_t>(kind)];
        const auto gpu = std::find(units.begin(), units.end(), UnitKind::gpu);
        const auto npu = std::find(units.begin(), units.end(), UnitKind::npu);
        if (gpu != units.end() && npu != units.end() && rates.gpu_seconds > 0.0 && rates.npu_seconds > 0.0 &&
            config.npu_side_cost > 0.0) {
            const bool npu_first = gpu_busy && rates.npu_seconds / rates.gpu_seconds < config.gpu_side_cost / config.npu_side_cost;
            if (npu_first != (npu < gpu)) std::iter_swap(gpu, npu);
        }
    }
    std::vector<std::size_t> out;
    for (const auto unit : units)
        for (std::size_t i = 0; i < executors.size(); ++i) {
            const auto& e = executors[i];
            if (e.kind == kind && e.unit == unit && (identity.empty() || e.identity == identity)) out.push_back(i);
        }
    return out;
}

struct AuxLane::State {
    Config config;
    std::vector<Executor> executors;
    mutable std::mutex mutex;
    std::condition_variable wake;
    std::deque<std::unique_ptr<Pending>> waiting;
    bool stopping = false;
    Counters totals;
    std::vector<std::thread> workers;
    std::vector<std::string> worker_units;

    [[nodiscard]] bool busy() const { return config.gpu_busy && config.gpu_busy(); }

    // Takes the first waiting job whose next executor is on `unit_id` with the given threading. Lock held.
    std::unique_ptr<Pending> take(std::string_view unit_id, bool threaded) {
        for (auto it = waiting.begin(); it != waiting.end(); ++it) {
            const auto& e = executors[(*it)->order[(*it)->next]];
            if (e.threaded != threaded || e.unit_id != unit_id) continue;
            auto out = std::move(*it);
            waiting.erase(it);
            return out;
        }
        return nullptr;
    }

    // One attempt on the job's next executor, without the lock. True when the job is finished (value or error set).
    bool step(Pending& p) {
        const std::size_t index = p.order[p.next];
        const auto& e = executors[index];
        const auto start = Clock::now();
        if (start > p.job.deadline) {
            {
                const std::lock_guard lock(mutex);
                ++totals.deadline;
            }
            p.promise.set_exception(std::make_exception_ptr(AuxError(
                AuxError::Code::deadline, "aux lane: " + std::string(kind_name(p.job.kind)) + " job passed its deadline before it started")));
            return true;
        }
        AuxResult result;
        std::string error;
        try {
            result = e.run(p.job);
        } catch (const std::exception& ex) {
            error = ex.what();
            if (error.empty()) error = "failed";
        } catch (...) {
            error = "failed";
        }
        const auto end = Clock::now();
        if (error.empty()) {
            result.identity = e.identity;
            result.unit = e.unit;
            result.unit_id = e.unit_id;
            result.fallbacks = p.fallbacks;
            result.queued_ms = ms_between(p.queued, start);
            result.run_ms = ms_between(start, end);
            {
                const std::lock_guard lock(mutex);
                auto& count = totals.units[index];
                ++count.jobs;
                count.busy_ms += result.run_ms;
                if (!p.inner) ++totals.completed;
            }
            p.promise.set_value(std::move(result));
            return true;
        }
        p.errors += (p.errors.empty() ? "" : "; ") + std::string(unit_name(e.unit)) + ":" + e.unit_id + ": " + error;
        ++p.next;
        const bool last = p.next == p.order.size();
        {
            const std::lock_guard lock(mutex);
            auto& count = totals.units[index];
            ++count.failed;
            count.busy_ms += ms_between(start, end);
            if (!p.inner) ++(last ? totals.failed : totals.fallbacks);
        }
        if (!last) {
            ++p.fallbacks;
            return false;
        }
        p.promise.set_exception(std::make_exception_ptr(
            AuxError(AuxError::Code::failed, "aux lane: " + std::string(kind_name(p.job.kind)) + " job failed on every unit (" + p.errors + ")")));
        return true;
    }

    void requeue(std::unique_ptr<Pending> p) {
        {
            const std::lock_guard lock(mutex);
            if (stopping) {
                p->promise.set_exception(std::make_exception_ptr(AuxError(AuxError::Code::stopped, "aux lane: stopped")));
                return;
            }
            waiting.push_front(std::move(p));
        }
        wake.notify_all();
    }

    void work(const std::string& unit_id) {
        while (true) {
            std::unique_ptr<Pending> p;
            {
                std::unique_lock lock(mutex);
                wake.wait(lock, [&] { return stopping || (p = take(unit_id, true)) != nullptr; });
                if (!p) return;
            }
            if (!step(*p)) requeue(std::move(p));
        }
    }

    // Queues with the lock held; starts nothing.
    std::future<AuxResult> enqueue(std::unique_ptr<Pending> p, std::unique_lock<std::mutex>& lock) {
        if (stopping) throw AuxError(AuxError::Code::stopped, "aux lane: stopped");
        if (waiting.size() >= config.capacity) {
            ++totals.queue_full;
            throw AuxError(AuxError::Code::queue_full,
                           "aux lane: " + std::to_string(waiting.size()) + " jobs are waiting (capacity " + std::to_string(config.capacity) + ")");
        }
        auto future = p->promise.get_future();
        waiting.push_back(std::move(p));
        totals.waiting_peak = std::max(totals.waiting_peak, waiting.size());
        lock.unlock();
        wake.notify_all();
        return future;
    }

    std::vector<std::size_t> planned(const AuxJob& job) {
        auto order = plan(config, executors, job.kind, job.identity, busy());
        if (order.empty()) {
            const std::lock_guard lock(mutex);
            ++totals.no_executor;
            throw AuxError(AuxError::Code::no_executor,
                           "aux lane: no executor for " + std::string(kind_name(job.kind)) +
                               (job.identity.empty() ? std::string() : " with identity " + job.identity) +
                               (config.force ? " on the forced unit " + std::string(unit_name(*config.force)) : std::string()));
        }
        return order;
    }
};

AuxLane::AuxLane(Config config) : state_(std::make_unique<State>()) {
    if (config.capacity == 0) throw std::invalid_argument("aux lane: capacity must be at least 1");
    state_->config = std::move(config);
}

AuxLane::~AuxLane() {
    std::deque<std::unique_ptr<Pending>> left;
    {
        const std::lock_guard lock(state_->mutex);
        state_->stopping = true;
        left.swap(state_->waiting);
    }
    state_->wake.notify_all();
    for (auto& worker : state_->workers) worker.join();
    for (auto& p : left) p->promise.set_exception(std::make_exception_ptr(AuxError(AuxError::Code::stopped, "aux lane: stopped")));
}

void AuxLane::add(Executor executor) {
    if (!executor.run) throw std::invalid_argument("aux lane: executor without a function");
    const auto& units = state_->config.units;
    const bool listed = std::any_of(units.begin(), units.end(),
                                    [&](const Unit& u) { return u.kind == executor.unit && u.id == executor.unit_id; });
    if (!listed)
        throw std::invalid_argument("aux lane: unit " + std::string(unit_name(executor.unit)) + ":" + executor.unit_id +
                                    " is not granted the aux role by the device declaration");
    const std::lock_guard lock(state_->mutex);
    UnitCount count;
    count.unit = executor.unit;
    count.unit_id = executor.unit_id;
    count.kind = executor.kind;
    count.identity = executor.identity;
    state_->totals.units.push_back(std::move(count));
    const bool threaded = executor.threaded;
    const std::string unit_id = executor.unit_id;
    state_->executors.push_back(std::move(executor));
    auto& known = state_->worker_units;
    if (threaded && std::find(known.begin(), known.end(), unit_id) == known.end()) {
        known.push_back(unit_id);
        state_->workers.emplace_back([state = state_.get(), unit_id] { state->work(unit_id); });
    }
}

std::future<AuxResult> AuxLane::submit(AuxJob job) {
    auto p = std::make_unique<Pending>();
    p->order = state_->planned(job);
    p->job = std::move(job);
    p->queued = Clock::now();
    std::unique_lock lock(state_->mutex);
    auto future = state_->enqueue(std::move(p), lock);
    if (!lock.owns_lock()) lock.lock();
    ++state_->totals.submitted;
    return future;
}

std::size_t AuxLane::drain(std::string_view unit_id, std::size_t max_jobs) {
    std::size_t taken = 0;
    while (taken < max_jobs) {
        std::unique_ptr<Pending> p;
        {
            const std::lock_guard lock(state_->mutex);
            p = state_->take(unit_id, false);
        }
        if (!p) break;
        ++taken;
        if (!state_->step(*p)) state_->requeue(std::move(p));
    }
    return taken;
}

AuxResult AuxLane::run_now(const AuxJob& job) {
    const auto order = state_->planned(job);
    const auto queued = Clock::now();
    {
        const std::lock_guard lock(state_->mutex);
        ++state_->totals.submitted;
    }
    std::string errors;
    std::uint32_t fallbacks = 0;
    for (std::size_t at = 0; at < order.size(); ++at) {
        auto p = std::make_unique<Pending>();
        p->job = job;
        p->order = {order[at]};
        p->queued = queued;
        p->fallbacks = fallbacks;
        p->inner = true;
        std::future<AuxResult> future;
        if (state_->executors[order[at]].threaded) {
            std::unique_lock lock(state_->mutex);
            future = state_->enqueue(std::move(p), lock);
        } else {
            future = p->promise.get_future();
            (void)state_->step(*p);
        }
        try {
            auto result = future.get();
            const std::lock_guard lock(state_->mutex);
            ++state_->totals.completed;
            return result;
        } catch (const AuxError& e) {
            if (e.code != AuxError::Code::failed) throw;
            errors += (errors.empty() ? "" : "; ") + std::string(e.what());
        }
        const bool last = at + 1 == order.size();
        const std::lock_guard lock(state_->mutex);
        ++(last ? state_->totals.failed : state_->totals.fallbacks);
        ++fallbacks;
    }
    throw AuxError(AuxError::Code::failed, "aux lane: " + std::string(kind_name(job.kind)) + " job failed on every unit (" + errors + ")");
}

Counters AuxLane::counters() const {
    const std::lock_guard lock(state_->mutex);
    return state_->totals;
}

bool AuxLane::has(JobKind kind) const {
    const std::lock_guard lock(state_->mutex);
    return std::any_of(state_->executors.begin(), state_->executors.end(), [&](const Executor& e) { return e.kind == kind; });
}

std::string AuxLane::receipt_json() const {
    const auto c = counters();
    std::string out = "{\"submitted\":" + std::to_string(c.submitted) + ",\"completed\":" + std::to_string(c.completed) +
                      ",\"queue_full\":" + std::to_string(c.queue_full) + ",\"no_executor\":" + std::to_string(c.no_executor) +
                      ",\"deadline\":" + std::to_string(c.deadline) + ",\"failed\":" + std::to_string(c.failed) +
                      ",\"fallbacks\":" + std::to_string(c.fallbacks) + ",\"waiting_peak\":" + std::to_string(c.waiting_peak) + ",\"units\":[";
    for (std::size_t i = 0; i < c.units.size(); ++i) {
        const auto& u = c.units[i];
        out += std::string(i ? "," : "") + "{\"unit\":" + json_string(unit_name(u.unit)) + ",\"id\":" + json_string(u.unit_id) +
               ",\"kind\":" + json_string(kind_name(u.kind)) + ",\"identity\":" + json_string(u.identity) +
               ",\"jobs\":" + std::to_string(u.jobs) + ",\"failed\":" + std::to_string(u.failed) + ",\"busy_ms\":" + number(u.busy_ms) + "}";
    }
    return out + "]}";
}

// ---- Prompt scoring client ---------------------------------------------------------------------------------

AuxResult pack_scores(const pflash::ScoreBuffers& scores) {
    const std::size_t n = scores.token_score.size();
    if (scores.cut_prob.size() != n || scores.request_mask.size() != n || scores.token_byte_begin.size() != n)
        throw std::invalid_argument("aux lane: score buffers of unequal length");
    AuxResult out;
    out.shape = {3, static_cast<std::uint32_t>(n)};
    out.tensor.reserve(3 * n);
    out.tensor.insert(out.tensor.end(), scores.token_score.begin(), scores.token_score.end());
    out.tensor.insert(out.tensor.end(), scores.cut_prob.begin(), scores.cut_prob.end());
    for (const auto mask : scores.request_mask) out.tensor.push_back(static_cast<float>(mask));
    out.integers = scores.token_byte_begin;
    return out;
}

pflash::ScoreBuffers unpack_scores(const AuxResult& result) {
    if (result.shape.size() != 2 || result.shape[0] != 3) throw std::invalid_argument("aux lane: a score result is [3][tokens]");
    const std::size_t n = result.shape[1];
    if (result.tensor.size() != 3 * n || result.integers.size() != n)
        throw std::invalid_argument("aux lane: score result size does not match its shape");
    pflash::ScoreBuffers out;
    const auto begin = result.tensor.begin();
    const auto offset = [&](std::size_t rows) { return begin + static_cast<std::ptrdiff_t>(rows * n); };
    out.token_score.assign(begin, offset(1));
    out.cut_prob.assign(offset(1), offset(2));
    out.request_mask.reserve(n);
    for (auto it = offset(2); it != offset(3); ++it) {
        const float value = *it;
        if (!(value >= 0.0f && value <= 255.0f) || value != static_cast<float>(static_cast<std::uint8_t>(value)))
            throw std::invalid_argument("aux lane: score result carries a mask value that is not a byte");
        out.request_mask.push_back(static_cast<std::uint8_t>(value));
    }
    out.token_byte_begin = result.integers;
    return out;
}

AuxJob score_job(std::string_view rendered, std::uint64_t session, std::string identity) {
    AuxJob job;
    job.kind = JobKind::score;
    job.inputs.push_back(std::as_bytes(std::span<const char>(rendered.data(), rendered.size())));
    job.session = session;
    job.identity = std::move(identity);
    return job;
}

Executor score_executor(UnitKind unit, std::string unit_id, std::string identity, std::shared_ptr<pflash::PromptScorer> scorer,
                        bool threaded) {
    if (!scorer) throw std::invalid_argument("aux lane: score executor without a scorer");
    Executor out;
    out.unit = unit;
    out.unit_id = std::move(unit_id);
    out.kind = JobKind::score;
    out.identity = std::move(identity);
    out.threaded = threaded;
    out.run = [scorer = std::move(scorer)](const AuxJob& job) {
        if (job.inputs.size() != 1) throw std::invalid_argument("aux lane: a score job has one input (the rendered prompt)");
        const auto bytes = job.inputs.front();
        return pack_scores(scorer->score(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()), job.session));
    };
    return out;
}

pflash::ScoreBuffers LaneScorer::score(std::string_view rendered, std::uint64_t session) {
    auto result = lane_.run_now(score_job(rendered, session, identity_));
    auto out = unpack_scores(result);
    result.tensor.clear();
    result.integers.clear();
    last_ = std::move(result);
    return out;
}

// ---- Vision client -----------------------------------------------------------------------------------------

AuxJob vision_job(const ImageView& image, std::string identity) {
    if (image.height == 0 || image.width == 0 || image.channels == 0 ||
        image.pixels.size() != static_cast<std::size_t>(image.height) * image.width * image.channels)
        throw std::invalid_argument("aux lane: image pixels do not match height x width x channels");
    AuxJob job;
    job.kind = JobKind::vision;
    job.inputs.push_back(std::as_bytes(image.pixels));
    job.shape = {image.height, image.width, image.channels};
    job.identity = std::move(identity);
    return job;
}

Executor vision_executor(UnitKind unit, std::string unit_id, std::string identity, ImageEncoder encoder, bool threaded) {
    if (!encoder) throw std::invalid_argument("aux lane: vision executor without an encoder");
    Executor out;
    out.unit = unit;
    out.unit_id = std::move(unit_id);
    out.kind = JobKind::vision;
    out.identity = std::move(identity);
    out.threaded = threaded;
    out.run = [encoder = std::move(encoder)](const AuxJob& job) {
        if (job.inputs.size() != 1 || job.shape.size() != 3) throw std::invalid_argument("aux lane: a vision job is one image with a 3-entry shape");
        const auto bytes = job.inputs.front();
        ImageView image;
        image.height = job.shape[0];
        image.width = job.shape[1];
        image.channels = job.shape[2];
        image.pixels = std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
        std::uint32_t count = 0;
        std::uint32_t hidden = 0;
        AuxResult result;
        result.tensor = encoder(image, count, hidden);
        if (result.tensor.size() != static_cast<std::size_t>(count) * hidden || result.tensor.empty())
            throw std::runtime_error("aux lane: the image encoder returned rows that do not match count x hidden");
        result.shape = {count, hidden};
        return result;
    };
    return out;
}

ImageRows image_rows(AuxResult result) {
    if (result.shape.size() != 2 || result.tensor.size() != static_cast<std::size_t>(result.shape[0]) * result.shape[1])
        throw std::invalid_argument("aux lane: a vision result is [count][hidden]");
    ImageRows out;
    out.count = result.shape[0];
    out.hidden = result.shape[1];
    out.rows = std::move(result.tensor);
    out.identity = std::move(result.identity);
    out.unit = result.unit;
    out.unit_id = std::move(result.unit_id);
    return out;
}

ImageRows encode_image(AuxLane& lane, const ImageView& image, std::string identity) {
    return image_rows(lane.run_now(vision_job(image, std::move(identity))));
}

} // namespace strix::serve::aux
