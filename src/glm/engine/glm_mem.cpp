// GLM engine: host memory accounting for the rank load (see glm_mem.hpp).
#include "glm_mem.hpp"
#include "strix/deploy/device.hpp"

#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

namespace strix::glm::engine::mem {

namespace {

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// "Key:   123 kB" lines (/proc/meminfo, /proc/self/status) -> bytes.
std::map<std::string, std::uint64_t> read_kb_file(const std::string& path) {
    std::map<std::string, std::uint64_t> m;
    std::ifstream f(path);
    if (!f) return m;
    std::string line;
    while (std::getline(f, line)) {
        const auto c = line.find(':');
        if (c == std::string::npos) continue;
        std::istringstream is(line.substr(c + 1));
        std::uint64_t v = 0;
        std::string unit;
        if (!(is >> v)) continue;
        is >> unit;
        m[line.substr(0, c)] = unit == "kB" ? v * 1024 : v;
    }
    return m;
}

std::uint64_t env_gib(const char* name, std::uint64_t def) {
    const char* e = std::getenv(name);
    if (e == nullptr || *e == 0) return def;
    char* end = nullptr;
    const double v = std::strtod(e, &end);
    if (end == e || !std::isfinite(v) || v < 0) throw std::invalid_argument(std::string(name) + ": not a size in GiB: " + e);
    return static_cast<std::uint64_t>(v * static_cast<double>(kGiB));
}

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

}  // namespace

Probe::Probe(Paths p) : p_(std::move(p)), t0_(now_s()) {
    if (p_.drm_device.empty()) {
        const char* device = std::getenv("STRIX_GLM_DRM_DEVICE");
        p_.drm_device = device && *device ? device : strix::deploy::first_drm_device();
    }
}

Snapshot Probe::read() const {
    Snapshot s;
    s.t = now_s() - t0_;
    const auto mi = read_kb_file(p_.meminfo);
    auto get = [](const std::map<std::string, std::uint64_t>& m, const char* k) {
        const auto it = m.find(k);
        return it == m.end() ? std::uint64_t{0} : it->second;
    };
    s.mem_total = get(mi, "MemTotal");
    s.mem_available = get(mi, "MemAvailable");
    s.mem_free = get(mi, "MemFree");
    s.cached = get(mi, "Cached");
    s.cma_free = get(mi, "CmaFree");
    s.file_lru = get(mi, "Active(file)") + get(mi, "Inactive(file)");
    s.gpu_active = get(mi, "GPUActive");
    s.meminfo_ok = mi.count("MemAvailable") != 0;
    s.cma_free_ok = mi.count("CmaFree") != 0;
    {  // /proc/zoneinfo: "        cma      3267584" per zone (pages); readable = layout known (no 'cma' lines = no CMA)
        std::ifstream z(p_.zoneinfo);
        s.zoneinfo_ok = static_cast<bool>(z);
        std::string tok;
        std::uint64_t v = 0;
        while (z >> tok)
            if (tok == "cma" && (z >> v)) s.cma_total += v * 4096;
    }
    s.shmem = get(mi, "Shmem");
    s.swap_total = get(mi, "SwapTotal");
    s.swap_free = get(mi, "SwapFree");
    {  // "some avg10=0.00 avg60=... total=..." / "full avg10=0.00 ..."
        std::ifstream z(p_.psi);
        std::string kind, kv;
        while (z >> kind >> kv) {
            if (kind == "full" && kv.rfind("avg10=", 0) == 0) s.psi_full10 = std::strtod(kv.c_str() + 6, nullptr);
            std::string rest;
            std::getline(z, rest);
        }
    }
    const auto st = read_kb_file(p_.status);
    s.rss = get(st, "VmRSS");
    s.rss_anon = get(st, "RssAnon");
    s.rss_file = get(st, "RssFile");
    s.rss_shmem = get(st, "RssShmem");
    s.hwm = get(st, "VmHWM");
    if (!p_.drm_device.empty()) {
        const auto d = strix::deploy::probe_device("", p_.drm_device);
        s.gtt_total = d.gtt_total_bytes.value_or(0);
        s.gtt_used = d.gtt_used_bytes.value_or(0);
        s.vram_total = d.vram_total_bytes.value_or(0);
        s.vram_used = d.vram_used_bytes.value_or(0);
        s.gpu = d.gtt_total_bytes && d.gtt_used_bytes && d.vram_used_bytes;
    }
    return s;
}

Margins Margins::from_env() {
    Margins m;
    m.ram = env_gib("STRIX_GLM_MEM_MARGIN_GIB", m.ram);
    m.gtt = env_gib("STRIX_GLM_GTT_MARGIN_GIB", m.gtt);
    return m;
}

std::string describe(const Snapshot& s) {
    return fmt("MemAvailable %.2f GiB (CmaFree %.2f of CMA %.2f, RAM headroom %.2f), swap used %.2f/%.2f GiB, page cache %.2f GiB, "
               "rss %.2f GiB, GPUActive %.2f GiB, gtt %s",
               gib(s.mem_available), gib(s.cma_free), gib(s.cma_total), gib(s.ram_headroom()), gib(s.swap_used()), gib(s.swap_total),
               gib(s.cached), gib(s.rss), gib(s.gpu_active), s.gpu ? fmt("%.2f/%.2f GiB used (headroom %.2f)", gib(s.gtt_used), gib(s.gtt_total), gib(s.gtt_headroom())).c_str()
                                 : "n/a (no amdgpu mem_info)");
}

void admit(const std::string& what, const std::vector<Term>& terms, const Snapshot& s, const Margins& m) {
    if (!s.meminfo_ok) throw BudgetError("memory budget: REFUSED " + what + ": /proc/meminfo (MemAvailable) unreadable");
    if (!s.zoneinfo_ok) std::fprintf(stderr, "mem: WARNING /proc/zoneinfo unreadable: every file page counted as CMA-resident\n");
    if (!s.cma_free_ok && s.cma_total > 0)
        std::fprintf(stderr, "mem: WARNING CmaFree missing with %.2f GiB of CMA: the whole CMA area is subtracted\n", gib(s.cma_total));
    std::uint64_t g = 0, h = 0;
    std::string list;
    for (const auto& t : terms) {
        g += t.gtt;
        h += t.host;
        list += fmt("\n    %-34s gtt %9.3f GiB  host %8.3f GiB", t.name.c_str(), gib(t.gtt), gib(t.host));
    }
    const bool gtt_ok = !s.gpu || g + m.gtt <= s.gtt_headroom();
    const bool ram_ok = g + h + m.ram <= s.ram_headroom();
    if (gtt_ok && ram_ok) return;
    std::string why;
    if (!gtt_ok) why += fmt(" GTT: need %.2f + margin %.2f GiB > headroom %.2f GiB.", gib(g), gib(m.gtt), gib(s.gtt_headroom()));
    if (!ram_ok)
        why += fmt(" RAM: need %.2f (gtt) + %.2f (host) + margin %.2f GiB > headroom %.2f GiB (MemAvailable %.2f - CmaFree %.2f"
                   " - file pages in CMA).", gib(g), gib(h), gib(m.ram), gib(s.ram_headroom()), gib(s.mem_available), gib(s.cma_free));
    throw BudgetError("memory budget: REFUSED " + what + " (fail closed, nothing more is allocated):" + why + "\n  terms:" + list +
                      "\n  node: " + describe(s));
}

std::uint64_t AllocModel::physical(std::uint64_t bytes) const {
    if (ratio > 0) return round_up(static_cast<std::uint64_t>(std::ceil(static_cast<double>(bytes) * ratio)), 4096);
    return round_up(bytes, granule);
}

std::string AllocModel::str() const {
    if (ratio > 0) return fmt("ratio %.4f%s", ratio, calibrated ? "" : " (assumed)");
    return fmt("granule %llu B%s", static_cast<unsigned long long>(granule), calibrated ? "" : " (assumed)");
}

AllocModel infer_alloc_model(std::uint64_t size, double measured) {
    AllocModel m;
    if (size == 0 || !std::isfinite(measured) || measured < 0.9 * static_cast<double>(size)) {
        m.granule = 2 * kMiB;  // not trusted: assume the worst plausible granule
        return m;
    }
    m.calibrated = true;
    for (const std::uint64_t g : {std::uint64_t{4096}, 64 * kKiB, 2 * kMiB})
        if (measured <= static_cast<double>(round_up(size, g)) * 1.005) {
            m.granule = g;
            return m;
        }
    m.ratio = measured / static_cast<double>(size);
    return m;
}

AllocModel model_from_batches(std::uint64_t size, const std::vector<double>& per_batch, bool& consistent) {
    consistent = false;
    if (per_batch.empty()) return infer_alloc_model(size, 0.0);
    const auto [lo, hi] = std::minmax_element(per_batch.begin(), per_batch.end());
    consistent = *hi - *lo <= 0.10 * *hi + 65536.0;
    AllocModel m = infer_alloc_model(size, *hi);
    if (!consistent) {
        const AllocModel g2 = infer_alloc_model(size, static_cast<double>(round_up(size, 2 * kMiB)));
        if (g2.physical(size) > m.physical(size)) m = g2;
        m.calibrated = false;
    }
    return m;
}

std::uint64_t prepared_bytes(std::uint64_t tr, std::uint64_t suh, std::uint64_t svh, std::uint64_t bias, std::uint64_t align) {
    std::uint64_t total = 0;
    for (const std::uint64_t b : {tr, suh, svh, bias}) {
        total = round_up(total, align);
        total += b;
    }
    return total;
}

std::uint64_t LoadPlan::dominant_size() const {
    // The calibration measures the per-allocation cost of the most frequent tensor-sized allocation. The engine's 1 GiB
    // raw arena chunks are excluded (with raw-only attention they outnumber every tensor size, and a
    // calibration of 2 x 64 x 1 GiB was refused on the real model (measurement)); their granule cost is nil anyway.
    constexpr std::uint64_t kArenaFloor = std::uint64_t{256} << 20;
    std::map<std::uint64_t, std::size_t> n;
    for (const auto b : device)
        if (b < kArenaFloor) ++n[b];
    std::uint64_t best = 0;
    std::size_t c = 0;
    for (const auto& [b, k] : n)
        if (k > c) {
            best = b;
            c = k;
        }
    return best;
}

std::uint64_t LoadPlan::physical(const AllocModel& m, std::size_t upto) const {
    std::uint64_t t = 0;
    const std::size_t n = std::min(upto, device.size());
    for (std::size_t i = 0; i < n; ++i) t += m.physical(device[i]);
    return t;
}

std::string LoadPlan::summary(const AllocModel& m) const {
    const std::uint64_t dom = dominant_size();
    const auto ndom = static_cast<std::size_t>(std::count(device.begin(), device.end(), dom));
    return fmt("plan: %zu device allocations, %.3f GiB logical, %.3f GiB physical (%s); dominant size %llu B x %zu; "
               "read %.1f GB; host: stage %.3f GiB, whole %.3f GiB, bounce %.3f GiB",
               device.size(), gib(logical), gib(physical(m)), m.str().c_str(), static_cast<unsigned long long>(dom), ndom,
               static_cast<double>(bytes_read) / 1e9, gib(stage_max), gib(whole_max), gib(bounce));
}

Watchdog Watchdog::from_env(std::uint64_t host_budget) {
    Watchdog w;
    // The default RSS limit covers the engine alone; budgeted caller memory must not read as a leak.
    w.max_rss = host_budget > std::numeric_limits<std::uint64_t>::max() - w.max_rss
                    ? std::numeric_limits<std::uint64_t>::max() : w.max_rss + host_budget;
    if (const char* e = std::getenv("STRIX_GLM_WATCHDOG"); e != nullptr && std::string(e) == "0") w.enabled = false;
    w.min_ram_headroom = env_gib("STRIX_GLM_WD_MIN_FREE_GIB", w.min_ram_headroom);
    w.max_swap_growth = env_gib("STRIX_GLM_WD_SWAP_GIB", w.max_swap_growth);
    w.max_rss = env_gib("STRIX_GLM_WD_RSS_GIB", w.max_rss);
    w.max_gtt_used = env_gib("STRIX_GLM_WD_GTT_GIB", w.max_gtt_used);
    if (const char* e = std::getenv("STRIX_GLM_WD_PSI_FULL"); e != nullptr && *e) w.max_psi_full10 = std::strtod(e, nullptr);
    return w;
}

bool set_oom_score_adj(int v) {
    const int fd = ::open("/proc/self/oom_score_adj", O_WRONLY | O_CLOEXEC);
    if (fd < 0) return false;
    const std::string s = std::to_string(v);
    const bool ok = ::write(fd, s.data(), s.size()) == static_cast<ssize_t>(s.size());
    ::close(fd);
    return ok;
}

struct Monitor::Sink {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::pair<int, std::string>> q;
    int fd = -1;
    bool stop = false, done = false;
    std::uint64_t queued = 0, written = 0;
    std::atomic<std::uint64_t> errors{0}, dropped{0};
};

namespace {

void writer_loop(const std::shared_ptr<Monitor::Sink>& k) {
    sigset_t pipe;  // a closed trace pipe must not SIGPIPE the rank: EPIPE is counted as a write error instead
    sigemptyset(&pipe);
    sigaddset(&pipe, SIGPIPE);
    (void)pthread_sigmask(SIG_BLOCK, &pipe, nullptr);
    for (;;) {
        std::pair<int, std::string> it;
        {
            std::unique_lock<std::mutex> g(k->mu);
            k->cv.wait(g, [&] { return k->stop || !k->q.empty(); });
            if (k->q.empty()) break;
            it = std::move(k->q.front());
            k->q.pop_front();
        }
        const char* p = it.second.data();
        std::size_t n = it.second.size();
        bool ok = true;
        while (n > 0) {
            const ssize_t r = ::write(it.first, p, n);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) {
                ok = false;
                break;
            }
            p += r;
            n -= static_cast<std::size_t>(r);
        }
        if (!ok && k->errors.fetch_add(1) == 0 && it.first != 2) {
            const std::string m = "mem: WARNING memory trace write failed (" + std::string(std::strerror(errno)) + "); further failures counted\n";
            if (::write(2, m.data(), m.size()) < 0) { /* stderr gone too */ }
        }
        std::lock_guard<std::mutex> g(k->mu);
        ++k->written;
        k->cv.notify_all();
    }
    std::lock_guard<std::mutex> g(k->mu);
    if (k->fd >= 0) ::close(k->fd);
    k->fd = -1;
    k->done = true;
    k->cv.notify_all();
}

}  // namespace

Monitor::Monitor(Probe probe, std::string trace_path, Watchdog wd, AbortFn on_abort, unsigned sample_ms, unsigned trace_ms,
                 unsigned abort_grace_ms)
    : probe_(std::move(probe)), trace_path_(std::move(trace_path)), wd_(wd), on_abort_(std::move(on_abort)),
      sample_ms_(std::max(sample_ms, 10U)), trace_ms_(std::max(trace_ms, sample_ms_)), grace_ms_(abort_grace_ms),
      sink_(std::make_shared<Sink>()) {
    start_ = probe_.read();
    if (wd_.enabled && !start_.meminfo_ok) throw std::runtime_error("memory watchdog: /proc/meminfo (MemAvailable) unreadable");
    max_gtt_.store(wd_.max_gtt_used);
    if (!trace_path_.empty()) {
        sink_->fd = trace_path_ == "-" ? ::dup(2) : ::open(trace_path_.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        if (sink_->fd < 0) throw std::runtime_error("memory trace: cannot open " + trace_path_ + ": " + std::strerror(errno));
    }
    if (!on_abort_)
        on_abort_ = [](const std::string&) { std::_Exit(86); };
    try {
        writer_ = std::thread([k = sink_] { writer_loop(k); });
        if (sink_->fd >= 0) push(sink_->fd, header());
        th_ = std::thread([this] { run(); });
    } catch (...) {  // partial start: never destroy a joinable thread, never leak the trace fd
        if (writer_.joinable()) {
            stop_writer();
        } else if (sink_->fd >= 0) {
            ::close(sink_->fd);
            sink_->fd = -1;
        }
        throw;
    }
}

void Monitor::stop_writer() {
    bool done = false;
    {
        std::unique_lock<std::mutex> g(sink_->mu);
        sink_->stop = true;
        sink_->cv.notify_all();
        done = sink_->cv.wait_for(g, std::chrono::seconds(2), [&] { return sink_->done; });
    }
    if (done) writer_.join();
    else writer_.detach();  // stuck in write(2): the thread keeps the shared Sink alive and exits when the write returns
}

Monitor::~Monitor() {
    stop_.store(true);
    if (th_.joinable()) th_.join();
    if (sink_->fd >= 0 && push(sink_->fd, line(probe_.read()))) lines_.fetch_add(1);
    stop_writer();
}

std::uint64_t Monitor::write_errors() const { return sink_->errors.load(); }
std::uint64_t Monitor::dropped() const { return sink_->dropped.load(); }

bool Monitor::push(int fd, std::string s) {
    std::lock_guard<std::mutex> g(sink_->mu);
    if (sink_->q.size() >= 256) {
        sink_->dropped.fetch_add(1);
        return false;
    }
    sink_->q.emplace_back(fd, std::move(s));
    ++sink_->queued;
    sink_->cv.notify_all();
    return true;
}

void Monitor::phase(const std::string& p) {
    std::lock_guard<std::mutex> g(mu_);
    phase_ = p;
}

std::string Monitor::header() {
    return "# t_s\tphase\trss_mib\tanon_mib\tshmem_mib\tcache_mib\tavail_mib\tcma_free_mib\tram_headroom_mib\tswap_used_mib\tgtt_used_mib"
           "\tgpu_active_mib\tpsi_full10\tdev_logical_mib\tread_mib\n";
}

std::string Monitor::line(const Snapshot& s) const {
    std::string ph;
    {
        std::lock_guard<std::mutex> g(mu_);
        ph = phase_;
    }
    const auto mib = [](std::uint64_t b) { return static_cast<double>(b) / static_cast<double>(kMiB); };
    return fmt("%.2f\t%s\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%s\t%.0f\t%.2f\t%.0f\t%.0f\n", s.t, ph.c_str(), mib(s.rss),
               mib(s.rss_anon), mib(s.rss_shmem), mib(s.cached), mib(s.mem_available), mib(s.cma_free), mib(s.ram_headroom()),
               mib(s.swap_used()), s.gpu ? fmt("%.0f", mib(s.gtt_used)).c_str() : "na", mib(s.gpu_active), s.psi_full10, mib(dev_.load()),
               mib(read_.load()));
}

std::string Monitor::check(const Snapshot& s) const {
    if (!wd_.enabled) return {};
    if (!s.meminfo_ok) return "telemetry lost: /proc/meminfo unreadable";
    if (start_.gpu && !s.gpu) return "telemetry lost: amdgpu mem_info counters (GTT / VRAM used) unreadable";
    if (start_.cma_free_ok && !s.cma_free_ok) return "telemetry lost: CmaFree missing from /proc/meminfo";
    if (wd_.min_ram_headroom && s.ram_headroom() < wd_.min_ram_headroom)
        return fmt("RAM headroom %.2f GiB < %.2f GiB (MemAvailable %.2f, CmaFree %.2f, GPUActive %.2f)", gib(s.ram_headroom()),
                   gib(wd_.min_ram_headroom), gib(s.mem_available), gib(s.cma_free), gib(s.gpu_active));
    const std::uint64_t su = s.swap_used(), s0 = start_.swap_used();
    if (wd_.max_swap_growth && su > s0 && su - s0 > wd_.max_swap_growth)
        return fmt("swap use grew %.2f GiB (> %.2f GiB) since start", gib(su - s0), gib(wd_.max_swap_growth));
    if (wd_.max_rss && s.rss > wd_.max_rss) return fmt("rank RSS %.2f GiB > %.2f GiB", gib(s.rss), gib(wd_.max_rss));
    if (wd_.max_psi_full10 > 0 && s.psi_full10 >= wd_.max_psi_full10)
        return fmt("memory PSI full avg10 %.1f %% >= %.1f %% (thrash)", s.psi_full10, wd_.max_psi_full10);
    const std::uint64_t mg = max_gtt_.load();
    if (mg && s.dev_counters() && s.dev_used() > mg)
        return fmt("device memory used (GTT + VRAM) %.2f GiB > limit %.2f GiB", gib(s.dev_used()), gib(mg));
    return {};
}

void Monitor::run() {
    auto next_trace = std::chrono::steady_clock::now();
    while (!stop_.load()) {
        const Snapshot s = probe_.read();
        const auto now = std::chrono::steady_clock::now();
        const std::string why = check(s);
        if (sink_->fd >= 0 && (now >= next_trace || !why.empty())) {
            if (push(sink_->fd, line(s))) lines_.fetch_add(1);
            next_trace = now + std::chrono::milliseconds(trace_ms_);
        }
        if (!why.empty()) {
            tripped_.store(true);
            const std::string msg = "WATCHDOG ABORT (glm memory): " + why + "; node: " + describe(s) + "\n";
            if (sink_->fd >= 0) push(sink_->fd, "# " + msg);
            push(2, msg);
            {  // bounded wait for the writer: logging can never hold the abort back
                std::unique_lock<std::mutex> g(sink_->mu);
                const std::uint64_t target = sink_->queued;
                sink_->cv.wait_for(g, std::chrono::milliseconds(grace_ms_), [&] { return sink_->written >= target; });
            }
            on_abort_(why);
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms_));
    }
}

}  // namespace strix::glm::engine::mem
