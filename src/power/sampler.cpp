#include "strix/power/sampler.hpp"

#include "strix/env_flags.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#define STRIX_POWER_POSIX 1
#else
#define STRIX_POWER_POSIX 0
#endif

namespace strix::power {
namespace {

namespace fs = std::filesystem;

bool finite(double v) noexcept { return std::isfinite(v); }

std::string errno_reason(int error, const std::string& path) {
    if (error == ENOENT || error == ENOTDIR) return "not found: " + path;
    if (error == EACCES || error == EPERM) return "permission denied: " + path;
    return std::string(std::strerror(error)) + ": " + path;
}

// A small text file holding one number, read again at every sample. POSIX keeps the descriptor and re-reads at offset 0.
class ValueFile {
public:
    ValueFile() = default;
    ValueFile(const ValueFile&) = delete;
    ValueFile& operator=(const ValueFile&) = delete;
    ValueFile(ValueFile&& other) noexcept : path_(std::move(other.path_)), fd_(other.fd_), signed_(other.signed_) { other.fd_ = -1; }
    ~ValueFile() { close_fd(); }

    // False with the reason when the file cannot be opened or does not hold a number.
    // signed_value: the channel can be negative (a temperature); every other channel refuses a negative number.
    bool open(const std::string& path, std::string& why, bool signed_value = false) {
        close_fd();
        path_ = path;
        signed_ = signed_value;
#if STRIX_POWER_POSIX
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) { why = errno_reason(errno, path); path_.clear(); return false; }
#else
        std::FILE* file = std::fopen(path.c_str(), "rb");
        if (!file) { why = errno_reason(errno, path); path_.clear(); return false; }
        std::fclose(file);
#endif
        double value = 0.0;
        if (!read(value)) { why = (errno ? errno_reason(errno, path) : "not a number: " + path); close_fd(); path_.clear(); return false; }
        return true;
    }
    [[nodiscard]] bool is_open() const noexcept { return !path_.empty() && (STRIX_POWER_POSIX ? fd_ >= 0 : true); }
    // A complete number only (parse_value).
    bool read(double& value) const noexcept {
        char text[64];
        std::size_t got = 0;
        errno = 0;
#if STRIX_POWER_POSIX
        if (fd_ < 0) return false;
        const auto n = ::pread(fd_, text, sizeof(text) - 1, 0);
        if (n <= 0) return false;
        got = static_cast<std::size_t>(n);
#else
        std::FILE* file = std::fopen(path_.c_str(), "rb");
        if (!file) return false;
        got = std::fread(text, 1, sizeof(text) - 1, file);
        std::fclose(file);
        if (!got) return false;
#endif
        errno = 0;
        double parsed = 0.0;
        // A text that fills the buffer may be cut short: not a number.
        if (got >= sizeof(text) - 1 || !parse_value(std::string_view(text, got), parsed)) return false;
        if (parsed < 0.0 && !signed_) return false;
        value = parsed;
        return true;
    }

private:
    void close_fd() noexcept {
#if STRIX_POWER_POSIX
        if (fd_ >= 0) ::close(fd_);
#endif
        fd_ = -1;
    }
    std::string path_;
    int fd_ = -1;
    bool signed_ = false;
};

std::string read_word(const fs::path& path) {
    std::FILE* file = std::fopen(path.string().c_str(), "rb");
    if (!file) return {};
    char text[96];
    const auto got = std::fread(text, 1, sizeof(text) - 1, file);
    std::fclose(file);
    std::string word(text, got);
    while (!word.empty() && (word.back() == '\n' || word.back() == '\r' || word.back() == ' ')) word.pop_back();
    return word;
}

std::vector<fs::path> children(const fs::path& directory) {
    std::vector<fs::path> out;
    std::error_code error;
    for (fs::directory_iterator it(directory, error), end; !error && it != end; it.increment(error)) out.push_back(it->path());
    std::sort(out.begin(), out.end());
    return out;
}

bool present(const fs::path& path) { std::error_code error; return fs::exists(path, error); }
void set_max(double& slot, double value) noexcept { slot = finite(slot) ? std::max(slot, value) : value; }
void set_if_absent(double& slot, double value) noexcept { if (!finite(slot)) slot = value; }

// amdgpu hwmon: power (the SoC package power tracking value on an integrated device, the card on a discrete one), the
// shader clock, the edge temperature and the busy percentage. World readable on a stock kernel.
class AmdgpuHwmon final : public Source {
public:
    explicit AmdgpuHwmon(std::string root) : root_(std::move(root)) {}
    std::string_view coverage_id(Domain domain) const noexcept override {
        return domain == Domain::package ? "host-packages" : "";
    }
    SourceStatus probe() override {
        SourceStatus status{"amdgpu-hwmon", false, {}};
        const fs::path base = fs::path(root_) / "sys" / "class" / "hwmon";
        std::string first_problem;
        for (const auto& dir : children(base)) {
            if (read_word(dir / "name") != "amdgpu") continue;
            Card card;
            std::string why, what = dir.filename().string() + ":";
            if (card.power.open((dir / "power1_average").string(), why)) what += " power1_average";
            else if (card.power.open((dir / "power1_input").string(), why)) what += " power1_input";
            else if (first_problem.empty()) first_problem = why;
            if (const auto label = read_word(dir / "power1_label"); card.power.is_open() && !label.empty()) what += "(" + label + ")";
            if (card.clock.open((dir / "freq1_input").string(), why)) what += " freq1_input";
            if (card.temp.open((dir / "temp1_input").string(), why, true)) what += " temp1_input";
            if (card.busy.open((dir / "device" / "gpu_busy_percent").string(), why)) what += " gpu_busy_percent";
            if (!card.power.is_open() && !card.clock.is_open() && !card.temp.is_open()) continue;
            // The driver shows in1 (vddnb) on an integrated device only and a fan on a card only. An integrated
            // device's power is the SoC package (the same domain as a package counter); a card is its own domain.
            // Neither marker: taken as package, so a package counter is never added to it twice.
            const bool integrated = present(dir / "in1_input"), card_fan = present(dir / "fan1_input") || present(dir / "pwm1");
            card.domain = !integrated && card_fan ? Domain::accelerator : Domain::package;
            if (card.power.is_open()) what += std::string(" domain=") + domain_name(card.domain) + (integrated || card_fan ? "" : "(assumed)");
            status.detail += (status.detail.empty() ? "" : "; ") + what;
            cards_.push_back(std::move(card));
        }
        status.available = !cards_.empty();
        if (!status.available)
            status.detail = first_problem.empty() ? "no amdgpu hwmon device under " + base.generic_string() : first_problem;
        return status;
    }
    void read(Sample& sample) noexcept override {
        for (const auto& card : cards_) {
            double v = 0.0;
            if (card.power.is_open()) { if (card.power.read(v)) sample.add_power(card.domain, v * 1e-6); else sample.fail_power(card.domain); }
            if (card.clock.read(v)) set_if_absent(sample.gpu_clock_mhz, v * 1e-6);
            if (card.temp.read(v)) set_max(sample.temp_c, v * 1e-3);
            if (card.busy.read(v) && v <= 100.0) set_if_absent(sample.gpu_busy_pct, v);
        }
    }

private:
    struct Card { ValueFile power, clock, temp, busy; Domain domain = Domain::package; };
    std::string root_;
    std::vector<Card> cards_;
};

// powercap RAPL package energy counters. Usually readable by root only; then the source is unavailable and says so.
class Rapl final : public Source {
public:
    explicit Rapl(std::string root) : root_(std::move(root)) {}
    std::string_view coverage_id(Domain domain) const noexcept override {
        return domain == Domain::package ? "host-packages" : "";
    }
    SourceStatus probe() override {
        SourceStatus status{"rapl", false, {}};
        const fs::path base = fs::path(root_) / "sys" / "class" / "powercap";
        for (const auto& dir : children(base)) {
            const auto zone = dir.filename().string();
            // Top-level zones only ("intel-rapl:0"); "intel-rapl:0:0" is a sub-zone already inside its package.
            if (zone.rfind("intel-rapl:", 0) != 0 || std::count(zone.begin(), zone.end(), ':') != 1) continue;
            if (read_word(dir / "name").rfind("package", 0) != 0) continue;
            Zone z;
            std::string why;
            z.path = (dir / "energy_uj").string();
            const bool opened = z.energy.open(z.path, why);
            const std::string failure = why;
            ValueFile range;
            if (range.open((dir / "max_energy_range_uj").string(), why)) range.read(z.range_uj);
            z.baseline = opened && z.energy.read(z.last_uj) && (z.range_uj <= 0.0 || z.last_uj <= z.range_uj);
            status.detail += (status.detail.empty() ? "" : "; ") + zone + ": energy_uj";
            if (!z.baseline) status.detail += " (probe failed: " + (failure.empty() ? "invalid baseline" : failure) + ")";
            zones_.push_back(std::move(z));
        }
        status.available = !zones_.empty();
        if (!status.available) status.detail = "no package zone under " + base.generic_string();
        return status;
    }
    void read(Sample& sample) noexcept override {
        // Each zone keeps its own baseline, so the cumulative total is right again as soon as every zone has been
        // read: a zone that failed for a while adds the energy of that while at its next read. A sample in which any
        // zone failed carries no counter at all (never the sum of the zones that happened to answer), so a window
        // that ends inside the failure is not covered by the counter instead of being short.
        bool all = true;
        for (auto& z : zones_) {
            double now = 0.0;
            if (!z.energy.is_open()) {
                std::string why;
                if (!z.energy.open(z.path, why)) { all = false; continue; }
            }
            if (!z.energy.read(now) || (z.range_uj > 0.0 && now > z.range_uj)) { all = false; continue; }
            if (!z.baseline) { z.last_uj = now; z.baseline = true; continue; }
            // The counter wraps at max_energy_range_uj. A drop with no known range is a reset: nothing is added.
            const double delta = now >= z.last_uj ? now - z.last_uj : (z.range_uj > 0.0 ? now + z.range_uj - z.last_uj : 0.0);
            z.last_uj = now;
            total_j_ += delta * 1e-6;
        }
        if (all) sample.add_energy(Domain::package, total_j_); else sample.fail_energy(Domain::package);
    }

private:
    struct Zone { ValueFile energy; std::string path; double range_uj = 0.0, last_uj = 0.0; bool baseline = false; };
    std::string root_;
    std::vector<Zone> zones_;
    double total_j_ = 0.0;
};

// Mean CPU frequency over the cpufreq policies, refreshed at most once per second (a policy read is not free).
class Cpufreq final : public Source {
public:
    explicit Cpufreq(std::string root) : root_(std::move(root)) {}
    SourceStatus probe() override {
        SourceStatus status{"cpufreq", false, {}};
        const fs::path base = fs::path(root_) / "sys" / "devices" / "system" / "cpu" / "cpufreq";
        std::string problem;
        for (const auto& dir : children(base)) {
            if (dir.filename().string().rfind("policy", 0) != 0) continue;
            ValueFile file;
            std::string why;
            if (file.open((dir / "scaling_cur_freq").string(), why)) files_.push_back(std::move(file));
            else if (problem.empty()) problem = why;
        }
        status.available = !files_.empty();
        status.detail = status.available ? std::to_string(files_.size()) + " policies: scaling_cur_freq"
            : problem.empty() ? "no cpufreq policy under " + base.generic_string() : problem;
        return status;
    }
    void read(Sample& sample) noexcept override {
        if (!finite(cached_) || sample.t_ns - last_ns_ >= 1'000'000'000LL || sample.t_ns < last_ns_) {
            double sum = 0.0;
            std::size_t n = 0;
            for (const auto& file : files_) { double v = 0.0; if (file.read(v)) { sum += v; ++n; } }
            cached_ = n ? sum / static_cast<double>(n) * 1e-3 : kAbsent;
            last_ns_ = sample.t_ns;
        }
        if (finite(cached_)) set_if_absent(sample.cpu_mhz, cached_);
    }

private:
    std::string root_;
    std::vector<ValueFile> files_;
    double cached_ = kAbsent;
    std::int64_t last_ns_ = 0;
};

struct Registry {
    std::mutex mutex;
    std::vector<std::pair<std::string, SourceFactory>> kinds;
    Registry() {
        kinds.emplace_back("rapl", [](const std::string& root) { return std::make_unique<Rapl>(root); });
        kinds.emplace_back("amdgpu-hwmon", [](const std::string& root) { return std::make_unique<AmdgpuHwmon>(root); });
        kinds.emplace_back("cpufreq", [](const std::string& root) { return std::make_unique<Cpufreq>(root); });
    }
};
Registry& registry() { static Registry r; return r; }

using Cover = Sampler::Cover;
using Covers = Sampler::Covers;
using Joint = Sampler::Joint;
constexpr std::int64_t kNoLo = std::numeric_limits<std::int64_t>::min(), kNoHi = std::numeric_limits<std::int64_t>::max();

// Adds the stretch between two consecutive samples' readings (ta, va) and (tb, vb) of one channel, clipped to [t0, t1].
// False (nothing added) when it is longer than the gap limit or the counter went down (a reset).
bool stretch(Cover& c, bool counter, std::int64_t ta, double va, std::int64_t tb, double vb, std::int64_t t0, std::int64_t t1,
             std::int64_t gap_ns) noexcept {
    if (tb <= ta || tb - ta > gap_ns || (counter && vb < va)) return false;
    const auto lo = std::max(ta, t0), hi = std::min(tb, t1);
    if (hi <= lo) return false;
    const double span = static_cast<double>(tb - ta);
    if (counter) {
        c.joules += (vb - va) * (static_cast<double>(hi - lo) / span);
        c.peak = std::max(c.peak, (vb - va) / (span * 1e-9));
    } else {
        const double pa = va + (vb - va) * (static_cast<double>(lo - ta) / span);
        const double pb = va + (vb - va) * (static_cast<double>(hi - ta) / span);
        c.joules += 0.5 * (pa + pb) * static_cast<double>(hi - lo) * 1e-9;
        c.peak = std::max({c.peak, pa, pb});
    }
    c.ns += hi - lo;
    return true;
}
// Keep physical coverage groups separate through integration and channel selection. A failed endpoint or an
// excessive gap excludes the stretch; joint records only observed sets of simultaneously covered channels.
void span(Covers& covers, Joint& joint, const Sample& a, const Sample& b, std::int64_t t0, std::int64_t t1, std::int64_t gap_ns) {
    const auto lo = std::max(a.t_ns, t0), hi = std::min(b.t_ns, t1);
    if (hi <= lo) return;
    std::vector<unsigned char> mask(covers.size());
    for (std::size_t g = 0; g < covers.size(); ++g) {
        const auto& x = a.hardware[g];
        const auto& y = b.hardware[g];
        if (finite(x.energy_j) && finite(y.energy_j) && stretch(covers[g][0], true, a.t_ns, x.energy_j, b.t_ns, y.energy_j, t0, t1, gap_ns))
            mask[g] |= 1;
        if (finite(x.power_w) && finite(y.power_w) && stretch(covers[g][1], false, a.t_ns, x.power_w, b.t_ns, y.power_w, t0, t1, gap_ns))
            mask[g] |= 2;
    }
    joint[mask] += hi - lo;
}

bool combine(const Covers& covers, const Joint& joint, const std::vector<unsigned char>& expected,
             const std::vector<Domain>& domains, std::int64_t window_ns, Receipt& r) {
    bool whole = false;
    for (std::size_t g = 0; g < covers.size(); ++g) if (expected[g] && domains[g] == Domain::system) whole = true;
    bool any = false, all = true, by_counter = false, by_power = false;
    std::vector<unsigned char> need(covers.size());
    for (std::size_t g = 0; g < covers.size(); ++g) {
        if (!expected[g] || (whole && domains[g] != Domain::system)) continue;
        const Cover& c = covers[g][0];
        const Cover& p = covers[g][1];
        const bool counter = c.ns > 0 && c.ns >= p.ns;
        const Cover& u = counter ? c : p;
        r.coverage += (r.coverage.empty() ? "" : "+") + std::string(domain_name(domains[g])) + ":" +
            (u.ns <= 0 ? "none" : counter ? "counter" : "power-integral");
        if (u.ns <= 0) { all = false; continue; }
        need[g] = counter ? 1 : 2;
        any = true;
        (counter ? by_counter : by_power) = true;
        r.energy_j += u.joules;
        r.avg_w += u.joules / (static_cast<double>(u.ns) * 1e-9);
        r.peak_w += p.ns > 0 ? p.peak : c.peak;  // sum of hardware peaks, an upper bound
    }
    if (!any) return false;
    std::int64_t covered = 0;
    if (all) for (const auto& [mask, ns] : joint) {
        bool common = true;
        for (std::size_t g = 0; g < need.size(); ++g) if ((mask[g] & need[g]) != need[g]) common = false;
        if (common) covered += ns;
    }
    r.available = true;
    r.complete = covered == window_ns;
    r.covered_ms = static_cast<double>(covered) * 1e-6;
    r.energy_source = by_counter && by_power ? "mixed" : by_counter ? "counter" : "power-integral";
    r.peak_w = std::max(r.peak_w, r.avg_w);
    return true;
}

double median(std::vector<double>& values) noexcept {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

void json_text(std::string& out, const std::string& text) {
    out += '"';
    for (const char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (static_cast<unsigned char>(c) < 0x20) { char u[8]; std::snprintf(u, sizeof(u), "\\u%04x", static_cast<unsigned>(c)); out += u; }
        else out += c;
    }
    out += '"';
}
void json_number(std::string& out, double value) {
    char text[40];
    std::snprintf(text, sizeof(text), "%.6g", finite(value) ? value : 0.0);
    out += text;
}
void json_member(std::string& out, const char* key, double value) { out += ",\""; out += key; out += "\":"; json_number(out, value); }
// A reading that may be absent: null, never a 0 that reads as a measured idle value.
void json_reading(std::string& out, const char* key, double value) {
    out += ",\""; out += key; out += "\":";
    if (finite(value)) json_number(out, value); else out += "null";
}
void receipt_json(std::string& out, const Receipt& r) {
    if (!r.available) { out += "{\"available\":false,\"reason\":"; json_text(out, r.reason); out += '}'; return; }
    out += "{\"available\":true";
    json_member(out, "duration_ms", r.duration_ms);
    json_member(out, "energy_j", r.energy_j);
    json_member(out, "avg_w", r.avg_w);
    json_member(out, "peak_w", r.peak_w);
    out += ",\"tokens\":" + std::to_string(r.tokens);
    json_member(out, "tok_per_j", r.tok_per_j);
    json_member(out, "gpu_clock_mhz_median", r.gpu_clock_mhz_median);
    json_member(out, "temp_c_max", r.temp_c_max);
    json_member(out, "cpu_mhz_median", r.cpu_mhz_median);
    json_member(out, "gpu_busy_pct_mean", r.gpu_busy_pct_mean);
    out += ",\"samples\":" + std::to_string(r.samples) + ",\"energy_source\":";
    json_text(out, r.energy_source);
    out += r.complete ? ",\"complete\":true" : ",\"complete\":false";
    json_member(out, "covered_ms", r.covered_ms);
    out += ",\"coverage\":";
    json_text(out, r.coverage);
    out += '}';
}
void sources_json(std::string& out, const std::vector<SourceStatus>& sources) {
    out += "\"sources\":[";
    bool first = true;
    for (const auto& s : sources) if (s.available) { if (!first) out += ','; first = false; json_text(out, s.name); }
    out += "],\"source_detail\":[";
    first = true;
    for (const auto& s : sources) {
        if (!first) out += ',';
        first = false;
        out += "{\"name\":"; json_text(out, s.name);
        out += s.available ? ",\"available\":true,\"detail\":" : ",\"available\":false,\"reason\":";
        json_text(out, s.detail);
        out += '}';
    }
    out += ']';
}

Receipt unavailable(std::string reason) { Receipt r; r.reason = std::move(reason); return r; }

struct Process {
    std::once_flag once;
    bool enabled = true;
    int period_ms = 100;
    std::vector<SourceStatus> sources;
    std::unique_ptr<Sampler> sampler;   // set once, read through `live`
    std::atomic<Sampler*> live{nullptr};
    std::int64_t start_ns = 0;
    std::atomic<std::int64_t> first_token{0};
};
Process& process() { static Process* p = new Process; return *p; }  // never destroyed: used from atexit

void exit_line() noexcept {
    try {
        auto& p = process();
        if (auto* s = p.live.load()) {
            s->cover(Sampler::now_ns(), kClosingWaitMs);
            Block block;
            block.enabled = true;
            block.sources = p.sources;
            block.request = s->lifetime_receipt();  // the running totals: the ring may have dropped the early samples
            block.prefill = block.decode = unavailable("process total");
            std::fprintf(stderr, "strix power: %s\n", block_json(block).c_str());
        }
        stop_process_sampler();
    } catch (...) {
    }
}

}  // namespace

void register_source(std::string kind, SourceFactory factory) {
    auto& r = registry();
    const std::lock_guard<std::mutex> lock(r.mutex);
    for (auto& entry : r.kinds) if (entry.first == kind) { entry.second = std::move(factory); return; }
    r.kinds.emplace_back(std::move(kind), std::move(factory));
}

std::vector<std::unique_ptr<Source>> make_sources(const std::string& root) {
    auto& r = registry();
    const std::lock_guard<std::mutex> lock(r.mutex);
    std::vector<std::unique_ptr<Source>> out;
    // The real machine is the filesystem root; an empty prefix would make every path relative to the working directory.
    const std::string base = root.empty() ? std::string("/") : root;
    for (const auto& entry : r.kinds) if (entry.second) if (auto source = entry.second(base)) out.push_back(std::move(source));
    return out;
}

const char* domain_name(Domain domain) noexcept {
    return domain == Domain::package ? "package" : domain == Domain::accelerator ? "accelerator" : "system";
}

void Sample::add_power(Domain domain, double watts) noexcept {
    if (!(watts >= 0.0) || !finite(watts)) { fail_power(domain); return; }
    double& slot = power_w[static_cast<std::size_t>(domain)];
    slot = finite(slot) ? slot + watts : watts;
}
void Sample::add_energy(Domain domain, double joules) noexcept {
    if (!(joules >= 0.0) || !finite(joules)) { fail_energy(domain); return; }
    double& slot = energy_j[static_cast<std::size_t>(domain)];
    slot = finite(slot) ? slot + joules : joules;
}
double Sample::total_power_w() const noexcept {
    const double whole = power_w[static_cast<std::size_t>(Domain::system)];
    if (finite(whole)) return whole;
    double sum = kAbsent;
    for (const double w : power_w) if (finite(w)) sum = finite(sum) ? sum + w : w;
    return sum;
}

bool parse_value(std::string_view text, double& value) noexcept {
    std::size_t i = 0;
    const bool negative = !text.empty() && text[0] == '-';
    if (negative || (!text.empty() && text[0] == '+')) i = 1;
    const std::size_t first = i;
    std::uint64_t magnitude = 0;
    for (; i < text.size() && text[i] >= '0' && text[i] <= '9'; ++i) {
        if (i - first >= 18) return false;  // past any channel's range, and past what a double holds exactly
        magnitude = magnitude * 10 + static_cast<std::uint64_t>(text[i] - '0');
    }
    if (i == first) return false;
    for (; i < text.size(); ++i)
        if (text[i] != ' ' && text[i] != '\t' && text[i] != '\n' && text[i] != '\r') return false;
    value = negative ? -static_cast<double>(magnitude) : static_cast<double>(magnitude);
    return true;
}

Sampler::Sampler(std::vector<std::unique_ptr<Source>> sources, Options options) : sources_(std::move(sources)), options_(options) {
    options_.period_ms = std::clamp(options_.period_ms, 1, 10000);
    options_.capacity = std::max<std::size_t>(options_.capacity, 4);
    if (options_.max_gap_ms <= 0) options_.max_gap_ms = std::max(1000, 3 * options_.period_ms);
    if (options_.min_interval_ms <= 0) options_.min_interval_ms = std::clamp(options_.period_ms / 10, 1, 5);
    options_.min_interval_ms = std::min(options_.min_interval_ms, options_.period_ms);
    for (auto& source : sources_) {
        SourceStatus status;
        try { status = source->probe(); } catch (const std::exception& e) { status.available = false; status.detail = e.what(); }
        if (status.available) live_.push_back(source.get());
        status_.push_back(std::move(status));
    }
    groups_.resize(live_.size());
    // Domain-major order keeps receipt labels stable; an unnamed source gets its own identity in each domain.
    for (std::size_t d = 0; d < kDomains; ++d) {
        std::map<std::string, std::size_t> named;
        for (std::size_t i = 0; i < live_.size(); ++i) {
            const auto domain = static_cast<Domain>(d);
            const std::string id(live_[i]->coverage_id(domain));
            const auto found = named.find(id);
            if (!id.empty() && found != named.end()) { groups_[i][d] = found->second; continue; }
            groups_[i][d] = domains_.size();
            if (!id.empty()) named.emplace(id, domains_.size());
            domains_.push_back(domain);
        }
    }
    life_.resize(domains_.size());
    expected_.resize(domains_.size());
    usable_ = !live_.empty();
    // Every ring slot owns its hardware readings from the start: storing a sample copies into storage the slot
    // already holds, so the ring's memory is fixed here and the sampling path never allocates for it. (Slots that
    // were filled lazily grew the heap by one block per sample until the ring wrapped, 65536 samples by default.)
    if (usable_) {
        Sample slot;
        slot.hardware.resize(domains_.size());
        ring_.assign(options_.capacity, slot);
    }
}

Sampler::~Sampler() { stop(); }

void Sampler::start() {
    if (!usable_) return;
    const std::lock_guard<std::mutex> lock(wake_mutex_);
    if (started_) return;
    started_ = true;
    stopping_ = false;
    thread_ = std::thread([this] { run(); });
}

void Sampler::stop() {
    {
        const std::lock_guard<std::mutex> lock(wake_mutex_);
        if (!started_) return;
        stopping_ = true;
    }
    wake_.notify_all();
    sampled_.notify_all();
    if (thread_.joinable()) thread_.join();
    const std::lock_guard<std::mutex> lock(wake_mutex_);
    started_ = false;
}

// One sample per period. cover() can ask for an earlier one; the worker then waits until min_interval_ms have passed
// since its last sample, so callers cannot drive the sensors (or the ring) faster than that.
void Sampler::run() {
    using clock = std::chrono::steady_clock;
    std::unique_lock<std::mutex> lock(wake_mutex_);
    while (!stopping_) {
        lock.unlock();
        const auto began = clock::now();
        sample_now();
        const auto stored = latest().t_ns;
        lock.lock();
        stored_ns_ = stored;
        sampled_.notify_all();
        const auto asked = [this] { return stopping_ || want_ns_ > stored_ns_; };
        wake_.wait_until(lock, began + std::chrono::milliseconds(options_.period_ms), asked);
        if (!stopping_ && want_ns_ > stored_ns_)
            wake_.wait_until(lock, began + std::chrono::milliseconds(options_.min_interval_ms), [this] { return stopping_; });
    }
}

bool Sampler::cover(std::int64_t t_ns, int timeout_ms) {
    if (!usable_) return false;
    std::unique_lock<std::mutex> lock(wake_mutex_);
    if (!started_ || stopping_) { lock.unlock(); return latest().t_ns >= t_ns; }
    if (stored_ns_ >= t_ns) return true;
    want_ns_ = std::max(want_ns_, t_ns);
    wake_.notify_all();
    return sampled_.wait_for(lock, std::chrono::milliseconds(std::max(timeout_ms, 0)), [&] { return stopping_ || stored_ns_ >= t_ns; }) &&
        stored_ns_ >= t_ns;
}

void Sampler::sample_now() {
    if (!usable_) return;
    const std::lock_guard<std::mutex> lock(read_mutex_);
    sample_locked(now_ns());
}

void Sampler::sample_at(std::int64_t t_ns) {
    if (!usable_) return;
    const std::lock_guard<std::mutex> lock(read_mutex_);
    sample_locked(t_ns);
}

void Sampler::sample_locked(std::int64_t t_ns) {
    {
        // Refused before any source is read: a stateful source (a counter baseline, a cache) must not advance for a
        // sample that will not be stored. read_mutex_ keeps the answer valid until the sample is stored below.
        const std::lock_guard<std::mutex> lock(mutex_);
        if (size_ && last_ns_ >= t_ns) return;
    }
    Sample sample;
    sample.t_ns = t_ns;
    sample.hardware.resize(domains_.size());
    std::vector<unsigned char> bad(domains_.size());
    for (std::size_t i = 0; i < live_.size(); ++i) {
        Sample part;
        part.t_ns = t_ns;
        live_[i]->read(part);
        for (std::size_t d = 0; d < kDomains; ++d) {
            const auto domain = static_cast<Domain>(d);
            auto& h = sample.hardware[groups_[i][d]];
            if (finite(part.energy_j[d])) { sample.add_energy(domain, part.energy_j[d]); h.energy_j = finite(h.energy_j) ? h.energy_j + part.energy_j[d] : part.energy_j[d]; h.expected = true; }
            if (finite(part.power_w[d])) { sample.add_power(domain, part.power_w[d]); h.power_w = finite(h.power_w) ? h.power_w + part.power_w[d] : part.power_w[d]; h.expected = true; }
            if (part.energy_bad & (1U << d)) { bad[groups_[i][d]] |= 1; h.expected = true; sample.fail_energy(domain); }
            if (part.power_bad & (1U << d)) { bad[groups_[i][d]] |= 2; h.expected = true; sample.fail_power(domain); }
        }
        if (finite(part.gpu_clock_mhz)) set_if_absent(sample.gpu_clock_mhz, part.gpu_clock_mhz);
        if (finite(part.temp_c)) set_max(sample.temp_c, part.temp_c);
        if (finite(part.cpu_mhz)) set_if_absent(sample.cpu_mhz, part.cpu_mhz);
        if (finite(part.gpu_busy_pct)) set_if_absent(sample.gpu_busy_pct, part.gpu_busy_pct);
    }
    for (std::size_t g = 0; g < bad.size(); ++g) {
        if (bad[g] & 1) sample.hardware[g].energy_j = kAbsent;
        if (bad[g] & 2) sample.hardware[g].power_w = kAbsent;
    }
    for (std::size_t d = 0; d < kDomains; ++d) {
        if (sample.energy_bad & (1U << d)) sample.energy_j[d] = kAbsent;
        if (sample.power_bad & (1U << d)) sample.power_w[d] = kAbsent;
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    for (std::size_t g = 0; g < expected_.size(); ++g) {
        const auto bit = 1U << static_cast<unsigned>(domains_[g]);
        if (finite(sample.hardware[g].energy_j)) seen_energy_ = static_cast<std::uint8_t>(seen_energy_ | bit);
        if (finite(sample.hardware[g].power_w)) seen_power_ = static_cast<std::uint8_t>(seen_power_ | bit);
        // Failed groups remain expected even when another device in the same domain succeeds.
        if (sample.hardware[g].expected) expected_[g] = 1;
    }
    if (size_)
        span(life_, life_joint_, ring_[(head_ + ring_.size() - 1) % ring_.size()], sample, kNoLo, kNoHi,
             static_cast<std::int64_t>(options_.max_gap_ms) * 1'000'000LL);
    if (!size_) first_ns_ = t_ns;
    last_ns_ = t_ns;
    ring_[head_] = sample;
    head_ = (head_ + 1) % ring_.size();
    size_ = std::min(size_ + 1, ring_.size());
}

Sample Sampler::latest() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return size_ ? ring_[(head_ + ring_.size() - 1) % ring_.size()] : Sample{};
}

bool Sampler::energy_available() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return (seen_energy_ | seen_power_) != 0;
}

Receipt Sampler::lifetime_receipt() const {
    if (!usable_) return unavailable("no readable source");
    Covers covers;
    Joint joint;
    std::vector<unsigned char> expected;
    std::int64_t total = 0;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        covers = life_;
        joint = life_joint_;
        expected = expected_;
        total = last_ns_ - first_ns_;
    }
    if (std::none_of(expected.begin(), expected.end(), [](auto e) { return e != 0; })) return unavailable("no power or energy channel");
    if (total <= 0) return unavailable("no samples");
    Receipt r;
    if (!combine(covers, joint, expected, domains_, total, r)) return unavailable("no covered stretch since the sampler started");
    r.duration_ms = static_cast<double>(total) * 1e-6;
    r.energy_source += "-lifetime";
    return r;
}

void Sampler::lifetime(double& joules, double& seconds) const {
    const auto r = lifetime_receipt();
    joules = r.available ? r.energy_j : 0.0;
    seconds = r.available ? r.covered_ms * 1e-3 : 0.0;
}

std::size_t Sampler::sample_count() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return size_;
}

// The stored samples that say something about [t0, t1], in time order: the newest one before t0 (`lead`), the
// `inside` ones within the window, and the oldest one after t1. Nothing is extended or interpolated here.
std::vector<Sample> Sampler::window_locked(std::int64_t t0_ns, std::int64_t t1_ns, std::size_t& inside, bool& lead) const {
    inside = 0;
    lead = false;
    std::vector<Sample> newest_first;
    for (std::size_t k = 0; k < size_; ++k) {
        const Sample& s = ring_[(head_ + ring_.size() - 1 - k) % ring_.size()];
        if (s.t_ns > t1_ns) { if (!newest_first.empty()) newest_first.clear(); newest_first.push_back(s); continue; }
        newest_first.push_back(s);
        if (s.t_ns < t0_ns) { lead = true; break; }
        ++inside;
    }
    return {newest_first.rbegin(), newest_first.rend()};
}

Receipt Sampler::summarize(std::int64_t t0_ns, std::int64_t t1_ns, std::uint64_t tokens) const {
    if (!usable_) return unavailable("no readable source");
    if (t1_ns <= t0_ns) return unavailable("empty window");
    std::size_t inside = 0;
    bool lead = false;
    std::vector<unsigned char> expected;
    std::vector<Sample> points;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        points = window_locked(t0_ns, t1_ns, inside, lead);
        expected = expected_;
    }
    if (std::none_of(expected.begin(), expected.end(), [](auto e) { return e != 0; })) return unavailable("no power or energy channel");
    if (points.empty()) return unavailable("no samples");
    Covers covers(domains_.size());
    Joint joint{};
    const auto gap_ns = static_cast<std::int64_t>(options_.max_gap_ms) * 1'000'000LL;
    for (std::size_t i = 1; i < points.size(); ++i) span(covers, joint, points[i - 1], points[i], t0_ns, t1_ns, gap_ns);
    Receipt r;
    if (!combine(covers, joint, expected, domains_, t1_ns - t0_ns, r)) return unavailable("window not covered by the stored samples");
    r.samples = inside;
    r.tokens = tokens;
    r.duration_ms = static_cast<double>(t1_ns - t0_ns) * 1e-6;
    r.tok_per_j = r.complete && tokens && r.energy_j > 0.0 ? static_cast<double>(tokens) / r.energy_j : 0.0;
    // Clock, temperature and load over the samples inside the window; its neighbours stand in when it has none.
    const std::size_t lo = inside && lead ? 1 : 0;
    const std::size_t hi = inside ? lo + inside : points.size();
    std::vector<double> clock, cpu;
    double busy = 0.0;
    std::size_t busy_n = 0;
    for (std::size_t i = lo; i < hi; ++i) {
        const auto& p = points[i];
        if (finite(p.gpu_clock_mhz)) clock.push_back(p.gpu_clock_mhz);
        if (finite(p.cpu_mhz)) cpu.push_back(p.cpu_mhz);
        if (finite(p.temp_c)) r.temp_c_max = std::max(r.temp_c_max, p.temp_c);
        if (finite(p.gpu_busy_pct)) { busy += p.gpu_busy_pct; ++busy_n; }
    }
    r.gpu_clock_mhz_median = median(clock);
    r.cpu_mhz_median = median(cpu);
    r.gpu_busy_pct_mean = busy_n ? busy / static_cast<double>(busy_n) : 0.0;
    return r;
}

// ---- process sampler ----------------------------------------------------------------------------------------------

void start_process_sampler() noexcept {
    try {
        auto& p = process();
        std::call_once(p.once, [&p] {
            p.enabled = envcfg::flag("STRIX_POWER_SAMPLER", true, EnvFlagMatch::exact);
            if (!p.enabled) return;
            p.period_ms = std::clamp(envcfg::integer("STRIX_POWER_SAMPLE_MS", 100), 10, 10000);
            Options options;
            options.period_ms = p.period_ms;
            auto sampler = std::make_unique<Sampler>(make_sources(), options);
            p.sources = sampler->sources();
            if (!sampler->available()) return;
            p.start_ns = Sampler::now_ns();
            sampler->start();
            p.sampler = std::move(sampler);
            p.live.store(p.sampler.get());
            std::atexit(exit_line);
        });
    } catch (...) {
    }
}

void stop_process_sampler() noexcept {
    try {
        if (auto* s = process().live.exchange(nullptr)) s->stop();
    } catch (...) {
    }
}

Sampler* process_sampler() noexcept { return process().live.load(); }
bool sampler_enabled() noexcept { start_process_sampler(); return process().enabled; }
std::vector<SourceStatus> process_sources() { start_process_sampler(); return process().sources; }

void mark_first_token() noexcept {
    auto& slot = process().first_token;
    if (slot.load(std::memory_order_relaxed)) return;
    std::int64_t expected = 0;
    slot.compare_exchange_strong(expected, Sampler::now_ns());
}
void reset_first_token() noexcept { process().first_token.store(0); }
std::int64_t first_token_ns() noexcept { return process().first_token.load(); }

namespace {
std::int64_t ms_ns(double ms) noexcept { return finite(ms) && ms > 0.0 ? static_cast<std::int64_t>(ms * 1e6) : 0; }

// A block of the process sampler. wait: ask its worker for the closing sample of the window first.
Block process_block(std::int64_t p0, std::int64_t d0, std::int64_t end, std::uint64_t prefill_tokens, std::uint64_t decode_tokens,
                    bool wait) {
    const bool enabled = sampler_enabled();
    auto* s = process_sampler();
    if (!s) {
        Block block;
        block.enabled = enabled;
        block.sources = process().sources;
        block.request = block.prefill = block.decode = unavailable(enabled ? "no readable source" : "STRIX_POWER_SAMPLER=0");
        return block;
    }
    // A hard deadline (kClosingWaitMs): the worker answers within its minimum spacing; when a sensor is slow the
    // deadline expires and the receipt says the tail of the window is not covered. Without `wait` the receipt is what
    // the ring holds now.
    if (wait) s->cover(end, kClosingWaitMs);
    return window_block(*s, p0, d0, end, prefill_tokens, decode_tokens);
}
}  // namespace

Block window_block(const Sampler& sampler, std::int64_t p0, std::int64_t d0, std::int64_t end, std::uint64_t prefill_tokens,
                   std::uint64_t decode_tokens) {
    Block block;
    block.enabled = true;
    block.sources = sampler.sources();
    d0 = std::clamp(d0, p0, std::max(p0, end));
    block.request = sampler.summarize(p0, end, prefill_tokens + decode_tokens);
    block.prefill = d0 > p0 ? sampler.summarize(p0, d0, prefill_tokens) : unavailable("no prefill phase");
    block.decode = end > d0 ? sampler.summarize(d0, end, decode_tokens) : unavailable("no decode phase");
    return block;
}

Block execution_block(std::int64_t begin_ns, std::int64_t end_ns, double decode_ms, std::uint64_t prefill_tokens,
                      std::uint64_t decode_tokens, bool wait) {
    return process_block(begin_ns, end_ns - ms_ns(decode_ms), end_ns, prefill_tokens, decode_tokens, wait);
}

Block request_block(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point stop,
                    std::uint64_t prompt_tokens, std::uint64_t generated_tokens) {
    const auto p0 = Sampler::to_ns(start), end = Sampler::to_ns(stop);
    const auto first = first_token_ns();
    const bool split = first > p0 && first < end && generated_tokens > 1;
    auto block = process_block(p0, split ? first : end, end, prompt_tokens, split ? generated_tokens - 1 : 0, true);
    // Several tokens but no first-token mark on this route: only the request total is honest.
    if (!split && generated_tokens > 1 && block.request.available) {
        block.request.tokens = prompt_tokens + generated_tokens;
        block.request.tok_per_j = block.request.complete && block.request.energy_j > 0.0
            ? static_cast<double>(block.request.tokens) / block.request.energy_j : 0.0;
        block.prefill = block.decode = unavailable("first token not marked on this route");
    }
    return block;
}

std::string block_json(const Block& block) {
    std::string out = "{\"schema\":\"strix-power-receipt-v1\",\"enabled\":";
    out += block.enabled ? "true" : "false";
    out += ",\"available\":";
    out += block.request.available ? "true," : "false,";
    sources_json(out, block.sources);
    out += ",\"request\":"; receipt_json(out, block.request);
    out += ",\"prefill\":"; receipt_json(out, block.prefill);
    out += ",\"decode\":"; receipt_json(out, block.decode);
    out += '}';
    return out;
}

std::string block_member(const Block& block) { return block.enabled ? "\"power\":" + block_json(block) : std::string(); }

std::string status_json() {
    const bool enabled = sampler_enabled();
    return status_json(process_sampler(), enabled, process().sources);
}

std::string status_json(const Sampler* s, bool enabled, const std::vector<SourceStatus>& sources) {
    // `available`: a sampler is running. `energy_available`: it has an energy counter or a power reading; a host
    // that only exposes clocks has the first and not the second, and its power members are null.
    const bool energy = s && s->energy_available();
    std::string out = "{\"schema\":\"strix-power-status-v1\",\"enabled\":";
    out += enabled ? "true" : "false";
    out += ",\"available\":";
    out += s ? "true" : "false";
    out += ",\"energy_available\":";
    out += energy ? "true," : "false,";
    sources_json(out, sources);
    if (s) {
        const auto last = s->latest();  // what the worker stored: a status request reads no sensor
        out += ",\"period_ms\":" + std::to_string(s->period_ms()) + ",\"samples\":" + std::to_string(s->sample_count()) + ",\"latest\":{\"age_ms\":";
        json_number(out, last.t_ns ? static_cast<double>(Sampler::now_ns() - last.t_ns) * 1e-6 : 0.0);
        json_reading(out, "power_w", last.total_power_w());
        json_reading(out, "gpu_clock_mhz", last.gpu_clock_mhz);
        json_reading(out, "temp_c", last.temp_c);
        json_reading(out, "cpu_mhz", last.cpu_mhz);
        json_reading(out, "gpu_busy_pct", last.gpu_busy_pct);
        const auto total = s->lifetime_receipt();
        out += "},\"since_start\":";
        if (total.available) {
            out += "{\"energy_j\":";
            json_number(out, total.energy_j);
            json_member(out, "duration_s", total.duration_ms * 1e-3);
            json_member(out, "covered_s", total.covered_ms * 1e-3);
            json_member(out, "avg_w", total.avg_w);
            out += total.complete ? ",\"complete\":true" : ",\"complete\":false";
            out += ",\"coverage\":";
            json_text(out, total.coverage);
            out += '}';
        } else {
            out += "{\"energy_j\":null,\"reason\":";
            json_text(out, total.reason);
            out += '}';
        }
    }
    out += '}';
    return out;
}

}  // namespace strix::power
