// Host tests of the shared power sampler: fake sysfs trees, no device, no thread unless stated.
#include "strix/power/sampler.hpp"
#include "strix/serve/json.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <atomic>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/stat.h>
#include <unistd.h>
#endif
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace {
using namespace strix::power;
namespace fs = std::filesystem;

void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
bool near(double a, double b, double tol = 1e-6) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b)); }
constexpr std::int64_t kSecond = 1'000'000'000LL;

struct Tree {
    fs::path root;
    explicit Tree(const char* tag) {
        root = fs::temp_directory_path() / ("strix-power-" + std::string(tag) + "-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(root);
    }
    ~Tree() {
        std::error_code error;
#if defined(__unix__) || defined(__APPLE__)
        for (fs::recursive_directory_iterator it(root, error), end; !error && it != end; it.increment(error))
            if (it->is_regular_file(error)) ::chmod(it->path().c_str(), 0644);
#endif
        fs::remove_all(root, error);
    }
    void put(const std::string& relative, const std::string& text) const {
        const auto path = root / relative;
        fs::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary | std::ios::trunc) << text << "\n";
    }
    void gpu(double watts, double mhz, double celsius, int busy = 99) const {
        put("sys/class/hwmon/hwmon5/name", "amdgpu");
        put("sys/class/hwmon/hwmon5/power1_label", "PPT");
        put("sys/class/hwmon/hwmon5/power1_average", std::to_string(static_cast<long long>(watts * 1e6)));
        put("sys/class/hwmon/hwmon5/freq1_input", std::to_string(static_cast<long long>(mhz * 1e6)));
        put("sys/class/hwmon/hwmon5/temp1_input", std::to_string(static_cast<long long>(celsius * 1e3)));
        put("sys/class/hwmon/hwmon5/device/gpu_busy_percent", std::to_string(busy));
        put("sys/class/hwmon/hwmon2/name", "k10temp");  // another driver: ignored
        put("sys/class/hwmon/hwmon2/temp1_input", "99000");
    }
    // Zone names hold ':' (not a legal file name everywhere): false where the tree cannot be made.
    bool rapl(long long uj, long long range = 1'000'000'000LL) const {
#ifdef _WIN32
        (void)uj; (void)range;
        return false;
#else
        put("sys/class/powercap/intel-rapl:0/name", "package-0");
        put("sys/class/powercap/intel-rapl:0/max_energy_range_uj", std::to_string(range));
        put("sys/class/powercap/intel-rapl:0/energy_uj", std::to_string(uj));
        put("sys/class/powercap/intel-rapl:0:0/name", "core");  // sub-zone: must not be counted twice
        put("sys/class/powercap/intel-rapl:0:0/energy_uj", "7");
        return true;
#endif
    }
    void cpu(long long khz) const {
        put("sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq", std::to_string(khz));
        put("sys/devices/system/cpu/cpufreq/policy1/scaling_cur_freq", std::to_string(khz + 2000));
    }
    [[nodiscard]] std::string path() const { return root.string(); }
};

SourceStatus status_of(const Sampler& sampler, const std::string& name) {
    for (const auto& s : sampler.sources()) if (s.name == name) return s;
    throw std::runtime_error("source not listed: " + name);
}

// Power only (no energy counter): trapezoid integral, interpolated window edges, clock median, temperature maximum.
void power_integral() {
    Tree tree("integral");
    tree.gpu(100.0, 2000.0, 60.0);
    tree.cpu(3'000'000);
    Sampler sampler(make_sources(tree.path()));
    require(sampler.available(), "amdgpu tree not available");
    require(status_of(sampler, "amdgpu-hwmon").available, "amdgpu source");
    require(status_of(sampler, "amdgpu-hwmon").detail.find("power1_average(PPT)") != std::string::npos, "amdgpu detail");
    require(!status_of(sampler, "rapl").available, "rapl must be unavailable without the tree");
    require(status_of(sampler, "rapl").detail.find("no package zone") != std::string::npos, "rapl reason");
    require(status_of(sampler, "cpufreq").available, "cpufreq source");
    sampler.sample_at(10 * kSecond);
    tree.gpu(200.0, 2700.0, 90.0);
    sampler.sample_at(11 * kSecond);
    tree.gpu(200.0, 2500.0, 80.0);
    sampler.sample_at(12 * kSecond);
    require(sampler.sample_count() == 3, "sample count");

    auto r = sampler.summarize(10 * kSecond, 11 * kSecond, 300);
    require(r.available && r.energy_source == "power-integral", "integral receipt");
    require(near(r.energy_j, 150.0) && near(r.avg_w, 150.0) && near(r.peak_w, 200.0), "100..200 W over 1 s");
    require(near(r.tok_per_j, 2.0) && r.tokens == 300 && r.samples == 2, "tok/J");
    require(near(r.gpu_clock_mhz_median, 2350.0) && near(r.temp_c_max, 90.0), "clock median / temperature max");
    require(near(r.cpu_mhz_median, 3001.0), "cpu clock mean of the policies");
    require(near(r.gpu_busy_pct_mean, 99.0), "busy");

    // Edges between samples: 125 W at 10.25 s and 175 W at 10.75 s.
    r = sampler.summarize(10 * kSecond + kSecond / 4, 10 * kSecond + 3 * kSecond / 4, 0);
    require(r.available && near(r.energy_j, 75.0) && near(r.peak_w, 175.0) && r.samples == 0, "interpolated window");
    require(r.tok_per_j == 0.0, "no tokens, no tok/J");
    // A window past the last sample has no reading: nothing is held or extended into it.
    r = sampler.summarize(12 * kSecond, 13 * kSecond, 0);
    require(!r.available && r.reason.find("not covered") != std::string::npos, "window past the last sample");
    r = sampler.summarize(11 * kSecond, 13 * kSecond, 7);
    require(r.available && !r.complete && near(r.covered_ms, 1000.0) && near(r.energy_j, 200.0) && r.tok_per_j == 0.0, "half covered window");
    require(!sampler.summarize(11 * kSecond, 11 * kSecond, 1).available, "empty window");
    // An older timestamp does not enter the ring.
    sampler.sample_at(5 * kSecond);
    require(sampler.sample_count() == 3, "ring stays ordered");
    double joules = 0.0, seconds = 0.0;
    sampler.lifetime(joules, seconds);
    require(near(joules, 350.0) && near(seconds, 2.0), "lifetime totals");
}

// Energy counter: preferred over the integral, wrap at max_energy_range_uj, sub-zones not double counted.
void counter_wrap() {
    Tree tree("wrap");
    tree.gpu(50.0, 1000.0, 40.0);
    if (!tree.rapl(999'000'000LL)) return;
    Sampler sampler(make_sources(tree.path()));
    require(status_of(sampler, "rapl").available, "rapl source");
    sampler.sample_at(1 * kSecond);
    tree.rapl(5'000'000LL);  // wrapped: 1 J to the range end + 5 J
    sampler.sample_at(2 * kSecond);
    tree.rapl(65'000'000LL);
    sampler.sample_at(3 * kSecond);
    auto r = sampler.summarize(1 * kSecond, 2 * kSecond, 12);
    require(r.available && r.energy_source == "counter", "counter receipt");
    require(near(r.energy_j, 6.0) && near(r.avg_w, 6.0) && near(r.tok_per_j, 2.0), "wrap delta");
    require(near(r.peak_w, 50.0), "peak from the power channel");
    r = sampler.summarize(1 * kSecond, 3 * kSecond, 0);
    require(near(r.energy_j, 66.0) && near(r.avg_w, 33.0), "two segments");
    r = sampler.summarize(2 * kSecond + kSecond / 2, 3 * kSecond, 0);
    require(near(r.energy_j, 30.0), "counter interpolated at the edge");
    double joules = 0.0, seconds = 0.0;
    sampler.lifetime(joules, seconds);
    require(near(joules, 66.0), "lifetime uses the counter");
}

// Counter only (no power channel): peak is the steepest segment.
void counter_only() {
    Tree tree("counter");
    if (!tree.rapl(0)) return;
    Sampler sampler(make_sources(tree.path()));
    require(sampler.available() && !status_of(sampler, "amdgpu-hwmon").available, "rapl only");
    sampler.sample_at(1 * kSecond);
    tree.rapl(10'000'000LL);
    sampler.sample_at(2 * kSecond);
    tree.rapl(40'000'000LL);
    sampler.sample_at(3 * kSecond);
    const auto r = sampler.summarize(1 * kSecond, 3 * kSecond, 0);
    require(r.available && near(r.energy_j, 40.0) && near(r.avg_w, 20.0) && near(r.peak_w, 30.0), "counter only");
    require(r.gpu_clock_mhz_median == 0.0 && r.temp_c_max == 0.0, "absent channels are zero");
}

// Nothing readable: unavailable with a reason, never a throw; the block still renders.
void missing_everything() {
    Tree tree("empty");
    Sampler sampler(make_sources(tree.path()));
    require(!sampler.available(), "empty tree must be unavailable");
    for (const auto& s : sampler.sources()) require(!s.available && !s.detail.empty(), "reason for " + s.name);
    sampler.start();
    sampler.sample_now();
    sampler.stop();
    const auto r = sampler.summarize(0, kSecond, 5);
    require(!r.available && r.reason == "no readable source", "unavailable receipt");
    Block block;
    block.enabled = true;
    block.sources = sampler.sources();
    block.request = block.prefill = block.decode = r;
    const auto json = strix::serve::parse_json(block_json(block));
    require(json.find("schema")->as_string() == "strix-power-receipt-v1", "schema");
    require(!json.find("available")->as_bool() && json.find("sources")->as_array().empty(), "block unavailable");
    require(json.find("source_detail")->as_array().size() == 3, "three built-in sources listed");
    require(json.find("prefill")->find("reason")->as_string() == "no readable source", "phase reason");
}

// A file that exists but cannot be read (the usual state of the package counter): that source is unavailable and says
// why; the others keep working. A partial hwmon directory (power file missing) still gives clock and temperature.
void permission_denied_and_partial() {
    Tree tree("denied");
    tree.gpu(80.0, 2400.0, 70.0);
    tree.rapl(1000);
    fs::remove(tree.root / "sys/class/hwmon/hwmon5/power1_average");
#if defined(__unix__) || defined(__APPLE__)
    const bool root = ::geteuid() == 0;
    ::chmod((tree.root / "sys/class/powercap/intel-rapl:0/energy_uj").c_str(), 0);
#else
    const bool root = true;
#endif
    Sampler sampler(make_sources(tree.path()));
    if (!root) {
        const auto rapl = status_of(sampler, "rapl");
        require(rapl.available && rapl.detail.find("permission denied") != std::string::npos, "rapl permission reason: " + rapl.detail);
    }
    require(status_of(sampler, "amdgpu-hwmon").available, "partial amdgpu still available");
    sampler.sample_at(kSecond);
    sampler.sample_at(2 * kSecond);
    const auto r = sampler.summarize(kSecond, 2 * kSecond, 0);
    if (!root) {
        require(!r.available && r.reason.find("not covered") != std::string::npos, "unreadable package remains expected");
    }
    require(near(sampler.latest().gpu_clock_mhz, 2400.0) && near(sampler.latest().temp_c, 70.0), "clock and temperature");
    // Garbage in a value file: not a number, source unavailable.
    Tree bad("garbage");
    bad.put("sys/class/hwmon/hwmon0/name", "amdgpu");
    bad.put("sys/class/hwmon/hwmon0/power1_average", "n/a");
    Sampler none(make_sources(bad.path()));
    require(!status_of(none, "amdgpu-hwmon").available, "garbage value accepted");
}

// A new device is one registered source: power adds up, temperature takes the maximum, no caller changes.
struct FakeCard final : Source {
    SourceStatus probe() override { return {"fake-card", true, "power temp"}; }
    void read(Sample& s) noexcept override {
        s.add_power(Domain::accelerator, 300.0);
        s.temp_c = std::isfinite(s.temp_c) ? std::max(s.temp_c, 75.0) : 75.0;
    }
};
void registered_source() {
    register_source("fake-card", [](const std::string&) { return std::make_unique<FakeCard>(); });
    Tree tree("extra");
    tree.gpu(100.0, 2000.0, 60.0);
    Sampler sampler(make_sources(tree.path()));
    require(status_of(sampler, "fake-card").available, "registered source listed");
    sampler.sample_at(kSecond);
    sampler.sample_at(2 * kSecond);
    const auto r = sampler.summarize(kSecond, 2 * kSecond, 800);
    require(near(r.energy_j, 400.0) && near(r.temp_c_max, 75.0) && near(r.tok_per_j, 2.0), "sources add up");
    register_source("fake-card", nullptr);
    Sampler again(make_sources(tree.path()));
    require(again.sources().size() == 3, "factory removed");
}

// The thread: samples arrive, stop joins, the ring wraps without losing order.
void thread_and_ring() {
    Tree tree("thread");
    tree.gpu(120.0, 2600.0, 85.0);
    Options options;
    options.period_ms = 2;
    options.capacity = 8;
    Sampler sampler(make_sources(tree.path()), options);
    const auto t0 = Sampler::now_ns();
    sampler.start();
    sampler.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    sampler.stop();
    sampler.stop();
    const auto t1 = Sampler::now_ns();
    require(sampler.sample_count() == 8, "ring full after wrap");
    const auto last = sampler.latest().t_ns;
    const auto r = sampler.summarize(last - 1'000'000, last, 0);
    require(r.available && r.complete && near(r.avg_w, 120.0, 1e-3) && near(r.gpu_clock_mhz_median, 2600.0), "threaded receipt");
    double joules = 0.0, seconds = 0.0;
    sampler.lifetime(joules, seconds);
    require(seconds > 0.1 && seconds < static_cast<double>(t1 - t0) * 1e-9 + 1e-3 && near(joules / seconds, 120.0, 1e-3), "lifetime beyond the ring");
}

// The real machine: the default root is the filesystem root, never a path relative to the working directory. Whatever
// this host exposes, a source is either available or names an absolute path in its reason; a short live run then gives
// a receipt when a power source is readable. Prints what it found (useful on a device node).
void real_machine_root() {
    Options options;
    options.period_ms = 20;
    Sampler sampler(make_sources(), options);
    for (const auto& s : sampler.sources()) {
        std::cout << "real machine: " << s.name << (s.available ? " available: " : " unavailable: ") << s.detail << "\n";
#if defined(__linux__)
        require(s.available || s.detail.find("/sys/") != std::string::npos, "relative system path in: " + s.detail);
#endif
    }
    if (!sampler.available()) return;
    const auto t0 = Sampler::now_ns();
    sampler.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sampler.stop();
    sampler.sample_now();
    const auto r = sampler.summarize(t0, Sampler::now_ns(), 0);
    if (r.available)
        std::cout << "real machine: " << r.energy_j << " J, " << r.avg_w << " W avg, " << r.peak_w << " W peak, clock "
                  << r.gpu_clock_mhz_median << " MHz, " << r.temp_c_max << " C, samples " << r.samples << ", " << r.energy_source << "\n";
    else std::cout << "real machine: no receipt: " << r.reason << "\n";
}

// The block JSON: both phases, numbers round-trip through the server's JSON reader.
void block_rendering() {
    Tree tree("json");
    tree.gpu(140.0, 2700.0, 91.0);
    Sampler sampler(make_sources(tree.path()));
    for (int i = 0; i <= 10; ++i) sampler.sample_at(i * kSecond);
    Block block;
    block.enabled = true;
    block.sources = sampler.sources();
    block.request = sampler.summarize(0, 10 * kSecond, 8192 + 127);
    block.prefill = sampler.summarize(0, 5 * kSecond, 8192);
    block.decode = sampler.summarize(5 * kSecond, 10 * kSecond, 127);
    const auto json = strix::serve::parse_json(block_json(block));
    require(json.find("available")->as_bool() && json.find("enabled")->as_bool(), "available");
    require(json.find("sources")->as_array().size() == 1 && json.find("sources")->as_array()[0].as_string() == "amdgpu-hwmon", "sources used");
    const auto* prefill = json.find("prefill");
    require(near(prefill->find("energy_j")->as_double(), 700.0) && near(prefill->find("avg_w")->as_double(), 140.0), "prefill energy");
    require(near(prefill->find("tok_per_j")->as_double(), 8192.0 / 700.0, 1e-5), "prefill tok/J");
    require(near(prefill->find("gpu_clock_mhz_median")->as_double(), 2700.0) && near(prefill->find("temp_c_max")->as_double(), 91.0), "clock / temperature");
    require(prefill->find("samples")->as_double() == 6.0 && prefill->find("energy_source")->as_string() == "power-integral", "samples / source");
    require(near(json.find("decode")->find("tok_per_j")->as_double(), 127.0 / 700.0, 1e-5), "decode tok/J");
    require(near(json.find("request")->find("energy_j")->as_double(), 1400.0), "request energy");
}

// ---- audit regressions ---------------------------------------------------------------------------------------------

// A source driven by the test: counts its reads and notices two threads inside read() at once.
struct Scripted final : Source {
    std::function<void(Sample&)> fn;
    std::atomic<int> reads{0}, inside{0};
    std::atomic<bool> overlap{false};
    SourceStatus probe() override { return {"scripted", true, "test"}; }
    void read(Sample& s) noexcept override {
        if (inside.fetch_add(1) != 0) overlap = true;
        ++reads;
        if (fn) fn(s);
        std::this_thread::yield();  // widen the window in which a second reader would be caught
        inside.fetch_sub(1);
    }
};
std::unique_ptr<Sampler> scripted(Scripted*& source, std::function<void(Sample&)> fn, Options options = {}) {
    auto owned = std::make_unique<Scripted>();
    owned->fn = std::move(fn);
    source = owned.get();
    std::vector<std::unique_ptr<Source>> sources;
    sources.push_back(std::move(owned));
    return std::make_unique<Sampler>(std::move(sources), options);
}
double seconds_of(const Sample& s) { return static_cast<double>(s.t_ns) * 1e-9; }

// Disjoint cards in one domain must both survive counter/power selection, including lifetime ring eviction.
void disjoint_hardware_channels() {
    std::vector<std::unique_ptr<Source>> sources;
    auto counter = std::make_unique<Scripted>();
    counter->fn = [](Sample& s) { s.add_energy(Domain::accelerator, 50.0 * seconds_of(s)); };
    auto power = std::make_unique<Scripted>();
    power->fn = [](Sample& s) {
        if (seconds_of(s) == 3.0) s.fail_power(Domain::accelerator);
        else s.add_power(Domain::accelerator, 300.0);
    };
    sources.push_back(std::move(counter));
    sources.push_back(std::move(power));
    Options options;
    options.capacity = 4;
    Sampler sampler(std::move(sources), options);
    sampler.sample_at(0);
    sampler.sample_at(kSecond);
    const auto r = sampler.summarize(0, kSecond, 700);
    require(r.available && r.complete && near(r.energy_j, 350.0) && near(r.tok_per_j, 2.0), "both cards contribute");
    require(r.energy_source == "mixed", "counter and power remain distinct");
    for (int i = 2; i <= 6; ++i) sampler.sample_at(i * kSecond);
    const auto total = sampler.lifetime_receipt();
    require(total.available && !total.complete && near(total.energy_j, 1500.0) && near(total.covered_ms, 4000.0), "lifetime retains both cards and their coverage");
    const auto failed = sampler.summarize(3 * kSecond, 4 * kSecond, 100);
    require(failed.available && !failed.complete && near(failed.energy_j, 50.0) && failed.tok_per_j == 0.0, "failed card stays expected");
}

void rapl_probe_recovery() {
#ifndef _WIN32
    // An empty member and a missing file both recover; the latter must reopen its descriptor.
    for (const bool missing : {false, true}) {
        Tree tree("probe-recovery");
        tree.rapl(0);
        const std::string zone = "sys/class/powercap/intel-rapl:1/";
        tree.put(zone + "name", "package-1");
        tree.put(zone + "max_energy_range_uj", "1000000");
        if (!missing) tree.put(zone + "energy_uj", "");
        Sampler sampler(make_sources(tree.path()));
        require(status_of(sampler, "rapl").detail.find("probe failed") != std::string::npos, "failed member reported alongside readable member");
        sampler.sample_at(0);
        tree.rapl(100000);
        sampler.sample_at(kSecond);
        const auto partial = sampler.summarize(0, kSecond, 10);
        require(!partial.complete && partial.tok_per_j == 0.0 && !partial.available, "probe failure cannot certify partial package");
        tree.put(zone + "energy_uj", "100000");
        sampler.sample_at(2 * kSecond);  // establish the recovered member's first baseline
        tree.rapl(300000);
        tree.put(zone + "energy_uj", "400000");
        sampler.sample_at(3 * kSecond);
        const auto recovered = sampler.summarize(2 * kSecond, 3 * kSecond, 10);
        require(recovered.complete && near(recovered.energy_j, 0.5), "recovered package includes both zones");
        require(!sampler.lifetime_receipt().complete, "recovery does not cover the earlier failure");
    }
#endif
}

void rapl_invalid_probe_baseline() {
    Tree tree("invalid-baseline");
    if (!tree.rapl(2500000, 1000000)) return;
    Sampler sampler(make_sources(tree.path()));
    require(status_of(sampler, "rapl").detail.find("invalid baseline") != std::string::npos, "invalid probe baseline reported");
    sampler.sample_at(0);
    tree.rapl(100000, 1000000);
    sampler.sample_at(kSecond);
    require(near(sampler.latest().energy_j[0], 0.0), "first valid baseline adds no delta");
    tree.rapl(200000, 1000000);
    sampler.sample_at(2 * kSecond);
    const auto r = sampler.summarize(kSecond, 2 * kSecond, 1);
    require(r.complete && near(r.energy_j, 0.1), "invalid predecessor cannot poison recovery");
}

// The ring's memory is fixed when the sampler is built: storing samples (thousands, as a server whose every finished
// request asks cover() for one) does not grow the heap before the ring wraps. glibc's statistics only; a sanitizer's
// allocator replaces them.
void ring_memory_fixed_at_construction() {
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define STRIX_POWER_TEST_SANITIZED 1
#endif
#endif
#if defined(__GLIBC__) && !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__) && !defined(STRIX_POWER_TEST_SANITIZED)
    Tree tree("ringmem");
    tree.gpu(30.0, 2900.0, 50.0);
    Sampler sampler(make_sources(tree.path()));  // the default ring (65536 samples), as the process sampler's
    require(sampler.available(), "amdgpu tree available");
    std::int64_t t = kSecond;
    for (int i = 0; i < 16; ++i) sampler.sample_at(t += kSecond / 1000);  // the sampling path's own transient blocks
    const auto heap = [] { malloc_trim(0); const auto m = mallinfo2(); return static_cast<long long>(m.uordblks + m.hblkhd); };
    const auto before = heap();
    constexpr int kSamples = 4000;
    for (int i = 0; i < kSamples; ++i) sampler.sample_at(t += kSecond / 1000);
    const auto growth = heap() - before;
    std::printf("heap over %d stored samples: %+lld bytes\n", kSamples, growth);
    require(growth < 1024, "ring memory grows with the stored samples: " + std::to_string(growth) + " bytes");
#else
    std::puts("SKIP ring_memory_fixed_at_construction: requires glibc without ASan/TSan");
#endif
#undef STRIX_POWER_TEST_SANITIZED
}

// Finding 2: sample_at() from other threads while the worker runs never enters a source twice at once, and a
// timestamp that will not be stored does not advance a stateful source.
void serialized_reads() {
    Scripted* source = nullptr;
    Options options;
    options.period_ms = 1;
    auto sampler = scripted(source, [](Sample& s) { s.add_power(Domain::package, 10.0); }, options);
    sampler->start();
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&] { for (int i = 0; i < 2000; ++i) { if (i % 2) sampler->sample_at(Sampler::now_ns()); else sampler->sample_now(); } });
    for (auto& thread : threads) thread.join();
    sampler->stop();
    require(!source->overlap.load(), "two threads read a source at the same time");
    const int before = source->reads.load();
    sampler->sample_at(sampler->latest().t_ns - 5);
    sampler->sample_at(sampler->latest().t_ns);
    require(source->reads.load() == before, "a refused timestamp still read the sources");
}

// Finding 4: a status view and a receipt read no sensor, and cover() cannot drive the worker past its minimum spacing.
void status_and_receipts_read_no_sensor() {
    Scripted* source = nullptr;
    Options options;
    options.period_ms = 1000;
    options.min_interval_ms = 20;
    auto sampler = scripted(source, [](Sample& s) { s.add_power(Domain::package, 10.0); }, options);
    sampler->sample_at(Sampler::now_ns() - kSecond / 2);
    sampler->sample_at(Sampler::now_ns());
    const int before = source->reads.load();
    const auto end = sampler->latest().t_ns;
    for (int i = 0; i < 200; ++i) {
        (void)status_json(sampler.get(), true, sampler->sources());
        (void)block_json(window_block(*sampler, end - kSecond / 4, end - kSecond / 8, end, 1, 1));
    }
    require(source->reads.load() == before, "status or receipt rendering read a sensor");
    require(sampler->cover(end, 0) && !sampler->cover(end + kSecond, 0), "cover without a worker only reports");
    require(source->reads.load() == before, "cover without a worker read a sensor");
    sampler->start();
    const auto t0 = Sampler::now_ns();
    int covered = 0;
    for (int i = 0; i < 400 && Sampler::now_ns() - t0 < 300'000'000LL; ++i) covered += sampler->cover(Sampler::now_ns(), 200) ? 1 : 0;
    const double elapsed_ms = static_cast<double>(Sampler::now_ns() - t0) * 1e-6;
    sampler->stop();
    const int worker_reads = source->reads.load() - before;
    require(covered > 0, "cover never obtained a closing sample");
    require(worker_reads <= static_cast<int>(elapsed_ms / 20.0) + 3,
            "worker sampled faster than its minimum spacing: " + std::to_string(worker_reads) + " reads in " + std::to_string(elapsed_ms) + " ms");
}

// Finding 5: the block of an execution span depends on the span alone: built again later, after more samples, it is
// the same text; and it is the energy of that span, not of a window counted back from "now".
void execution_window_is_immutable() {
    Scripted* source = nullptr;
    auto sampler = scripted(source, [](Sample& s) { s.add_power(Domain::package, seconds_of(s) < 8.5 ? 100.0 : 900.0); });
    for (int i = 0; i <= 8; ++i) sampler->sample_at(i * kSecond);
    const auto first = block_json(window_block(*sampler, 2 * kSecond, 6 * kSecond, 8 * kSecond, 4000, 60));
    for (int i = 9; i <= 20; ++i) sampler->sample_at(i * kSecond);  // the response is rendered 12 s later, under load
    const auto later = block_json(window_block(*sampler, 2 * kSecond, 6 * kSecond, 8 * kSecond, 4000, 60));
    require(first == later, "the receipt of one execution span changed with the time it was rendered");
    const auto json = strix::serve::parse_json(first);
    require(near(json.find("request")->find("energy_j")->as_double(), 600.0) && json.find("request")->find("complete")->as_bool(), "request span");
    require(near(json.find("prefill")->find("energy_j")->as_double(), 400.0) && near(json.find("decode")->find("energy_j")->as_double(), 200.0), "phase spans");
    require(near(json.find("decode")->find("tok_per_j")->as_double(), 60.0 / 200.0), "decode tok/J");
}

// Finding 6: a window older than the ring is not filled in from the oldest retained sample; the totals since the start
// come from the running accumulator, not from the ring.
void expired_history() {
    Scripted* source = nullptr;
    Options options;
    options.capacity = 4;
    auto sampler = scripted(source, [](Sample& s) { s.add_energy(Domain::package, 100.0 * seconds_of(s)); }, options);
    for (int i = 1; i <= 10; ++i) sampler->sample_at(i * kSecond);  // 100 W for 9 s; the ring keeps 7..10 s
    auto r = sampler->summarize(1 * kSecond, 10 * kSecond, 90);
    require(r.available && !r.complete, "a window reaching past the ring must not be complete");
    require(near(r.energy_j, 300.0) && near(r.covered_ms, 3000.0) && near(r.avg_w, 100.0) && r.tok_per_j == 0.0, "only the retained 3 s are reported");
    r = sampler->summarize(1 * kSecond, 3 * kSecond, 0);
    require(!r.available && r.reason.find("not covered") != std::string::npos, "an expired window is unavailable");
    const auto total = sampler->lifetime_receipt();
    require(total.available && total.complete && near(total.energy_j, 900.0) && near(total.duration_ms, 9000.0) && near(total.avg_w, 100.0), "process total");
    require(total.energy_source == "counter-lifetime", "process total source");
}

// Finding 7: a package counter does not hide a card. Disjoint domains add up and the receipt names both; inside one
// domain the counter replaces the power reading instead of being added to it.
void disjoint_domains() {
    Scripted* source = nullptr;
    auto sampler = scripted(source, [](Sample& s) {
        s.add_energy(Domain::package, 50.0 * seconds_of(s));
        s.add_power(Domain::package, 55.0);       // the same package through its power reading: not added
        s.add_power(Domain::accelerator, 300.0);  // a card outside the package
    });
    for (int i = 1; i <= 3; ++i) sampler->sample_at(i * kSecond);
    const auto r = sampler->summarize(1 * kSecond, 3 * kSecond, 700);
    require(r.available && r.complete && near(r.energy_j, 700.0) && near(r.avg_w, 350.0), "package counter plus card integral");
    require(r.coverage == "package:counter+accelerator:power-integral" && r.energy_source == "mixed", "coverage: " + r.coverage);
    require(near(r.peak_w, 355.0) && near(r.tok_per_j, 1.0), "peak and tok/J over both domains");
    require(near(sampler->latest().total_power_w(), 355.0), "latest power sums the domains");
    // A whole-machine meter replaces the parts.
    auto metered = scripted(source, [](Sample& s) { s.add_power(Domain::system, 400.0); s.add_power(Domain::accelerator, 300.0); });
    metered->sample_at(kSecond);
    metered->sample_at(2 * kSecond);
    const auto m = metered->summarize(kSecond, 2 * kSecond, 0);
    require(near(m.energy_j, 400.0) && m.coverage == "system:power-integral", "system domain");
    // A domain the sampler measures but that has no reading in the window is named, and the receipt is not complete.
    auto lost = scripted(source, [](Sample& s) {
        s.add_power(Domain::package, 50.0);
        if (seconds_of(s) < 2.5) s.add_power(Domain::accelerator, 300.0);
    });
    for (int i = 1; i <= 6; ++i) lost->sample_at(i * kSecond);
    const auto l = lost->summarize(4 * kSecond, 6 * kSecond, 10);
    require(l.available && !l.complete && near(l.energy_j, 100.0) && l.coverage == "package:power-integral+accelerator:none" && l.tok_per_j == 0.0,
            "lost domain: " + l.coverage);
}

// Finding 8: a counter made of several zones is published only when every zone was read. A sample in which one zone
// failed carries no counter (it is not the sum of the zones that answered), and the energy that zone used meanwhile
// is not moved into the stretch in which it recovers.
void partial_counter_failure() {
    // Any source: a failed part of a domain removes the domain from the sample.
    Scripted* source = nullptr;
    auto sampler = scripted(source, [](Sample& s) {
        s.add_energy(Domain::package, 10.0 * seconds_of(s));
        if (seconds_of(s) > 1.5 && seconds_of(s) < 2.5) s.fail_energy(Domain::package);  // the second zone did not answer
        else s.add_energy(Domain::package, 10.0 * seconds_of(s));
    });
    for (int i = 1; i <= 4; ++i) sampler->sample_at(i * kSecond);
    auto r = sampler->summarize(1 * kSecond, 2 * kSecond, 0);
    require(!r.available, "a window ending in the failure reported a partial counter");
    r = sampler->summarize(3 * kSecond, 4 * kSecond, 0);
    require(r.available && r.complete && near(r.energy_j, 20.0), "after the failure");
    r = sampler->summarize(1 * kSecond, 4 * kSecond, 0);
    require(r.available && !r.complete && near(r.energy_j, 20.0) && near(r.covered_ms, 1000.0), "the failure is a gap, not energy");

    // The built-in package counter, two zones (zone names need ':').
#ifndef _WIN32
    Tree tree("zones");
    const auto zone = [&](int index, long long uj) {
        const auto dir = "sys/class/powercap/intel-rapl:" + std::to_string(index) + "/";
        tree.put(dir + "name", "package-" + std::to_string(index));
        tree.put(dir + "max_energy_range_uj", "1000000000");
        tree.put(dir + "energy_uj", std::to_string(uj));
    };
    zone(0, 0); zone(1, 0);
    Options options;
    options.max_gap_ms = 2500;
    Sampler rapl(make_sources(tree.path()), options);
    require(status_of(rapl, "rapl").available, "two zones");
    rapl.sample_at(1 * kSecond);
    zone(0, 10'000'000);
    tree.put("sys/class/powercap/intel-rapl:1/energy_uj", "");  // zone 1 cannot be read at 2 s
    rapl.sample_at(2 * kSecond);
    require(!std::isfinite(rapl.latest().energy_j[0]), "partial package total published");
    zone(0, 20'000'000); zone(1, 20'000'000);
    rapl.sample_at(3 * kSecond);
    zone(0, 30'000'000); zone(1, 30'000'000);
    rapl.sample_at(4 * kSecond);
    r = rapl.summarize(1 * kSecond, 2 * kSecond, 0);
    require(!r.available, "the second that ends in the failed read reported the 10 J of the zone that answered");
    r = rapl.summarize(2 * kSecond, 3 * kSecond, 0);
    require(!r.available, "the recovery second reported energy (30 J with the other second's energy before the fix)");
    r = rapl.summarize(3 * kSecond, 4 * kSecond, 0);
    require(r.available && r.complete && near(r.energy_j, 20.0), "after recovery");
    require(near(rapl.latest().energy_j[0], 60.0), "the cumulative total holds both zones again");
    r = rapl.summarize(1 * kSecond, 4 * kSecond, 0);
    require(r.available && !r.complete && near(r.energy_j, 20.0) && near(r.covered_ms, 1000.0), "the failure is not bridged within the gap limit");
    // A value past the counter's range is a failed read.
    zone(0, 2'000'000'000LL);
    rapl.sample_at(5 * kSecond);
    require(!std::isfinite(rapl.latest().energy_j[0]), "counter value past its range accepted");
#endif
}

// Finding 9: failed power reads in the middle of a window are a gap in the receipt, not silence.
void missing_power_interval() {
    Scripted* source = nullptr;
    auto sampler = scripted(source, [](Sample& s) {
        const double t = seconds_of(s);
        if (t > 1.5 && t < 3.5) s.fail_power(Domain::package); else s.add_power(Domain::package, 100.0);
    });
    for (int i = 0; i <= 5; ++i) sampler->sample_at(i * kSecond);
    const auto r = sampler->summarize(0, 5 * kSecond, 500);
    require(r.available && !r.complete, "a window with failed reads must not be complete");
    require(near(r.energy_j, 200.0) && near(r.covered_ms, 2000.0) && near(r.avg_w, 100.0) && near(r.duration_ms, 5000.0), "covered part only");
    require(r.tok_per_j == 0.0, "tok/J of a partial energy");
    const auto json = strix::serve::parse_json(block_json(window_block(*sampler, 0, 5 * kSecond, 5 * kSecond, 500, 0)));
    require(!json.find("request")->find("complete")->as_bool() && near(json.find("request")->find("covered_ms")->as_double(), 2000.0), "block says so");
    // A failed read is never bridged, however short and whatever the gap limit: at the default 100 ms period, valid
    // readings at 0 and 200 ms around a failed one at 100 ms cover nothing.
    constexpr std::int64_t kTick = 100'000'000LL;
    Options options;
    options.max_gap_ms = 3000;
    for (const Options& each : {Options{}, options}) {
        auto cadence = scripted(source, [](Sample& s) {
            if (s.t_ns == kTick) s.fail_power(Domain::package); else s.add_power(Domain::package, 100.0);
        }, each);
        for (int i = 0; i <= 3; ++i) cadence->sample_at(i * kTick);
        require(!cadence->summarize(0, 2 * kTick, 20).available, "a failed read between two valid ones was bridged");
        const auto c = cadence->summarize(0, 3 * kTick, 30);
        require(c.available && !c.complete && near(c.energy_j, 10.0) && near(c.covered_ms, 100.0) && c.tok_per_j == 0.0, "only 200..300 ms is covered");
        double joules = 0.0, seconds = 0.0;
        cadence->lifetime(joules, seconds);
        require(near(joules, 10.0) && near(seconds, 0.1), "lifetime bridges no failed read either");
    }
    // The gap limit still bounds two consecutive valid samples that are far apart.
    auto sparse = scripted(source, [](Sample& s) { s.add_power(Domain::package, 100.0); }, options);
    sparse->sample_at(0);
    sparse->sample_at(2 * kSecond);
    sparse->sample_at(6 * kSecond);
    require(sparse->summarize(0, 2 * kSecond, 0).complete && !sparse->summarize(2 * kSecond, 6 * kSecond, 0).available, "gap limit");
}

// A domain whose source is there but whose reads all fail stays in the receipt: the receipt is not complete and carries
// no tok/J, instead of reporting the other domain as the whole.
void failing_domain_stays_expected() {
    Scripted* source = nullptr;
    auto sampler = scripted(source, [](Sample& s) { s.add_power(Domain::package, 50.0); s.fail_power(Domain::accelerator); });
    for (int i = 1; i <= 3; ++i) sampler->sample_at(i * kSecond);
    const auto r = sampler->summarize(1 * kSecond, 3 * kSecond, 100);
    require(r.available && !r.complete, "a domain that only fails dropped out of the receipt");
    require(r.coverage == "package:power-integral+accelerator:none" && near(r.energy_j, 100.0), "coverage: " + r.coverage);
    require(r.covered_ms == 0.0 && r.tok_per_j == 0.0, "no common coverage, no tok/J");
    const auto total = sampler->lifetime_receipt();
    require(total.available && !total.complete && total.covered_ms == 0.0, "lifetime keeps the failing domain");
    require(sampler->energy_available(), "energy_available is about readings that succeeded");
    // Only failures: nothing to report, but still a measured domain.
    auto dead = scripted(source, [](Sample& s) { s.fail_energy(Domain::package); });
    dead->sample_at(kSecond);
    dead->sample_at(2 * kSecond);
    require(!dead->energy_available() && dead->summarize(kSecond, 2 * kSecond, 0).reason.find("not covered") != std::string::npos, "all reads failed");
}

// covered_ms is the time every domain covers at once (the intersection), not the shortest of their covered times.
void covered_time_is_the_intersection() {
    constexpr std::int64_t kTick = 100'000'000LL;
    Scripted* source = nullptr;
    // package covers 0..100 ms, the card 200..300 ms: no instant has both.
    auto apart = scripted(source, [](Sample& s) {
        if (s.t_ns <= kTick) s.add_power(Domain::package, 50.0); else s.fail_power(Domain::package);
        if (s.t_ns >= 2 * kTick) s.add_power(Domain::accelerator, 300.0); else s.fail_power(Domain::accelerator);
    });
    for (int i = 0; i <= 3; ++i) apart->sample_at(i * kTick);
    auto r = apart->summarize(0, 3 * kTick, 0);
    require(r.available && !r.complete && near(r.energy_j, 35.0), "both parts are reported");
    require(r.covered_ms == 0.0, "disjoint coverage reported as common: " + std::to_string(r.covered_ms));
    double joules = 0.0, seconds = 0.0;
    apart->lifetime(joules, seconds);
    require(near(joules, 35.0) && seconds == 0.0, "lifetime common coverage of disjoint domains");
    // package covers 0..200 ms, the card 100..300 ms: 100 ms in common.
    auto overlap = scripted(source, [](Sample& s) {
        if (s.t_ns <= 2 * kTick) s.add_power(Domain::package, 50.0); else s.fail_power(Domain::package);
        if (s.t_ns >= kTick) s.add_power(Domain::accelerator, 300.0); else s.fail_power(Domain::accelerator);
    });
    for (int i = 0; i <= 3; ++i) overlap->sample_at(i * kTick);
    r = overlap->summarize(0, 3 * kTick, 0);
    require(!r.complete && near(r.energy_j, 70.0) && near(r.covered_ms, 100.0), "partial overlap: " + std::to_string(r.covered_ms));
    overlap->lifetime(joules, seconds);
    require(near(seconds, 0.1), "lifetime partial overlap");
    r = overlap->summarize(kTick, 2 * kTick, 0);
    require(r.complete && near(r.covered_ms, 100.0) && near(r.energy_j, 35.0), "the common part alone is complete");
}

// The closing wait is a deadline: a sensor that hangs costs a request at most kClosingWaitMs, and the receipt then says
// the tail is not covered.
void closing_wait_has_a_deadline() {
    require(kClosingWaitMs <= 10, "closing wait above 10 ms");
    Scripted* source = nullptr;
    std::atomic<bool> slow{false};
    Options options;
    options.period_ms = 20;
    auto sampler = scripted(source, [&](Sample& s) {
        s.add_power(Domain::package, 100.0);
        if (slow.load()) std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }, options);
    sampler->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    slow = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(60));  // the worker is now inside a slow read
    const auto begin = Sampler::now_ns() - 50'000'000LL;
    const auto end = Sampler::now_ns();
    const bool reached = sampler->cover(end, kClosingWaitMs);
    const double waited_ms = static_cast<double>(Sampler::now_ns() - end) * 1e-6;
    const auto r = sampler->summarize(begin, end, 5);
    slow = false;
    sampler->stop();
    require(!reached, "a blocked sensor cannot have delivered the closing sample");
    require(waited_ms < 100.0, "the closing wait ran past its deadline: " + std::to_string(waited_ms) + " ms");
    require(!r.complete && r.tok_per_j == 0.0, "a receipt without its closing sample must be incomplete");
}

// Finding 10: a host that exposes clocks but no power is not a host that draws 0 W.
void clock_only_status() {
    Tree tree("clocks");
    tree.cpu(3'000'000);
    Sampler sampler(make_sources(tree.path()));
    require(sampler.available() && !sampler.energy_available(), "clock-only sampler");
    sampler.sample_at(kSecond);
    sampler.sample_at(2 * kSecond);
    const auto json = strix::serve::parse_json(status_json(&sampler, true, sampler.sources()));
    require(json.find("available")->as_bool() && !json.find("energy_available")->as_bool(), "energy_available must be false");
    require(json.find("latest")->find("power_w")->is_null() && json.find("latest")->find("gpu_clock_mhz")->is_null(), "absent readings are null");
    require(near(json.find("latest")->find("cpu_mhz")->as_double(), 3001.0), "the clock that is there");
    require(json.find("since_start")->find("energy_j")->is_null(), "no energy total");
    require(!sampler.summarize(kSecond, 2 * kSecond, 1).available, "no receipt");
    // With a power channel the same view carries numbers.
    Tree full("clocks-power");
    full.gpu(60.0, 2000.0, 50.0);
    Sampler powered(make_sources(full.path()));
    powered.sample_at(kSecond);
    powered.sample_at(2 * kSecond);
    const auto ok = strix::serve::parse_json(status_json(&powered, true, powered.sources()));
    require(ok.find("energy_available")->as_bool() && near(ok.find("latest")->find("power_w")->as_double(), 60.0), "power status");
    require(near(ok.find("since_start")->find("energy_j")->as_double(), 60.0), "energy total");
    const auto off = strix::serve::parse_json(status_json(nullptr, false, {}));
    require(!off.find("enabled")->as_bool() && !off.find("available")->as_bool() && !off.find("latest"), "off status");
}

// Finding 11: with the sampler switched off a CLI prints no "power" member at all.
void off_switch_member() {
    Block block;
    require(block_member(block).empty(), "a disabled block must print nothing");
    block.enabled = true;
    block.request = block.prefill = block.decode = Receipt{};
    const auto member = block_member(block);
    require(member.rfind("\"power\":{\"schema\":\"strix-power-receipt-v1\"", 0) == 0, "enabled member: " + member);
}

// Finding 12: a value file holds one complete non-negative number or it is not read.
void strict_parsing() {
    double v = -1.0;
    require(parse_value("100000000\n", v) && v == 100000000.0, "plain number");
    require(parse_value("0", v) && v == 0.0 && parse_value("-5000\n", v) && v == -5000.0, "zero and sign");
    for (const char* bad : {"100000000garbage", "", "\n", "-", "12 34", "1e3", "0x10", "nan", "inf", "1.5", "1234567890123456789012", " 5"})
        require(!parse_value(bad, v), std::string("accepted: '") + bad + "'");
    const auto card = [](const char* tag, const std::string& power) {
        auto tree = std::make_unique<Tree>(tag);
        tree->put("sys/class/hwmon/hwmon0/name", "amdgpu");
        tree->put("sys/class/hwmon/hwmon0/power1_average", power);
        return tree;
    };
    {
        const auto tree = card("prefix", "100000000garbage");
        Sampler sampler(make_sources(tree->path()));
        require(!status_of(sampler, "amdgpu-hwmon").available, "numeric prefix accepted as 100 W");
    }
    {
        const auto tree = card("negative", "-100000000");
        Sampler sampler(make_sources(tree->path()));
        require(!status_of(sampler, "amdgpu-hwmon").available, "negative power accepted");
    }
    {
        const auto tree = card("turns-bad", "100000000");
        Sampler sampler(make_sources(tree->path()));
        require(status_of(sampler, "amdgpu-hwmon").available, "good card");
        sampler.sample_at(kSecond);
        tree->put("sys/class/hwmon/hwmon0/power1_average", "-100000000");
        sampler.sample_at(2 * kSecond);
        require(!std::isfinite(sampler.latest().power_w[0]), "negative reading stored");
        tree->put("sys/class/hwmon/hwmon0/power1_average", "100000000 W");
        sampler.sample_at(3 * kSecond);
        require(!std::isfinite(sampler.latest().power_w[0]), "trailing bytes stored");
        const auto r = sampler.summarize(kSecond, 3 * kSecond, 0);
        require(!r.available, "a receipt from invalid readings");
    }
    // A negative temperature is a temperature.
    Tree cold("cold");
    cold.put("sys/class/hwmon/hwmon0/name", "amdgpu");
    cold.put("sys/class/hwmon/hwmon0/temp1_input", "-5000");
    Sampler sampler(make_sources(cold.path()));
    sampler.sample_at(kSecond);
    require(near(sampler.latest().temp_c, -5.0), "negative temperature");
}

}  // namespace

int main() {
    try {
        power_integral();
        counter_wrap();
        counter_only();
        missing_everything();
        permission_denied_and_partial();
        registered_source();
        thread_and_ring();
        block_rendering();
        ring_memory_fixed_at_construction();
        serialized_reads();
        status_and_receipts_read_no_sensor();
        execution_window_is_immutable();
        expired_history();
        disjoint_domains();
        disjoint_hardware_channels();
        rapl_probe_recovery();
        rapl_invalid_probe_baseline();
        partial_counter_failure();
        missing_power_interval();
        failing_domain_stays_expected();
        covered_time_is_the_intersection();
        closing_wait_has_a_deadline();
        clock_only_status();
        off_switch_member();
        strict_parsing();
        real_machine_root();
        std::cout << "power sampler tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "power sampler test failed: " << e.what() << "\n";
        return 1;
    }
}
