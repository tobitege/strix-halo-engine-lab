// Energy and power as a shared measurement: host only, model agnostic, no device runtime.
//
// A Sampler owns a list of Sources and one worker thread that reads them at a fixed period into a fixed ring of
// Samples. Nothing here needs elevated rights or changes a system setting: a source reads what the machine already
// exposes. Missing sources report why; discovered package counters retry unreadable members. A run never
// fails because of this file.
//
// Domains: every energy counter and power reading belongs to one Domain (package, accelerator, system). Energy of a
// window is computed per physical coverage group (counter when it covers at least as much as that group's power
// readings, otherwise the power integral). Disjoint groups are added, including those in the same domain. A receipt
// names what it covers ("package:counter+accelerator:power-integral"): it is the energy of those domains, not of the
// whole machine, unless the coverage says "system".
//
// Coverage: nothing is extended or invented. A channel covers the stretch between two consecutive samples when both
// hold a valid reading of it and they are at most Options::max_gap_ms apart (linear interpolation inside it). Anything
// else is not covered: a failed read (the stretches on both sides of it), a window older than the ring, a window that
// ends after the last sample. A domain a source owns is expected in every receipt, also when its reads only fail. A
// receipt carries complete and covered_ms (the time every expected hardware group covers at once); an incomplete receipt
// reports the energy of the covered parts only and no tok_per_j.
//
// Threads: the sources are read by one thread at a time (the worker, or a caller of sample_now/sample_at). Readers of
// the ring never touch a sensor: status and receipts use what the worker stored. cover() asks the worker for one
// closing sample and waits for it up to a deadline (kClosingWaitMs for a request); the worker never samples faster
// than Options::min_interval_ms.
//
// Process sampler: start_process_sampler() starts one sampler per process (the device gate calls it, so an engine does
// not). The receipt block "strix-power-receipt-v1" is what every CLI and the server print. Switch:
// STRIX_POWER_SAMPLER=0 turns the sampler and every receipt member off; STRIX_POWER_SAMPLE_MS sets the period (default
// 100, range 10..10000).
//
// A new device (a discrete card, an accelerator, a wall meter, another OS) is one Source subclass plus one
// register_source() call; no caller changes. See docs/engine/power-receipt.md.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace strix::power {

inline constexpr double kAbsent = std::numeric_limits<double>::quiet_NaN();

// What a counter or a power reading measures. package: the CPU or SoC package (an integrated GPU is inside it);
// accelerator: discrete cards, outside every package; system: the whole machine (a wall or board meter), which
// replaces the other two when present.
enum class Domain : std::uint8_t { package = 0, accelerator = 1, system = 2 };
inline constexpr std::size_t kDomains = 3;
[[nodiscard]] const char* domain_name(Domain domain) noexcept;

// One reading of every channel. A channel nobody filled stays NaN.
struct Sample {
    struct Reading { double energy_j = kAbsent, power_w = kAbsent; bool expected = false; };
    std::vector<Reading> hardware;  // sampler-owned physical coverage groups, independent of domain totals
    std::int64_t t_ns = 0;  // steady clock
    std::array<double, kDomains> energy_j{kAbsent, kAbsent, kAbsent};  // cumulative since the source's first read
    std::array<double, kDomains> power_w{kAbsent, kAbsent, kAbsent};   // instantaneous
    double gpu_clock_mhz = kAbsent;
    double temp_c = kAbsent;      // maximum over the sources that report one
    double cpu_mhz = kAbsent;     // mean over the CPU frequency policies
    double gpu_busy_pct = kAbsent;
    std::uint8_t energy_bad = 0, power_bad = 0;  // domains a source failed to read in this sample (bit = domain)

    // Readings within one source and domain add up. A negative or non-finite value is a failed read.
    void add_power(Domain domain, double watts) noexcept;
    void add_energy(Domain domain, double joules) noexcept;
    // A source that could not read part of its hardware group: that channel is absent, never a partial total.
    void fail_power(Domain domain) noexcept { power_bad = static_cast<std::uint8_t>(power_bad | (1U << static_cast<unsigned>(domain))); }
    void fail_energy(Domain domain) noexcept { energy_bad = static_cast<std::uint8_t>(energy_bad | (1U << static_cast<unsigned>(domain))); }
    // Sum of the power domains read (system alone when present); NaN when none.
    [[nodiscard]] double total_power_w() const noexcept;
};

// One sysfs-style number: an optional sign and decimal digits, then only blanks or line ends. Anything else (a
// numeric prefix followed by other bytes, an empty text, more than 18 digits) is not a number.
[[nodiscard]] bool parse_value(std::string_view text, double& value) noexcept;

struct SourceStatus {
    std::string name;      // "amdgpu-hwmon", "rapl", "cpufreq", ...
    bool available = false;
    std::string detail;    // available: what is read (channels, domain); unavailable: the reason
};

// A device or counter. probe() runs once before the first read and must not throw; read() adds what it has to the
// sample (add_power / add_energy with its domain, fail_* when a channel it owns could not be read, temperature takes
// the maximum, the clocks are set when absent) and must be cheap: it runs on the sampler thread every period. Calls
// are serialized by the sampler: a source needs no lock of its own.
class Source {
public:
    virtual ~Source() = default;
    virtual SourceStatus probe() = 0;
    virtual void read(Sample& sample) noexcept = 0;
    // Empty: this source owns disjoint hardware. Equal nonempty IDs in one domain mean the exact same hardware
    // (including every member of an aggregate), so counter and power are alternatives. Use one source per channel
    // in a named group. Stable after probe().
    virtual std::string_view coverage_id(Domain) const noexcept { return {}; }
};

// root is the filesystem prefix of the system tree: empty means the real machine ("/"); a test passes a directory
// holding a fake sys/ tree.
using SourceFactory = std::function<std::unique_ptr<Source>(const std::string& root)>;
// Adds a source kind for every sampler made afterwards. The built-in kinds are registered already.
void register_source(std::string kind, SourceFactory factory);
[[nodiscard]] std::vector<std::unique_ptr<Source>> make_sources(const std::string& root = {});

struct Receipt {
    bool available = false;
    std::string reason;            // when not available
    double duration_ms = 0.0, energy_j = 0.0, avg_w = 0.0, peak_w = 0.0;
    std::uint64_t tokens = 0;
    double tok_per_j = 0.0;        // 0 when tokens or energy is 0, or the receipt is not complete
    double gpu_clock_mhz_median = 0.0, temp_c_max = 0.0, cpu_mhz_median = 0.0, gpu_busy_pct_mean = 0.0;
    std::size_t samples = 0;       // samples inside the window
    std::string energy_source;     // "counter" | "power-integral" | "mixed"
    // complete: every hardware group the sampler measures covers the whole window. Otherwise energy_j is the energy of the
    // covered parts only, avg_w its mean power, and covered_ms the time all of those domains cover at once (the
    // intersection of their covered stretches; 0 when one of them has no reading).
    bool complete = false;
    double covered_ms = 0.0;
    std::string coverage;          // "package:counter+accelerator:power-integral"; ":none" for a domain with no reading
};

struct Options {
    int period_ms = 100;
    std::size_t capacity = 1U << 16;  // ring size in samples
    int max_gap_ms = 0;       // longest stretch between two consecutive samples that still counts as covered; 0: max(1000, 3 periods)
    int min_interval_ms = 0;  // shortest spacing of worker samples when cover() asks for one; 0: period / 10 within 1..5
};

class Sampler {
public:
    explicit Sampler(std::vector<std::unique_ptr<Source>> sources, Options options = {});
    ~Sampler();
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    // True when at least one source is available.
    [[nodiscard]] bool available() const noexcept { return usable_; }
    [[nodiscard]] const std::vector<SourceStatus>& sources() const noexcept { return status_; }

    void start();  // worker thread; no-op when nothing is available or already started
    void stop();
    // Reads every source on the calling thread and stores the sample (tests pass the timestamp). Serialized with the
    // worker; a timestamp that is not newer than the last sample is refused before any source is read.
    void sample_now();
    void sample_at(std::int64_t t_ns);
    // Makes the ring reach t_ns: asks the worker for one sample (never faster than min_interval_ms) and waits at most
    // timeout_ms for it. True when a sample at or after t_ns is stored. Without a worker it only reports.
    bool cover(std::int64_t t_ns, int timeout_ms);
    [[nodiscard]] Sample latest() const;
    [[nodiscard]] std::size_t sample_count() const;
    // True once any domain delivered an energy counter or a power reading.
    [[nodiscard]] bool energy_available() const;
    // Every stretch measured since the first sample (not limited by the ring): duration_ms is the span first..last
    // sample, covered_ms / complete as in a window receipt.
    [[nodiscard]] Receipt lifetime_receipt() const;
    void lifetime(double& joules, double& seconds) const;  // energy and covered time of lifetime_receipt()
    [[nodiscard]] Receipt summarize(std::int64_t t0_ns, std::int64_t t1_ns, std::uint64_t tokens) const;
    [[nodiscard]] int period_ms() const noexcept { return options_.period_ms; }
    [[nodiscard]] int min_interval_ms() const noexcept { return options_.min_interval_ms; }

    [[nodiscard]] static std::int64_t now_ns() noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    [[nodiscard]] static std::int64_t to_ns(std::chrono::steady_clock::time_point t) noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
    }

    // One channel of one hardware group: joules and nanoseconds over its covered stretches and the steepest reading.
    struct Cover { double joules = 0.0; std::int64_t ns = 0; double peak = 0.0; };
    using Covers = std::vector<std::array<Cover, 2>>;  // per hardware group: counter, power
    // Only observed combinations are stored; each group has counter bit 1 and power bit 2.
    // Ordered by length, then bytewise over that explicit length. The default vector ordering lowers to a memcmp whose
    // bound comes from a pointer difference, which GCC 13 at -O3 cannot bound (-Wstringop-overread).
    struct MaskLess {
        bool operator()(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b) const noexcept {
            if (a.size() != b.size()) return a.size() < b.size();
            for (std::size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return a[i] < b[i];
            return false;
        }
    };
    using Joint = std::map<std::vector<unsigned char>, std::int64_t, MaskLess>;

private:
    void run();
    void sample_locked(std::int64_t t_ns);  // read_mutex_ held
    [[nodiscard]] std::vector<Sample> window_locked(std::int64_t t0_ns, std::int64_t t1_ns, std::size_t& inside, bool& lead) const;

    std::vector<std::unique_ptr<Source>> sources_;
    std::vector<SourceStatus> status_;
    std::vector<Source*> live_;
    std::vector<std::array<std::size_t, kDomains>> groups_;
    std::vector<Domain> domains_;
    Options options_;
    bool usable_ = false;
    mutable std::mutex mutex_;       // ring, lifetime
    std::mutex read_mutex_;          // one reader of the sources at a time
    std::vector<Sample> ring_;
    std::size_t head_ = 0, size_ = 0;
    Covers life_{};
    Joint life_joint_{};
    std::int64_t first_ns_ = 0, last_ns_ = 0;
    std::uint8_t seen_energy_ = 0, seen_power_ = 0;  // domains that delivered a reading
    std::vector<unsigned char> expected_;  // hardware groups a source read or failed to read
    std::thread thread_;
    std::mutex wake_mutex_;
    std::condition_variable wake_, sampled_;
    std::int64_t want_ns_ = 0, stored_ns_ = 0;  // wake_mutex_: newest timestamp asked for / stored by the worker
    bool stopping_ = false, started_ = false;
};

// ---- the shared receipt block -------------------------------------------------------------------------------------

struct Block {
    bool enabled = false;      // false: STRIX_POWER_SAMPLER=0
    std::vector<SourceStatus> sources;
    Receipt request, prefill, decode;
};

// The process sampler, or null when it is off, not started, or no source is available.
[[nodiscard]] Sampler* process_sampler() noexcept;
// Idempotent. Honors STRIX_POWER_SAMPLER / STRIX_POWER_SAMPLE_MS. Never throws.
void start_process_sampler() noexcept;
void stop_process_sampler() noexcept;
[[nodiscard]] bool sampler_enabled() noexcept;
// Source list of the process sampler (probed even when nothing is available, so a receipt can say why).
[[nodiscard]] std::vector<SourceStatus> process_sources();

// First token of the current request (set once until reset_first_token()); the shared greedy loop calls it.
void mark_first_token() noexcept;
void reset_first_token() noexcept;
[[nodiscard]] std::int64_t first_token_ns() noexcept;  // 0 when not marked

// The receipt of the windows [p0, d0] (prefill, absent when d0 <= p0), [d0, end] (decode, absent when end <= d0) and
// [p0, end] (request) of one sampler. Reads the ring only; call Sampler::cover(end) first for a window that just ended.
[[nodiscard]] Block window_block(const Sampler& sampler, std::int64_t p0_ns, std::int64_t d0_ns, std::int64_t end_ns,
                                 std::uint64_t prefill_tokens, std::uint64_t decode_tokens);
// Block of a request the process executed over [begin, end] (the execution timestamps, never the time a response is
// rendered): decode is its last decode_ms, prefill everything before. Time inside the span in which the machine ran
// other requests belongs to the window: the receipt is machine energy over the span. wait: ask the worker for one
// closing sample, for at most kClosingWaitMs; when the deadline expires, or without wait (a stream, whose terminal
// chunk must not be held back), the receipt is what the ring holds and says the tail is not covered. Build it once
// per request and reuse the result.
inline constexpr int kClosingWaitMs = 10;
[[nodiscard]] Block execution_block(std::int64_t begin_ns, std::int64_t end_ns, double decode_ms,
                                    std::uint64_t prefill_tokens, std::uint64_t decode_tokens, bool wait = true);
// Block for a request [start, stop]: split at the first-token mark when it lies inside (prefill includes the first
// token, as the request timing does; decode is the remaining generated - 1 tokens), otherwise one prefill phase.
[[nodiscard]] Block request_block(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point stop,
                                  std::uint64_t prompt_tokens, std::uint64_t generated_tokens);
// {"schema":"strix-power-receipt-v1",...}; one line, no trailing newline.
[[nodiscard]] std::string block_json(const Block& block);
// "power":{...} for a JSON object a CLI prints, or empty when the sampler is switched off: the member is then absent
// and the output has the shape it had before the receipt existed.
[[nodiscard]] std::string block_member(const Block& block);
// Process view for a status page: sources, the latest stored reading and the totals since the sampler started. Reads
// no sensor. A channel without a reading is null, never 0; energy_available tells a clock-only host apart.
[[nodiscard]] std::string status_json();
[[nodiscard]] std::string status_json(const Sampler* sampler, bool enabled, const std::vector<SourceStatus>& sources);

}  // namespace strix::power
