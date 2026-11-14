#pragma once

#include "strix/serve/prompt_scorer.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Aux lane: one queue for the side models of a serving process (prompt scoring, image encoding, embeddings).
//
// A side model is small and compute-bound. Where it runs is a per-job choice between the units the device
// declaration grants the `aux` role: the GPU, an NPU, or the CPU reference. The lane owns that choice, the
// fallback order, the bound on waiting work and the accounting; an executor only turns one job into one result.
//
// Unit choice (R-aux). A side job of GPU duration D_g costs a busy GPU about `gpu_side_cost` x D_g. On an NPU it
// lasts D_n and costs the GPU `npu_side_cost` x D_n. So the NPU goes first only while the GPU is busy and
// D_n / D_g < gpu_side_cost / npu_side_cost, with both durations measured. On an idle GPU, or without measured
// durations, the GPU goes first. The CPU reference is the last fallback unless the order says otherwise.
//
// Host-only code, no device types. With only a GPU executor registered, `run_now` is the direct call it replaces:
// same thread, same arguments, same result.
namespace strix::serve::aux {

enum class JobKind : std::uint8_t { score, vision, embed };
enum class UnitKind : std::uint8_t { gpu, npu, cpu };
inline constexpr std::size_t kJobKinds = 3;
[[nodiscard]] std::string_view kind_name(JobKind kind) noexcept;
[[nodiscard]] std::string_view unit_name(UnitKind unit) noexcept;
[[nodiscard]] UnitKind parse_unit(std::string_view text); // gpu | npu | cpu; throws std::invalid_argument

using Clock = std::chrono::steady_clock;

struct AuxJob {
    JobKind kind = JobKind::score;
    // Borrowed: the memory must stay valid until the result (or the error) is delivered.
    std::vector<std::span<const std::byte>> inputs;
    std::vector<std::uint32_t> shape; // meaning set by the kind (vision: height, width, channels)
    std::uint64_t session = 0;        // conversation whose executor state may be reused; 0 = none
    // Arithmetic identity the result must carry. Empty accepts any registered executor of the kind; a request
    // whose outputs are cached or compared across units names the identity it needs.
    std::string identity;
    // A job that has not started by then fails with Code::deadline instead of running late.
    Clock::time_point deadline = Clock::time_point::max();
};

struct AuxResult {
    std::vector<float> tensor;
    std::vector<std::uint32_t> shape;
    std::vector<std::uint32_t> integers; // integer side channel of the kind (score: token byte offsets)
    // Filled by the lane.
    std::string identity;
    UnitKind unit = UnitKind::gpu;
    std::string unit_id;
    std::uint32_t fallbacks = 0; // executors that failed before this one answered
    double queued_ms = 0.0;
    double run_ms = 0.0;
};

struct AuxError : std::runtime_error {
    enum class Code { queue_full, no_executor, deadline, failed, stopped };
    AuxError(Code c, const std::string& what) : std::runtime_error(what), code(c) {}
    Code code;
};

// One way to run one kind of job on one unit.
struct Executor {
    UnitKind unit = UnitKind::gpu;
    std::string unit_id; // declared device id
    JobKind kind = JobKind::score;
    std::string identity; // arithmetic identity of its results
    // Fills tensor / shape / integers. A throw makes the lane try the next unit of the order.
    std::function<AuxResult(const AuxJob&)> run;
    // true: the lane owns one worker thread per unit id and runs the job there.
    // false: the unit belongs to another thread (a GPU context and its queue). Jobs run inside `drain` or
    //        `run_now`, on the thread that calls them.
    bool threaded = true;
};

struct Unit {
    UnitKind kind = UnitKind::gpu;
    std::string id;
};

// Measured duration of one job of a kind, per unit; 0 = not measured.
struct Rates {
    double gpu_seconds = 0.0;
    double npu_seconds = 0.0;
};

struct Config {
    // Units that may take aux work (from the device declaration; see aux_units.hpp). An executor for any other
    // unit is refused at registration: an undeclared unit gets no work.
    std::vector<Unit> units;
    std::size_t capacity = 64; // waiting jobs; a submit beyond it fails with Code::queue_full
    // Fallback order per job kind, indexed by JobKind.
    std::array<std::array<UnitKind, 3>, kJobKinds> order{{{UnitKind::gpu, UnitKind::npu, UnitKind::cpu},
                                                         {UnitKind::gpu, UnitKind::npu, UnitKind::cpu},
                                                         {UnitKind::gpu, UnitKind::npu, UnitKind::cpu}}};
    std::array<Rates, kJobKinds> rates{};
    double gpu_side_cost = 0.85; // share of a side job's GPU duration a busy GPU loses
    double npu_side_cost = 0.10; // share of an NPU job's duration the GPU loses while it runs
    // Whether the GPU has main work now (a session in prefill or decode). Empty = never busy.
    std::function<bool()> gpu_busy;
    // Operator override: only this unit is used (no fallback to another kind of unit).
    std::optional<UnitKind> force;
};

// Applies STRIX_AUX_UNIT = auto | gpu | npu | cpu to `base`. Throws std::invalid_argument for another value.
[[nodiscard]] Config with_environment(Config base);

// Pure decision: indices into `executors` in the order a job should try them. Empty = nothing can run it.
[[nodiscard]] std::vector<std::size_t> plan(const Config& config, std::span<const Executor> executors, JobKind kind,
                                            std::string_view identity, bool gpu_busy);

struct UnitCount {
    UnitKind unit = UnitKind::gpu;
    std::string unit_id;
    JobKind kind = JobKind::score;
    std::string identity;
    std::uint64_t jobs = 0;
    std::uint64_t failed = 0;
    double busy_ms = 0.0;
};

struct Counters {
    std::uint64_t submitted = 0;
    std::uint64_t completed = 0;
    std::uint64_t queue_full = 0;
    std::uint64_t no_executor = 0;
    std::uint64_t deadline = 0;
    std::uint64_t failed = 0;    // every executor of the order failed
    std::uint64_t fallbacks = 0; // executor failures that another unit absorbed
    std::size_t waiting_peak = 0;
    std::vector<UnitCount> units; // one per registered executor, registration order
};

class AuxLane {
public:
    explicit AuxLane(Config config);
    ~AuxLane(); // stops the workers; waiting jobs fail with Code::stopped
    AuxLane(const AuxLane&) = delete;
    AuxLane& operator=(const AuxLane&) = delete;

    // Registration is not concurrent with submit. Throws std::invalid_argument for an executor without a function
    // or on a unit the configuration does not list.
    void add(Executor executor);

    // Queues the job. Throws AuxError (queue_full, no_executor, stopped) without side effects.
    [[nodiscard]] std::future<AuxResult> submit(AuxJob job);
    // Runs waiting jobs whose next executor is a non-threaded executor of `unit_id`, on the calling thread.
    // Returns the number of jobs taken. The owner of that unit calls it between its own steps.
    std::size_t drain(std::string_view unit_id, std::size_t max_jobs = static_cast<std::size_t>(-1));
    // Runs the job to completion from the calling thread: a non-threaded executor runs here (the caller owns that
    // unit), a threaded one runs on its worker while this call waits. Throws AuxError.
    [[nodiscard]] AuxResult run_now(const AuxJob& job);

    [[nodiscard]] Counters counters() const;
    // {"submitted":3,"completed":3,...,"units":[{"unit":"gpu","id":"main","kind":"score","identity":"...",...}]}
    [[nodiscard]] std::string receipt_json() const;
    [[nodiscard]] bool has(JobKind kind) const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

// ---- Prompt scoring client ---------------------------------------------------------------------------------
// tensor = [3][tokens] (token_score, cut_prob, request_mask as 0..255), integers = token_byte_begin. Exact both ways.
[[nodiscard]] AuxResult pack_scores(const pflash::ScoreBuffers& scores);
[[nodiscard]] pflash::ScoreBuffers unpack_scores(const AuxResult& result); // throws std::invalid_argument
// A score job borrows `rendered`.
[[nodiscard]] AuxJob score_job(std::string_view rendered, std::uint64_t session, std::string identity = {});
[[nodiscard]] Executor score_executor(UnitKind unit, std::string unit_id, std::string identity,
                                      std::shared_ptr<pflash::PromptScorer> scorer, bool threaded);
// The scorer the selection layer holds: every call goes through the lane.
class LaneScorer final : public pflash::PromptScorer {
public:
    explicit LaneScorer(AuxLane& lane, std::string identity = {}) : lane_(lane), identity_(std::move(identity)) {}
    pflash::ScoreBuffers score(std::string_view rendered, std::uint64_t session) override;
    [[nodiscard]] const AuxResult& last() const noexcept { return last_; } // unit and timing of the last call; tensors moved out

private:
    AuxLane& lane_;
    std::string identity_;
    AuxResult last_;
};

// ---- Vision client -----------------------------------------------------------------------------------------
struct ImageView {
    std::uint32_t height = 0;
    std::uint32_t width = 0;
    std::uint32_t channels = 0;
    std::span<const std::uint8_t> pixels; // [height][width][channels]
};
struct ImageRows {
    std::vector<float> rows; // [count][hidden]
    std::uint32_t count = 0;
    std::uint32_t hidden = 0;
    std::string identity;
    UnitKind unit = UnitKind::gpu;
    std::string unit_id;
};
// Returns [count][hidden] rows and sets the two dimensions.
using ImageEncoder = std::function<std::vector<float>(const ImageView& image, std::uint32_t& count, std::uint32_t& hidden)>;
[[nodiscard]] AuxJob vision_job(const ImageView& image, std::string identity = {}); // borrows the pixels
[[nodiscard]] Executor vision_executor(UnitKind unit, std::string unit_id, std::string identity, ImageEncoder encoder,
                                       bool threaded);
[[nodiscard]] ImageRows image_rows(AuxResult result); // throws std::invalid_argument on a malformed result
[[nodiscard]] ImageRows encode_image(AuxLane& lane, const ImageView& image, std::string identity = {});

} // namespace strix::serve::aux
