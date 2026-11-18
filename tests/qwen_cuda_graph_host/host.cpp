#include "strix/cuda/core/decode_rows.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>

// No assert/abort: failures are ordinary process exits, including on Windows.
constexpr unsigned kMaxStepRows = 8, kMaxSegments = 8, kScoreRowsPerWave = 4;
constexpr unsigned kMaxSelected = 2048, kRun = 128;
constexpr unsigned kTreeRows = 64, kTreeSeg = 256;  // tree scorer tiles (qwen_qsa.cu)
struct dim3 {
    unsigned x = 0, y = 0, z = 0;
    dim3() = default;
    dim3(unsigned a, unsigned b = 1U, unsigned c = 1U) : x(a), y(b), z(c) {}
};
constexpr int cudaSuccess = 0, cudaStreamCaptureModeRelaxed = 0, cudaMemcpyHostToDevice = 0;
constexpr int cudaDevAttrMultiProcessorCount = 0;
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone, active, invalidated };
using cudaGraph_t = void*;
cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
int fault = 0, calls = 0, graphs = 0, ended = 0;
int step() {
    if (++calls != fault) return cudaSuccess;
    if (status == active) status = invalidated;
    return 1;
}
void ck(int code, const char* message) { if (code != cudaSuccess) throw std::runtime_error(message); }
int cudaStreamBeginCapture(int, int) {
    const int code = step();
    if (code == cudaSuccess) status = active;
    return code;
}
int cudaMemcpyAsync(void*, const void*, std::size_t, int, int) { return step(); }
int cudaStreamIsCapturing(int, cudaStreamCaptureStatus* out) { *out = status; return cudaSuccess; }
int cudaStreamEndCapture(int, cudaGraph_t* out) {
    ++ended;
    status = cudaStreamCaptureStatusNone;
    *out = &graphs;
    ++graphs;
    return cudaSuccess;
}
void cudaGraphDestroy(cudaGraph_t) { --graphs; }
int cudaGetDevice(int* dev) { *dev = 0; return cudaSuccess; }
int cudaDeviceGetAttribute(int* count, int, int) { *count = 80; return cudaSuccess; }
bool decode_rows_enabled() { return true; }
namespace core = strix::cuda::core;  // production calls core::decode_rows_max() (the real host-safe header)
struct Workspace { unsigned max_blocks = 65536, chunk_rows = 8; };
struct MultiShape {
    unsigned rows = 0, pool_grid = 0, score_grid = 0, slot_runs = 0;
    dim3 tree_grid{0U, 0U, 0U};
    bool head_all = false;
    bool tree_aligned = true;
};
namespace qsa {
struct LayerState { void *kv, *index, *pooled; std::uint32_t* valid; };
struct SegmentState { unsigned rows; std::uint64_t pos; LayerState state; };
struct SegmentMap { unsigned dummy; };
void prepare_replay(const SegmentState* state, unsigned, const Workspace&, SegmentMap&) {
    ++*state->state.valid;
    ck(step(), "prepare");
}
}
struct Session {
    std::uint32_t mtp_pooled_valid = 0;
    std::uint64_t length = 2052;
    void *mtp_kv = nullptr, *mtp_index = nullptr, *mtp_pooled = nullptr;
} S;
struct Driver {
    std::uint64_t draft_pos_pinned[kMaxStepRows + 1]{}, draft_pos[kMaxStepRows]{};
    qsa::SegmentMap qsa_maps_pinned[20]{}, qsa_maps[20]{};
    unsigned token = 0;
    unsigned* draft_token = &token;
    Workspace qws;
    int stream = 0;
} m;
#include "production.inc"

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void capture_test() {
    // Two descriptors, begin, three copies, first device operation. Replay also prepares descriptors.
    for (const bool replay : {false, true}) {
        for (int fail = 1; fail <= (replay ? 3 : 7); ++fail) {
            S.mtp_pooled_valid = 0;
            fault = fail; calls = 0; ended = 0;
            bool threw = false;
            try { transaction(!replay, replay); } catch (const std::runtime_error&) { threw = true; }
            require(threw, "fault did not propagate");
            require(S.mtp_pooled_valid == 0, "failed setup left pooled validity advanced");
            require(status == cudaStreamCaptureStatusNone, "capture left active/invalidated");
            require(graphs == 0, "cleanup leaked graph");
            require(ended == ((!replay && fail >= 4) ? 1 : 0), "incorrect capture cleanup");
        }
    }
    calls = 0; fault = 0;
    transaction(true, false);
    require(S.mtp_pooled_valid == 2 && graphs == 0 && status == cudaStreamCaptureStatusNone,
            "successful setup did not retain valid metadata");
}
void pool_test() {
    const std::uint32_t rows[kMaxSegments]{1};
    MultiShape shape;
    require(graph_shape(rows, 1, m.qws, shape), "one-row graph declined");
    require(shape.pool_grid >= 80 && shape.pool_grid <= 160, "cold pool serialized instead of sized for device");
    for (const unsigned blocks : {0U, 1U, 513U, 65536U}) {
        unsigned visits = 0, busy = 0, longest = 0;
        for (unsigned cta = 0; cta < shape.pool_grid; ++cta) {
            unsigned iterations = 0;
            for (unsigned block = cta; block < blocks; block += shape.pool_grid) { ++visits; ++iterations; }
            busy += iterations != 0;
            longest = std::max(longest, iterations);
        }
        require(visits == blocks, "pool grid misses or duplicates work");
        require(busy == std::min(blocks, shape.pool_grid), "empty CTA handling");
        require(longest == (blocks + shape.pool_grid - 1) / shape.pool_grid, "unbalanced cold fill");
    }
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected case");
        if (std::string_view(argv[1]) == "capture") capture_test(); else pool_test();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
