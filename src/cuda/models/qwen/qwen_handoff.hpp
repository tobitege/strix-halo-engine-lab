// Streaming hand-off of a Qwen prompt state from the CUDA prefill service to a node's HIP server
// (`--rtx-state-import`, routes /admin/rtx-import/{layout,begin,chunks,commit,abort}; format strix-qwen-rtx-state-v1).
// The GPU thread enqueues finished pieces (row ranges of a row component, or byte ranges of the fixed blob) as soon as
// they are final. A packer thread packs them into multi-segment chunk bodies (optionally bp-v1 encoded on a small
// worker pool, Options::codec) with their SHA-256 (X-Chunk-Sha256, X-Chunk-Seq), while poster threads send the
// previous chunks on keep-alive connections (Options::connections). commit sends the chunk count, the segment byte
// count and the SHA-256 of the chunk digests in sequence order. Nothing here touches the device: the caller hands over
// host memory that stays valid until commit().
#pragma once

#include "strix/bp_codec.hpp"
#include "strix/qwen_prefill_trace.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace strix::qwen_handoff {

struct RowComponent { std::uint32_t id = 0; std::string name; std::uint64_t bytes_per_row = 0; std::uint32_t shift = 0; };
struct FixedComponent { std::uint32_t id = 0; std::string name; std::uint64_t bytes = 0, offset = 0; };
struct Layout {
    std::string format;
    std::vector<RowComponent> rows;
    std::vector<FixedComponent> fixed;
    std::uint64_t row_bytes = 0, fixed_bytes = 0, layout_identity = 0, max_context = 0;
    std::vector<std::string> codecs;  // chunk encodings the node decodes (layout `codecs`; empty on older nodes)
    std::string raw_json;
};

struct Piece {
    bool fixed = false;
    std::uint32_t component = 0;
    std::uint64_t first = 0;   // first row (row component) or byte offset (fixed)
    std::uint64_t count = 0;   // rows (row component) or 0 for fixed
    const std::uint8_t* data = nullptr;
    std::uint64_t bytes = 0;
    std::uint64_t trace_id = 0;
};

struct Options {
    // "" or "identity": plain chunks (exactly the protocol without a codec). "bp-v1": encoded chunks
    // (X-Chunk-Encoding: bp-v1) when the node lists the codec, else plain unless require_codec.
    std::string codec;
    bool require_codec = false;
    unsigned encode_threads = 4;  // codec workers besides the packer thread
    unsigned connections = 1;     // chunk POSTs in flight, one keep-alive connection each
    std::uint64_t chunk_bytes = std::uint64_t{32} << 20;  // target plain body size of one chunks POST
    qwen_prefill_trace::Trace* trace = nullptr; // must outlive sender
};

struct Timing {
    double layout_s = 0, begin_s = 0, send_s = 0, commit_s = 0;
    double encode_s = 0;  // packer time spent packing / encoding and hashing chunk bodies
    std::uint64_t chunks = 0, bytes = 0;  // bytes: segment data bytes (as commit reports them)
    std::uint64_t wire_bytes = 0;         // chunk bodies on the wire (encoded frames, or plain bodies)
    std::string encoding = "identity";
};

// Element width of a component for the byte-plane transform: 4 for the FP32 recurrent matrices and the host words,
// 2 for the BF16 rest (KV, index, conv histories, MTP rows). Only the ratio depends on it.
[[nodiscard]] std::uint32_t element_width(const std::string& component);
// The wire body of one chunk: the plain multi-segment body (encoded == false), or its bp-v1 frame (each segment header
// a width-1 region, each piece's data a region of widths[i]).
[[nodiscard]] std::vector<std::uint8_t> build_chunk_body(std::span<const Piece> pieces, std::span<const std::uint32_t> widths,
    bool encoded, bp_codec::WorkerPool* pool);

class Sender {
public:
    // import_url: "http://host:port/admin/rtx-import" (the routes' prefix on the node).
    Sender(const std::string& import_url, std::string api_key, int timeout_s, Options options);
    Sender(const std::string& import_url, std::string api_key, int timeout_s, std::uint64_t chunk_bytes = std::uint64_t{32} << 20);
    ~Sender();
    Sender(const Sender&) = delete;
    Sender& operator=(const Sender&) = delete;
    // Component index by name (row or fixed); throws when the node does not have it.
    [[nodiscard]] std::uint32_t row_id(const std::string& name);
    [[nodiscard]] std::uint32_t fixed_id(const std::string& name);

    [[nodiscard]] const Layout& layout();                    // GET .../layout (cached)
    // POST .../begin; producer ids name the CUDA arithmetic (nonzero). Negotiates the chunk encoding.
    // local_rows q > 0 (cooperative prefill): the node takes rows [0, q) from its own checkpoint of prompt[0, q);
    // enqueue then drops every row below q - shift (the fixed state is always sent).
    void begin(const std::vector<std::uint32_t>& tokens, std::uint64_t position, std::uint64_t numerical_id,
               std::uint64_t reduction_id, std::uint64_t state_update_id, std::uint64_t local_rows = 0);
    void enqueue(const Piece& piece);                         // thread-safe, non-blocking
    // Waits until every enqueued piece is sent, then POSTs commit; returns the node's reply JSON. Throws on any error.
    std::string commit();
    void abort() noexcept;
    [[nodiscard]] const Timing& timing() const noexcept { return timing_; }

private:
    struct Impl;
    void pack();
    void post(unsigned connection);
    Impl* impl_ = nullptr;
    Timing timing_;
};

}  // namespace strix::qwen_handoff
