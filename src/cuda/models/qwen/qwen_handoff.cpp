#include "qwen_handoff.hpp"

#include "qwen_http.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/recurrent_state_policy.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace strix::qwen_handoff {
namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

std::string hex(const ModelContentIdentity& d) {
    static const char* k = "0123456789abcdef";
    std::string s;
    for (const auto b : d.bytes) { s.push_back(k[b >> 4]); s.push_back(k[b & 15]); }
    return s;
}

// u64 fields may arrive as JSON numbers, decimal strings or "0x" hex strings (identities exceed 2^53).
std::uint64_t u64_of(const serve::Json* j, const char* what) {
    if (!j) throw std::runtime_error(std::string("rtx-import layout lacks ") + what);
    const std::string& t = j->is_string() ? j->as_string() : j->number_source();
    return t.rfind("0x", 0) == 0 ? std::stoull(t.substr(2), nullptr, 16) : std::stoull(t);
}

void put32(std::uint8_t* out, std::uint32_t v) { for (int i = 0; i < 4; ++i) out[i] = static_cast<std::uint8_t>(v >> (8 * i)); }
void put64(std::uint8_t* out, std::uint64_t v) { for (int i = 0; i < 8; ++i) out[i] = static_cast<std::uint8_t>(v >> (8 * i)); }

// The 24-byte segment header of a piece (docs/rtx-state-import.md).
void segment_header(const Piece& p, std::uint8_t* out) {
    put32(out, p.fixed ? 1U : 0U);
    put32(out + 4, p.component);
    put32(out + 8, static_cast<std::uint32_t>(p.fixed ? (p.first & 0xffffffffULL) : p.first));
    put32(out + 12, static_cast<std::uint32_t>(p.fixed ? (p.first >> 32) : p.count));
    put64(out + 16, p.bytes);
}

bool is_identity(const std::string& codec) { return codec.empty() || codec == "identity"; }

}  // namespace

std::uint32_t element_width(const std::string& component) {
    if (component == "recurrent_matrix")  // the stored element of the active recurrent-state contract
        return static_cast<std::uint32_t>(recurrent_state_element_bytes(recurrent_state_policy().type));
    if (component == "ngram_history" || component == "position_history") return 4;
    return 2;
}

std::vector<std::uint8_t> build_chunk_body(std::span<const Piece> pieces, std::span<const std::uint32_t> widths, bool encoded,
    bp_codec::WorkerPool* pool) {
    if (!encoded) {
        std::uint64_t total = 0;
        for (const auto& p : pieces) total += 24 + p.bytes;
        std::vector<std::uint8_t> body(static_cast<std::size_t>(total));
        std::uint8_t* at = body.data();
        for (const auto& p : pieces) {
            segment_header(p, at);
            if (p.bytes) std::memcpy(at + 24, p.data, static_cast<std::size_t>(p.bytes));
            at += 24 + p.bytes;
        }
        return body;
    }
    std::vector<std::uint8_t> headers(24 * pieces.size());
    std::vector<bp_codec::Region> regions;
    regions.reserve(2 * pieces.size());
    for (std::size_t i = 0; i < pieces.size(); ++i) {
        segment_header(pieces[i], headers.data() + 24 * i);
        regions.push_back({headers.data() + 24 * i, 24, 1});
        regions.push_back({pieces[i].data, static_cast<std::size_t>(pieces[i].bytes), i < widths.size() ? widths[i] : 1U});
    }
    return bp_codec::encode(regions, pool);
}

struct Sender::Impl {
    struct Wire {
        std::vector<std::uint8_t> body;
        std::string digest_hex;
        std::uint64_t seq = 0, pieces = 0, data_bytes = 0;
    };
    std::string origin, base_path;  // e.g. "http://node:8080", "/admin/rtx-import"
    int timeout_s;
    Options opts;
    qwen_http::Client client;  // control requests (layout, begin, commit, abort)
    qwen_http::Headers auth;
    Layout layout;
    bool have_layout = false;
    std::vector<std::uint32_t> row_width, fixed_width;
    std::string id;
    bool encoded = false;
    std::uint64_t local_rows = 0;  // cooperative: rows [0, local_rows - shift) of each row component stay on the node
    std::unique_ptr<bp_codec::WorkerPool> pool;
    std::mutex m;
    std::condition_variable cv;
    std::deque<Piece> queue;   // GPU thread -> packer
    std::deque<Wire> wires;    // packer -> posters (bounded)
    bool closing = false, packed_all = false, failed = false, started = false;
    std::string error;
    std::uint64_t in_flight = 0;  // pieces enqueued and not yet posted
    std::uint64_t seq = 0;
    Clock::time_point first;
    Sha256Fast manifest;
    std::thread packer;
    std::vector<std::thread> posters;
    Impl(std::string o, std::string path, const std::string& key, int timeout, Options options)
        : origin(std::move(o)), base_path(std::move(path)), timeout_s(timeout), opts(std::move(options)), client(origin, timeout) {
        if (!key.empty()) auth.emplace_back("Authorization", "Bearer " + key);
        opts.connections = std::max(1U, opts.connections);
        if (!opts.chunk_bytes) opts.chunk_bytes = std::uint64_t{32} << 20;
    }
    void fail(const std::string& what) {
        const std::lock_guard<std::mutex> lock(m);
        if (!failed) { failed = true; error = what; }
        cv.notify_all();
    }
};

namespace {
std::pair<std::string, std::string> split_url(const std::string& url) {
    const auto u = qwen_http::parse_url(url);
    std::string path = u.path;
    while (path.size() > 1 && path.back() == '/') path.pop_back();
    return {"http://" + u.host + ":" + u.port, path == "/" ? std::string("/admin/rtx-import") : path};
}
}  // namespace

Sender::Sender(const std::string& import_url, std::string api_key, int timeout_s, Options options)
    : impl_(new Impl(split_url(import_url).first, split_url(import_url).second, api_key, timeout_s, std::move(options))) {
    impl_->packer = std::thread([this] { pack(); });
    for (unsigned c = 0; c < impl_->opts.connections; ++c) impl_->posters.emplace_back([this, c] { post(c); });
}

Sender::Sender(const std::string& import_url, std::string api_key, int timeout_s, std::uint64_t chunk_bytes)
    : Sender(import_url, std::move(api_key), timeout_s, Options{"", false, 4, 1, chunk_bytes}) {}

Sender::~Sender() {
    {
        std::lock_guard<std::mutex> lock(impl_->m);
        impl_->closing = true;
    }
    impl_->cv.notify_all();
    if (impl_->packer.joinable()) impl_->packer.join();
    for (auto& t : impl_->posters) if (t.joinable()) t.join();
    delete impl_;
}

const Layout& Sender::layout() {
    // v1 echoes the consumer identity; it does not negotiate the producer codec.
    require_f32_recurrent_state(recurrent_state_policy().type, "node handoff (no producer state contract)");
    if (impl_->have_layout) return impl_->layout;
    const auto t0 = Clock::now();
    auto [status, body] = impl_->client.get(impl_->base_path + "/layout", impl_->auth);
    if (status != 200) throw std::runtime_error("rtx-import layout: HTTP " + std::to_string(status) + " " + body);
    const auto j = serve::parse_json(body);
    Layout& l = impl_->layout;
    l.raw_json = body;
    if (const auto* f = j.find("format")) l.format = f->as_string();
    if (l.format != "strix-qwen-rtx-state-v1") throw std::runtime_error("rtx-import layout: unexpected format " + l.format);
    for (const auto& r : j.find("rows")->as_array())
        l.rows.push_back({static_cast<std::uint32_t>(l.rows.size()), r.find("name") ? r.find("name")->as_string() : std::string(),
                          u64_of(r.find("bytes_per_row"), "rows.bytes_per_row"), static_cast<std::uint32_t>(u64_of(r.find("shift"), "rows.shift"))});
    for (const auto& f : j.find("fixed")->as_array())
        l.fixed.push_back({static_cast<std::uint32_t>(l.fixed.size()), f.find("name") ? f.find("name")->as_string() : std::string(),
                           u64_of(f.find("bytes"), "fixed.bytes"), u64_of(f.find("offset"), "fixed.offset")});
    l.row_bytes = u64_of(j.find("row_bytes"), "row_bytes");
    l.fixed_bytes = u64_of(j.find("fixed_bytes"), "fixed_bytes");
    l.layout_identity = u64_of(j.find("layout_identity"), "layout_identity");
    if (const auto* c = j.find("max_context")) l.max_context = u64_of(c, "max_context");
    if (const auto* c = j.find("codecs"); c && c->is_array())
        for (const auto& name : c->as_array()) if (name.is_string()) l.codecs.push_back(name.as_string());
    for (const auto& r : l.rows) impl_->row_width.push_back(element_width(r.name));
    for (const auto& f : l.fixed) impl_->fixed_width.push_back(element_width(f.name));
    impl_->have_layout = true;
    timing_.layout_s = secs(t0);
    return l;
}

std::uint32_t Sender::row_id(const std::string& name) {
    for (const auto& r : layout().rows) if (r.name == name) return r.id;
    throw std::runtime_error("node layout has no row component " + name);
}

std::uint32_t Sender::fixed_id(const std::string& name) {
    for (const auto& f : layout().fixed) if (f.name == name) return f.id;
    throw std::runtime_error("node layout has no fixed component " + name);
}

void Sender::begin(const std::vector<std::uint32_t>& tokens, std::uint64_t position, std::uint64_t numerical_id,
                   std::uint64_t reduction_id, std::uint64_t state_update_id, std::uint64_t local_rows) {
    const auto& l = layout();
    const auto t0 = Clock::now();
    const std::string& codec = impl_->opts.codec;
    bool want = false;
    if (!is_identity(codec)) {
        if (codec != bp_codec::codec_name) throw std::runtime_error("rtx-import: unknown hand-off codec " + codec);
        want = std::find(l.codecs.begin(), l.codecs.end(), codec) != l.codecs.end();
        if (!want && impl_->opts.require_codec) throw std::runtime_error("rtx-import: the node does not decode " + codec);
    }
    std::string body = "{\"tokens\":[";
    for (std::size_t i = 0; i < tokens.size(); ++i) { if (i) body += ','; body += std::to_string(tokens[i]); }
    body += "],\"position\":" + std::to_string(position) + ",\"producer\":{\"numerical\":\"" + std::to_string(numerical_id) +
            "\",\"reduction\":\"" + std::to_string(reduction_id) + "\",\"state_update\":\"" + std::to_string(state_update_id) +
            "\"},\"layout_identity\":\"" + std::to_string(l.layout_identity) + "\"";
    if (want) body += ",\"encoding\":\"" + codec + "\"";
    if (local_rows) body += ",\"local_rows\":" + std::to_string(local_rows);
    body += "}";
    auto hdr = impl_->auth;
    hdr.emplace_back("Content-Type", "application/json");
    auto [status, reply] = impl_->client.post(impl_->base_path + "/begin", hdr,
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(body.data()), body.size()));
    if (status != 200) throw std::runtime_error("rtx-import begin: HTTP " + std::to_string(status) + " " + reply);
    const auto j = serve::parse_json(reply);
    const auto* id = j.find("id");
    if (!id) throw std::runtime_error("rtx-import begin: no id");
    if (want) {
        const auto* e = j.find("encoding");
        if (!e || !e->is_string() || e->as_string() != codec) throw std::runtime_error("rtx-import begin: the node did not accept " + codec);
        impl_->pool = std::make_unique<bp_codec::WorkerPool>(impl_->opts.encode_threads);
    }
    {
        const std::lock_guard<std::mutex> lock(impl_->m);
        impl_->id = id->is_string() ? id->as_string() : id->number_source();
        impl_->encoded = want;
        impl_->local_rows = local_rows;
        timing_.encoding = want ? codec : std::string("identity");
    }
    timing_.begin_s = secs(t0);
}

void Sender::enqueue(const Piece& in) {
    // Cooperative: the rows the node prefills itself are dropped (whole pieces, or the leading rows of one).
    Piece piece = in;
    if (!piece.fixed && impl_->local_rows) {
        const auto shift = impl_->layout.rows.at(piece.component).shift;
        const std::uint64_t local = impl_->local_rows > shift ? impl_->local_rows - shift : 0;
        if (piece.first + piece.count <= local) return;
        if (piece.first < local) {
            const auto skip = local - piece.first, row = piece.bytes / piece.count;
            piece.first = local;
            piece.count -= skip;
            piece.data += skip * row;
            piece.bytes -= skip * row;
        }
    }
    // A piece above the chunk size (the 113 MB recurrent matrices) is split (fixed: byte ranges, a multiple of 256;
    // rows: whole rows), so its chunks pipeline across connections and the node decodes one while the next arrives.
    std::vector<Piece> parts;
    const std::uint64_t limit = impl_->opts.chunk_bytes;
    if (piece.bytes > limit && piece.fixed) {
        const std::uint64_t step = std::max<std::uint64_t>(256, limit / 256 * 256);
        for (std::uint64_t at = 0; at < piece.bytes; at += step) {
            Piece p = piece;
            p.first = piece.first + at;
            p.data = piece.data + at;
            p.bytes = std::min(step, piece.bytes - at);
            parts.push_back(p);
        }
    } else if (piece.bytes > limit && piece.count > 1 && piece.bytes % piece.count == 0) {
        const std::uint64_t row = piece.bytes / piece.count, rows = std::max<std::uint64_t>(1, limit / row);
        for (std::uint64_t r = 0; r < piece.count; r += rows) {
            Piece p = piece;
            p.first = piece.first + r;
            p.count = std::min(rows, piece.count - r);
            p.data = piece.data + r * row;
            p.bytes = p.count * row;
            parts.push_back(p);
        }
    } else {
        parts.push_back(piece);
    }
    {
        std::lock_guard<std::mutex> lock(impl_->m);
        if (impl_->failed) throw std::runtime_error("rtx-import sender failed: " + impl_->error);
        for (const auto& p : parts) {
            if (impl_->opts.trace) impl_->opts.trace->mark("enqueue", p.trace_id, {}, p.first, p.count, p.bytes);
            impl_->queue.push_back(p);
        }
        impl_->in_flight += parts.size();
    }
    impl_->cv.notify_all();
}

// Packer: batches pieces into chunk bodies (in enqueue order), encodes and hashes them, assigns the sequence numbers
// and extends the manifest in that order; at most `connections + 1` bodies wait for a poster.
void Sender::pack() {
    Impl& s = *impl_;
    for (;;) {
        std::vector<Piece> batch;
        std::uint64_t bytes = 0;
        {
            std::unique_lock<std::mutex> lock(s.m);
            s.cv.wait(lock, [&] { return s.closing || !s.queue.empty(); });
            if (s.queue.empty() && s.closing) break;
            while (!s.queue.empty() && (batch.empty() || bytes + s.queue.front().bytes <= s.opts.chunk_bytes)) {
                bytes += s.queue.front().bytes;
                batch.push_back(s.queue.front());
                s.queue.pop_front();
            }
            if (!s.started) { s.started = true; s.first = Clock::now(); }
        }
        Impl::Wire wire;
        wire.pieces = batch.size();
        wire.data_bytes = bytes;
        bool failed = false;
        {
            const std::lock_guard<std::mutex> lock(s.m);
            failed = s.failed;
        }
        if (!failed) {
            try {
                wire.seq = s.seq++;
                if (s.opts.trace) {
                    for (const auto& p : batch) s.opts.trace->mark("chunk_piece", p.trace_id,
                        std::to_string(wire.seq), p.first, p.count, p.bytes);
                    s.opts.trace->mark("encode_start", wire.seq);
                }
                const auto t0 = Clock::now();
                std::vector<std::uint32_t> widths;
                widths.reserve(batch.size());
                for (const auto& p : batch) {
                    const auto& table = p.fixed ? s.fixed_width : s.row_width;
                    widths.push_back(p.component < table.size() ? table[p.component] : 1U);
                }
                wire.body = build_chunk_body(batch, widths, s.encoded, s.pool.get());
                const auto digest = sha256_bytes_fast(std::as_bytes(std::span<const std::uint8_t>(wire.body)));
                wire.digest_hex = hex(digest);
                s.manifest.update(std::as_bytes(std::span<const std::uint8_t>(digest.bytes)));
                const double took = secs(t0);
                const std::lock_guard<std::mutex> lock(s.m);
                if (s.opts.trace) s.opts.trace->mark("encode_end", wire.seq);
                timing_.encode_s += took;
            } catch (const std::exception& e) {
                s.fail(e.what());
                wire.body.clear();
            }
        }
        {
            std::unique_lock<std::mutex> lock(s.m);
            s.cv.wait(lock, [&] { return s.failed || s.wires.size() <= s.opts.connections; });
            s.wires.push_back(std::move(wire));
        }
        s.cv.notify_all();
    }
    {
        const std::lock_guard<std::mutex> lock(s.m);
        s.packed_all = true;
    }
    s.cv.notify_all();
}

// Poster: sends packed bodies on its own keep-alive connection (chunks may complete out of sequence order; the node
// orders the digests by X-Chunk-Seq).
void Sender::post(unsigned /*connection*/) {
    Impl& s = *impl_;
    std::unique_ptr<qwen_http::Client> client;
    for (;;) {
        Impl::Wire wire;
        bool skip = false;
        std::string id;
        bool encoded = false;
        {
            std::unique_lock<std::mutex> lock(s.m);
            s.cv.wait(lock, [&] { return !s.wires.empty() || s.packed_all; });
            if (s.wires.empty()) break;
            wire = std::move(s.wires.front());
            s.wires.pop_front();
            skip = s.failed;
            id = s.id;
            encoded = s.encoded;
        }
        s.cv.notify_all();  // room for the packer
        if (!skip) {
            try {
                if (!client) client = std::make_unique<qwen_http::Client>(s.origin, s.timeout_s);
                auto hdr = s.auth;
                hdr.emplace_back("Content-Type", "application/octet-stream");
                hdr.emplace_back("X-Chunk-Sha256", wire.digest_hex);
                hdr.emplace_back("X-Chunk-Seq", std::to_string(wire.seq));
                if (encoded) hdr.emplace_back("X-Chunk-Encoding", std::string(bp_codec::codec_name));
                if (s.opts.trace) s.opts.trace->mark("post_start", wire.seq);
                auto [status, reply] = client->post(s.base_path + "/chunks?id=" + id, hdr, wire.body);
                if (s.opts.trace) s.opts.trace->mark("post_ack", wire.seq);
                if (status != 200) throw std::runtime_error("rtx-import chunks: HTTP " + std::to_string(status) + " " + reply);
                const std::lock_guard<std::mutex> lock(s.m);
                ++timing_.chunks;
                timing_.bytes += wire.data_bytes;
                timing_.wire_bytes += wire.body.size();
            } catch (const std::exception& e) {
                s.fail(e.what());
            }
        }
        {
            const std::lock_guard<std::mutex> lock(s.m);
            s.in_flight -= wire.pieces;
            timing_.send_s = std::chrono::duration<double>(Clock::now() - s.first).count();
        }
        s.cv.notify_all();
    }
}

std::string Sender::commit() {
    if (impl_->opts.trace) impl_->opts.trace->mark("commit_wait_start");
    {
        std::unique_lock<std::mutex> lock(impl_->m);
        impl_->cv.wait(lock, [&] { return impl_->in_flight == 0 || impl_->failed; });
        if (impl_->failed) throw std::runtime_error("rtx-import sender failed: " + impl_->error);
    }
    const auto t0 = Clock::now();
    if (impl_->opts.trace) impl_->opts.trace->mark("commit_start");
    const auto manifest = impl_->manifest.finalize();
    const std::string body = "{\"chunks\":" + std::to_string(timing_.chunks) + ",\"bytes\":" + std::to_string(timing_.bytes) +
                             ",\"manifest_sha256\":\"" + hex(manifest) + "\"}";
    auto hdr = impl_->auth;
    hdr.emplace_back("Content-Type", "application/json");
    auto [status, reply] = impl_->client.post(impl_->base_path + "/commit?id=" + impl_->id, hdr,
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(body.data()), body.size()));
    timing_.commit_s = secs(t0);
    if (status != 200) throw std::runtime_error("rtx-import commit: HTTP " + std::to_string(status) + " " + reply);
    if (impl_->opts.trace) impl_->opts.trace->mark("commit_end");
    return reply;
}

void Sender::abort() noexcept {
    try {
        if (!impl_->id.empty()) (void)impl_->client.post(impl_->base_path + "/abort?id=" + impl_->id, impl_->auth, {});
    } catch (...) {
    }
}

}  // namespace strix::qwen_handoff
