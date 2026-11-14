// Host g4-v1 graph: HF Gemma4 text semantics, with the declared scalar/reduction recipe.
#include "target.hpp"
#include "gemma4/exl3/format.hpp"
#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cfenv>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>

namespace strix::gemma4 {
namespace {
class OutputPool {
public:
    explicit OutputPool(std::uint32_t count) {
        for (std::uint32_t i = 0; i < std::clamp(count, 1U, 16U); ++i) workers_.emplace_back([this] {
            std::size_t last = 0;
            for (;;) {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [&] { return stop_ || epoch_ != last; });
                if (stop_) return;
                last = epoch_; lock.unlock();
                try {
                    for (;;) {
                        const auto b = next_.fetch_add(16);
                        if (b >= count_) break;
                        job_(b, std::min(count_, b + 16));
                    }
                } catch (...) {
                    std::lock_guard failure_lock(mutex_);
                    if (!failure_) failure_ = std::current_exception();
                }
                lock.lock();
                if (--remaining_ == 0) done_.notify_one();
            }
        });
    }
    ~OutputPool() {
        { std::lock_guard lock(mutex_); stop_ = true; }
        ready_.notify_all();
        for (auto& worker : workers_) worker.join();
    }
    void run(std::size_t count, std::function<void(std::size_t, std::size_t)> job) {
        std::unique_lock lock(mutex_);
        count_ = count; job_ = std::move(job); next_ = 0; remaining_ = workers_.size(); failure_ = nullptr; ++epoch_;
        ready_.notify_all(); done_.wait(lock, [&] { return remaining_ == 0; });
        if (failure_) std::rethrow_exception(failure_);
    }
private:
    std::mutex mutex_;
    std::condition_variable ready_, done_;
    std::vector<std::thread> workers_;
    std::function<void(std::size_t, std::size_t)> job_;
    std::exception_ptr failure_;
    std::atomic<std::size_t> next_{0};
    std::size_t count_ = 0, remaining_ = 0, epoch_ = 0;
    bool stop_ = false;
};
std::vector<float> vector_weight(LinearView v) {
    std::vector<float> out(v.inputs);
    for (std::uint32_t i = 0; i < v.inputs; ++i) out[i] = weight_at(v, 0, i);
    return out;
}
void norm(const std::vector<float>& x, LinearView w, std::vector<float>& out, std::size_t rows, std::uint32_t width, float eps) {
    const auto scale = vector_weight(w);
    out.resize(rows * width);
    for (std::size_t r = 0; r < rows; ++r) {
        const float sum = lane_dot(x.data() + r * width,x.data() + r * width,width);
        if (!std::isfinite(sum)) throw std::runtime_error("nonfinite g4-v1 norm reduction");
        rms_norm(x.data() + r * width, scale.data(), out.data() + r * width, width, eps);
    }
}
void finite(std::span<const float> values) {
    for (const float x : values) if (!std::isfinite(x)) throw std::runtime_error("nonfinite g4-v1 execution");
}
struct Cache {
    std::uint32_t width = 0;
    bool global = false;
    std::vector<float> k, v;
    std::vector<std::uint64_t> tags;
};
struct Overlay {
    std::vector<float> k, v;
};
void validate_config(const ModelConfig& c) {
    if (!c.hidden || c.hidden > hidden || !c.layers || c.layers > layers || c.layer.size() != c.layers ||
        !c.inter || c.inter > inter || !c.vocab || c.vocab > vocab || !c.heads || c.heads > heads ||
        !c.kv_heads || c.kv_heads > kv_heads || c.heads % c.kv_heads || !c.ple_dim || c.ple_dim > ple_dim ||
        !c.window || c.window > window || !c.max_positions || c.max_positions > max_positions || c.eps != norm_eps)
        throw std::runtime_error("invalid host graph configuration");
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const auto& a = c.layer[l];
        if (!a.head_dim || a.head_dim % 2 || a.head_dim > (a.attention == AttentionKind::global ? 512U : 256U) || a.producer > l)
            throw std::runtime_error("invalid host attention configuration");
        const auto& p = c.layer[a.producer];
        if (p.producer != a.producer || p.attention != a.attention || p.head_dim != a.head_dim)
            throw std::runtime_error("invalid host KV producer alias");
    }
}
bool same_config(const ModelConfig& a, const ModelConfig& b) {
    if (a.hidden != b.hidden || a.layers != b.layers || a.inter != b.inter || a.vocab != b.vocab ||
        a.heads != b.heads || a.kv_heads != b.kv_heads || a.ple_dim != b.ple_dim || a.window != b.window ||
        a.max_positions != b.max_positions || a.eps != b.eps || a.layer.size() != b.layer.size()) return false;
    for (std::size_t l = 0; l < a.layer.size(); ++l)
        if (a.layer[l].attention != b.layer[l].attention || a.layer[l].head_dim != b.layer[l].head_dim || a.layer[l].producer != b.layer[l].producer) return false;
    return true;
}
std::uint32_t host_workers(std::uint32_t requested) {
    const char* setting = std::getenv("STRIX_G4_HOST_PARALLEL");
    return setting && std::string_view(setting) == "0" ? 1U : requested;
}
}
struct HostStream::Impl {
    ModelConfig config;
    std::uint64_t frontier = 0;
    bool busy = false;
    std::vector<Cache> cache;
    struct Txn final : KvTxn {
        std::shared_ptr<Impl> stream;
        std::uint64_t start;
        std::vector<Overlay> overlay;
        std::size_t rows = 0;
        std::size_t reserved = 0;
        bool evaluated = false, published = false;
        explicit Txn(std::shared_ptr<Impl> s) : stream(std::move(s)), start(stream->frontier), overlay(stream->cache.size()) {}
        ~Txn() override { if (!published) stream->busy = false; }
        std::uint64_t frontier() const override { return start; }
        void reserve_rows(std::size_t count) override {
            if (rows || reserved || published || !count || count > stream->config.max_positions - start)
                throw std::runtime_error("invalid host overlay reservation");
            reserved = count;
        }
        void commit(std::size_t count) override {
            if (published || !evaluated || count > rows || stream->frontier != start) throw std::runtime_error("invalid/stale KV commit");
            // All potentially throwing allocation precedes any publication or ring mutation.
            for (auto& c : stream->cache) if (c.global && c.width) {
                const auto length = std::size_t(start + count);
                c.k.resize(length * c.width); c.v.resize(length * c.width); c.tags.resize(length, UINT64_MAX);
            }
            for (std::size_t r = 0; r < count; ++r) {
                const auto p = start + r;
                for (std::size_t l = 0; l < stream->cache.size(); ++l) {
                    auto& c = stream->cache[l];
                    if (!c.width) continue;
                    const auto slot = c.global ? std::size_t(p) : std::size_t(p % stream->config.window);
                    std::copy_n(overlay[l].k.data() + r * c.width, c.width, c.k.data() + slot * c.width);
                    std::copy_n(overlay[l].v.data() + r * c.width, c.width, c.v.data() + slot * c.width);
                    c.tags[slot] = p;
                }
            }
            stream->frontier += count; published = true; stream->busy = false;
        }
        const float* read(std::uint32_t producer, std::uint64_t position, bool value) const {
            const auto& c = stream->cache[producer];
            if (position >= start) {
                if (position - start >= rows) throw std::runtime_error("KV overlay position missing");
                const auto& data = value ? overlay[producer].v : overlay[producer].k;
                return data.data() + std::size_t(position - start) * c.width;
            }
            const auto slot = c.global ? std::size_t(position) : std::size_t(position % stream->config.window);
            if (slot >= c.tags.size() || c.tags[slot] != position) throw std::runtime_error("committed KV tag missing");
            return (value ? c.v : c.k).data() + slot * c.width;
        }
    };
};
HostStream::HostStream(const ModelConfig& config) : impl_(std::make_shared<Impl>()) {
    validate_config(config);
    impl_->config = config;
    impl_->cache.resize(config.layers);
    for (std::uint32_t l = 0; l < config.layers; ++l) if (config.layer[l].producer == l) {
        auto& c = impl_->cache[l];
        c.width = config.kv_heads * config.layer[l].head_dim;
        c.global = config.layer[l].attention == AttentionKind::global;
        if (!c.global) {
            c.k.resize(std::size_t(config.window) * c.width); c.v.resize(c.k.size());
            c.tags.resize(config.window, UINT64_MAX);
        }
    }
}
HostStream::~HostStream() = default;
std::uint64_t HostStream::retained_bytes() const {
    if (impl_->busy) throw std::invalid_argument("cannot size an in-flight host stream");
    std::uint64_t total=sizeof(HostStream)+sizeof(Impl);
    const auto add=[&](std::size_t count,std::size_t width) {
        if(count>(UINT64_MAX-total)/width) throw std::overflow_error("host retained size overflow");
        total+=std::uint64_t(count)*width;
    };
    add(impl_->config.layer.capacity(),sizeof(impl_->config.layer[0]));
    add(impl_->cache.capacity(),sizeof(Cache));
    for(const auto& cache:impl_->cache) {
        add(cache.k.capacity(),sizeof(float));add(cache.v.capacity(),sizeof(float));
        add(cache.tags.capacity(),sizeof(std::uint64_t));
    }
    return total;
}
std::unique_ptr<HostStream> HostStream::snapshot() const {
    if (impl_->busy) throw std::invalid_argument("cannot snapshot an in-flight host stream");
    auto copy = std::make_unique<HostStream>(impl_->config);
    copy->impl_ = std::make_shared<Impl>(*impl_);
    return copy;
}
std::unique_ptr<KvTxn> HostStream::begin() {
    if (impl_->busy) throw std::runtime_error("stream already has an unpublished KV transaction");
    auto txn = std::make_unique<Impl::Txn>(impl_);
    impl_->busy = true;
    return txn;
}
std::uint64_t HostStream::frontier() const { return impl_->frontier; }
KvView HostStream::kv_view(std::uint32_t layer, const KvTxn* transaction) const {
    const auto& c = impl_->cache.at(layer);
    if (!c.width) throw std::runtime_error("layer has no host KV");
    KvView v{c.k.data(), c.v.data(), c.tags.data(), c.width, std::uint32_t(c.global ? c.tags.size() : impl_->config.window), c.global};
    v.frontier = impl_->frontier;
    if (transaction) {
        const auto* tx = dynamic_cast<const Impl::Txn*>(transaction);
        if (!tx || tx->stream != impl_ || tx->published) throw std::invalid_argument("transaction does not belong to this stream");
        v.overlay_k = tx->overlay[layer].k.data(); v.overlay_v = tx->overlay[layer].v.data();
        v.frontier = tx->start; v.candidate_rows = std::uint32_t(tx->rows);
    }
    return v;
}
std::string HostStream::digest() const {
    strix::Sha256Fast h;
    h.update({reinterpret_cast<const std::byte*>(&impl_->frontier), sizeof(impl_->frontier)});
    for (const auto& c : impl_->cache) if (c.width) {
        const auto begin = c.global || impl_->frontier < impl_->config.window ? 0 : impl_->frontier - impl_->config.window;
        for (auto p = begin; p < impl_->frontier; ++p) {
            const auto slot = c.global ? std::size_t(p) : std::size_t(p % impl_->config.window);
            h.update({reinterpret_cast<const std::byte*>(&c.tags[slot]), sizeof(std::uint64_t)});
            h.update({reinterpret_cast<const std::byte*>(c.k.data() + slot * c.width), c.width * sizeof(float)});
            h.update({reinterpret_cast<const std::byte*>(c.v.data() + slot * c.width), c.width * sizeof(float)});
        }
    }
    return strix::model_content_identity_hex(h.finalize());
}
HostWeights bind_host_weights(const ModelManifest& m) {
    HostWeights w; w.config = m.config;
    const std::string p = "model.language_model.";
    auto get = [&](const std::string& name) { return tensor(m, p + name).view; };
    w.embedding = get("embed_tokens.weight"); w.ple_table = get("embed_tokens_per_layer.weight");
    w.ple_model = get("per_layer_model_projection.weight"); w.ple_model_norm = get("per_layer_projection_norm.weight");
    w.final_norm = get("norm.weight");
    for (const auto& t : m.tensors) if (t.name == "lm_head.weight") w.head = t.view;
    for (std::uint32_t l = 0; l < m.config.layers; ++l) {
        LayerWeights y;
        const auto s = "layers." + std::to_string(l) + ".";
        y.input_norm = get(s + "input_layernorm.weight"); y.post_attention_norm = get(s + "post_attention_layernorm.weight");
        y.pre_ffn_norm = get(s + "pre_feedforward_layernorm.weight"); y.post_ffn_norm = get(s + "post_feedforward_layernorm.weight");
        y.post_ple_norm = get(s + "post_per_layer_input_norm.weight"); y.scalar = get(s + "layer_scalar");
        y.q = get(s + "self_attn.q_proj.weight"); y.o = get(s + "self_attn.o_proj.weight"); y.q_norm = get(s + "self_attn.q_norm.weight");
        if (m.config.layer[l].producer == l) {
            y.k = get(s + "self_attn.k_proj.weight"); y.v = get(s + "self_attn.v_proj.weight"); y.k_norm = get(s + "self_attn.k_norm.weight");
        }
        y.gate = get(s + "mlp.gate_proj.weight"); y.up = get(s + "mlp.up_proj.weight"); y.down = get(s + "mlp.down_proj.weight");
        y.ple_gate = get(s + "per_layer_input_gate.weight"); y.ple_projection = get(s + "per_layer_projection.weight");
        w.layer.push_back(y);
    }
    return w;
}
struct HostTarget::Impl {
    HostWeights w;
    RopeTables tables;
    OutputPool pool;
    LinearRecipe recipe;
    Impl(HostWeights weights, RopeTables t, std::uint32_t workers, LinearRecipe r) : w(std::move(weights)), tables(std::move(t)), pool(host_workers(workers)), recipe(std::move(r)) {}
    void linear(LinearView v, const std::vector<float>& x, std::size_t rows, std::vector<float>& out) {
        out.resize(rows * v.outputs);
        // A named recipe (g4-a8-v1) owns its quantized formats; EXL3 and the rest keep their recipes.
        if (recipe && recipe(v, x.data(), rows, out.data(), [this](std::size_t count, const std::function<void(std::size_t, std::size_t)>& job) { pool.run(count, job); })) {
            finite(out);
            return;
        }
        if (v.format == WeightFormat::exl3_mul1) {
            std::vector<float> transformed(rows * v.inputs);
            exl3::input_transform(v, x.data(), unsigned(rows), transformed.data());
            pool.run(v.outputs, [&](std::size_t b, std::size_t e) {
                exl3::core_linear(v, transformed.data(), unsigned(rows), out.data(), unsigned(b), unsigned(e));
            });
            exl3::output_transform(v, out.data(), unsigned(rows));
            return;
        }
        pool.run(v.outputs, [&](std::size_t b, std::size_t e) {
            std::vector<float> decoded(v.inputs);
            for (auto o = b; o < e; ++o) {
                for (std::uint32_t i = 0; i < v.inputs; ++i) decoded[i] = weight_at(v, std::uint32_t(o), i);
                for (std::size_t r = 0; r < rows; ++r)
                    out[r * v.outputs + o] = chunk16_dot([&](std::uint32_t i) { return decoded[i]; }, x.data() + r * v.inputs, v.inputs);
            }
        });
        finite(out);
    }
    void head_norm_rope(float* x, std::uint32_t count, std::uint32_t dim, const float* scale, std::uint64_t position, bool global) {
        const auto pairs = global ? 256U : 128U;
        const auto table = global ? tables.global : tables.sliding;
        const auto base = std::size_t(position) * pairs * 2;
        for (std::uint32_t h = 0; h < count; ++h) {
            float* v = x + h * dim;
            if (!std::isfinite(lane_dot(v,v,dim))) throw std::runtime_error("nonfinite g4-v1 head norm reduction");
            rms_norm(v, scale, v, dim, w.config.eps);
            for (std::uint32_t i = 0; i < std::min(dim / 2U, global ? 64U : 128U); ++i) {
                const float a = v[i], b = v[i + dim / 2U], cs = table[base + 2U * i], sn = table[base + 2U * i + 1];
                v[i] = a * cs - b * sn; v[i + dim / 2U] = b * cs + a * sn;
            }
        }
    }
};
HostTarget::HostTarget(HostWeights w, RopeTables tables, std::uint32_t workers, LinearRecipe recipe) : impl_(std::make_unique<Impl>(std::move(w), std::move(tables), workers, std::move(recipe))) {
    validate_config(impl_->w.config);
    if (impl_->w.layer.size() != impl_->w.config.layers) throw std::runtime_error("incomplete host layer weights");
    if (std::fegetround() != FE_TONEAREST) throw std::runtime_error("g4-v1 requires round-to-nearest-even");
    volatile float minimum = bit_cast<float>(1U), one = 1.0F;
    if (bit_cast<std::uint32_t>(float(minimum * one)) != 1U) throw std::runtime_error("g4-v1 requires gradual underflow");
    const auto& t = impl_->tables;
    if (!t.positions || t.positions > max_positions || t.sliding.size() != std::size_t(t.positions) * 256 || t.global.size() != std::size_t(t.positions) * 512) throw std::runtime_error("invalid immutable rotary tables");
}
HostTarget::~HostTarget() = default;
ForwardResult HostTarget::forward_rows(std::span<const RowPlan> plans) {
    const auto& w = impl_->w;
    const auto& c = w.config;
    std::vector<const RowPlan*> rows;
    std::set<std::size_t> outputs;
    std::map<HostStream::Impl::Txn*, std::vector<std::uint64_t>> transactions;
    ForwardResult result;
    for (const auto& row : plans) if (row.active) {
        auto* tx = dynamic_cast<HostStream::Impl::Txn*>(row.transaction);
        if (!tx || tx->published || (tx->rows && (!tx->reserved || !tx->evaluated)) || !same_config(tx->stream->config,c)) throw std::runtime_error("invalid host row transaction");
        if (row.token >= c.vocab || row.position >= c.max_positions || row.position >= impl_->tables.positions || row.output_row >= plans.size() || !outputs.insert(row.output_row).second) throw std::runtime_error("invalid host row plan");
        rows.push_back(&row); transactions[tx].push_back(row.position);
        result.rows = std::max(result.rows, row.output_row + 1);
    }
    if (rows.size() > 8) throw std::runtime_error("g4-v1 host admits at most eight rows per pass");
    for (auto& [tx, positions] : transactions) {
        std::sort(positions.begin(), positions.end());
        for (std::size_t i = 0; i < positions.size(); ++i)
            if (positions[i] != tx->start + tx->rows + i) throw std::runtime_error("target transaction rows must be a contiguous prefix");
        if (tx->reserved && positions.size() > tx->reserved - tx->rows) throw std::runtime_error("host overlay reservation exceeded");
    }
    for (auto& [tx, positions] : transactions) {
        tx->rows += positions.size();
        tx->evaluated = false;
        for (std::size_t l = 0; l < tx->overlay.size(); ++l) {
            tx->overlay[l].k.resize(tx->rows * tx->stream->cache[l].width);
            tx->overlay[l].v.resize(tx->overlay[l].k.size());
        }
    }
    const auto n = rows.size();
    result.vocab = c.vocab; result.tap_width = 5 * c.hidden;
    result.logits.resize(result.rows * c.vocab); result.taps.resize(result.rows * result.tap_width);
    if (n == 0) return result;
    std::vector<float> residual(n * c.hidden), x, q, k, v, attention, out, gate, up, ple;
    for (std::size_t r = 0; r < n; ++r)
        for (std::uint32_t i = 0; i < c.hidden; ++i) residual[r * c.hidden + i] = embedding_value(weight_at(w.embedding, rows[r]->token, i));
    impl_->linear(w.ple_model, residual, n, ple);
    for (auto& a : ple) a = a * model_ple_scale;
    const auto ple_norm = vector_weight(w.ple_model_norm);
    for (std::size_t r = 0; r < n; ++r)
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            auto* slice = ple.data() + (r * c.layers + l) * c.ple_dim;
            if (!std::isfinite(lane_dot(slice,slice,c.ple_dim))) throw std::runtime_error("nonfinite g4-v1 PLE norm reduction");
            rms_norm(slice, ple_norm.data(), slice, c.ple_dim, c.eps);
            for (std::uint32_t i = 0; i < c.ple_dim; ++i) {
                const float table = embedding_value(weight_at(w.ple_table, rows[r]->token, l * c.ple_dim + i), ple_embedding_scale);
                slice[i] = (slice[i] + table) * ple_combine_scale;
            }
        }
    finite(ple);
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const auto& y = w.layer[l]; const auto& lc = c.layer[l];
        const auto d = lc.head_dim, qw = c.heads * d, kw = c.kv_heads * d;
        const bool global = lc.attention == AttentionKind::global;
        norm(residual, y.input_norm, x, n, c.hidden, c.eps);
        impl_->linear(y.q, x, n, q);
        const auto qnorm = vector_weight(y.q_norm);
        for (std::size_t r = 0; r < n; ++r) impl_->head_norm_rope(q.data() + r * qw, c.heads, d, qnorm.data(), rows[r]->position, global);
        if (lc.producer == l) {
            impl_->linear(y.k, x, n, k); impl_->linear(y.v, x, n, v);
            const auto knorm = vector_weight(y.k_norm);
            for (std::size_t r = 0; r < n; ++r) {
                impl_->head_norm_rope(k.data() + r * kw, c.kv_heads, d, knorm.data(), rows[r]->position, global);
                for (std::uint32_t h = 0; h < c.kv_heads; ++h) {
                    auto* head = v.data() + r * kw + h * d;
                    if (!std::isfinite(lane_dot(head,head,d))) throw std::runtime_error("nonfinite g4-v1 V norm reduction");
                    rms_norm(head,nullptr,head,d,c.eps);
                }
                auto* tx = static_cast<HostStream::Impl::Txn*>(rows[r]->transaction);
                const auto index = std::size_t(rows[r]->position - tx->start) * kw;
                std::copy_n(k.data() + r * kw, kw, tx->overlay[l].k.data() + index);
                std::copy_n(v.data() + r * kw, kw, tx->overlay[l].v.data() + index);
            }
            finite(k); finite(v);
        }
        finite(q);
        attention.resize(n * qw);
        // Attention head outputs are independent; each key/dimension sum retains its declared order.
        for (std::size_t r = 0; r < n; ++r) {
            const auto p = rows[r]->position;
            const auto first = global || p < c.window ? 0 : p - c.window + 1;
            const auto* tx = static_cast<HostStream::Impl::Txn*>(rows[r]->transaction);
            for (std::uint32_t h = 0; h < c.heads; ++h) {
                const auto g = h / (c.heads / c.kv_heads);
                std::vector<float> scores(std::size_t(p - first + 1));
                float maximum = -std::numeric_limits<float>::infinity();
                for (auto j = first; j <= p; ++j) {
                    const auto score = lane_dot(q.data() + r * qw + h * d, tx->read(lc.producer, j, false) + g * d, d);
                    scores[std::size_t(j - first)] = score;
                    maximum = std::max(maximum, score);
                }
                finite(scores);
                for (auto& score : scores) {
                    const float difference = score - maximum;
                    if (!std::isfinite(difference)) throw std::runtime_error("nonfinite g4-v1 attention score difference");
                    score = declared_exp(difference);
                }
                float denominator = 0.0F;
                std::vector<float> numerator(d, 0.0F);
                for (auto chunk = first / attention_chunk; chunk <= p / attention_chunk; ++chunk) {
                    const auto b = std::max(first, chunk * attention_chunk), e = std::min(p + 1, (chunk + 1) * attention_chunk);
                    float part = 0.0F;
                    for (auto j = b; j < e; ++j) part = part + scores[std::size_t(j - first)];
                    denominator = denominator + part;
                    for (std::uint32_t i = 0; i < d; ++i) {
                        float acc = 0.0F;
                        for (auto j = b; j < e; ++j) acc = acc + scores[std::size_t(j - first)] * tx->read(lc.producer, j, true)[g * d + i];
                        numerator[i] = numerator[i] + acc;
                    }
                }
                for (std::uint32_t i = 0; i < d; ++i) attention[r * qw + h * d + i] = numerator[i] / denominator;
            }
        }
        finite(attention);
        impl_->linear(y.o, attention, n, out); norm(out, y.post_attention_norm, x, n, c.hidden, c.eps);
        for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = residual[i] + x[i];
        norm(residual, y.pre_ffn_norm, x, n, c.hidden, c.eps);
        impl_->linear(y.gate, x, n, gate); impl_->linear(y.up, x, n, up);
        for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = gelu_tanh(gate[i]) * up[i];
        finite(gate);
        impl_->linear(y.down, gate, n, out); norm(out, y.post_ffn_norm, x, n, c.hidden, c.eps);
        for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = residual[i] + x[i];
        impl_->linear(y.ple_gate, residual, n, gate);
        for (std::size_t r = 0; r < n; ++r)
            for (std::uint32_t i = 0; i < c.ple_dim; ++i) gate[r * c.ple_dim + i] = gelu_tanh(gate[r * c.ple_dim + i]) * ple[(r * c.layers + l) * c.ple_dim + i];
        finite(gate);
        impl_->linear(y.ple_projection, gate, n, out); norm(out, y.post_ple_norm, x, n, c.hidden, c.eps);
        const float scalar = weight_at(y.scalar, 0, 0);
        for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = (residual[i] + x[i]) * scalar;
        finite(residual);
        for (std::size_t t = 0; t < 5; ++t) if (tap_layers[t] == l)
            for (std::size_t r = 0; r < n; ++r) std::copy_n(residual.data() + r * c.hidden, c.hidden, result.taps.data() + rows[r]->output_row * result.tap_width + t * c.hidden);
    }
    norm(residual, w.final_norm, x, n, c.hidden, c.eps);
    if (std::any_of(rows.begin(),rows.end(),[](const RowPlan* r) { return r->export_hidden; })) {
        result.hidden.resize(result.rows * c.hidden);
        for (std::size_t r = 0; r < n; ++r) if (rows[r]->export_hidden)
            std::copy_n(x.data() + r * c.hidden,c.hidden,result.hidden.data() + rows[r]->output_row * c.hidden);
    }
    impl_->linear(w.head.data ? w.head : w.embedding, x, n, out); // unscaled head weights
    for (std::size_t r = 0; r < n; ++r)
        for (std::uint32_t i = 0; i < c.vocab; ++i) result.logits[rows[r]->output_row * c.vocab + i] = softcap(out[r * c.vocab + i]);
    finite(result.logits); finite(result.taps);
    for (const auto& [tx, positions] : transactions) { (void)positions; tx->evaluated = true; }
    return result;
}
OwnedRopeTables load_rope_tables(const std::filesystem::path& directory) {
    const auto receipt = read_json(directory / "tables.json");
    using namespace model_detail;
    if (require(receipt, "identity").as_string() != arithmetic_id) throw std::runtime_error("rotary identity mismatch");
    const auto positions = integer(require(receipt, "positions"));
    if (!positions || positions > max_positions) throw std::runtime_error("invalid rotary position limit");
    OwnedRopeTables t;
    t.sliding = std::make_shared<MappedFile>(directory / "sliding.f32"); t.global = std::make_shared<MappedFile>(directory / "global.f32");
    if (t.sliding->size() != positions * 256 * 4 || t.global->size() != positions * 512 * 4) throw std::runtime_error("rotary table size mismatch");
    const auto sh = hash_bytes(t.sliding->data(), t.sliding->size()), gh = hash_bytes(t.global->data(), t.global->size());
    const auto& hashes = require(receipt, "sha256");
    if (require(hashes, "sliding.f32").as_string() != sh || require(hashes, "global.f32").as_string() != gh) throw std::runtime_error("rotary table hash mismatch");
    t.view = {{reinterpret_cast<const float*>(t.sliding->data()), t.sliding->size() / 4},
        {reinterpret_cast<const float*>(t.global->data()), t.global->size() / 4}, std::uint32_t(positions), sh + ":" + gh};
    finite(t.view.sliding); finite(t.view.global);
    for (std::uint32_t p = 0; p < positions; ++p)
        for (std::uint32_t i = 64; i < 256; ++i)
            if (bit_cast<std::uint32_t>(t.view.global[std::size_t(p) * 512 + 2 * i]) != 0x3f800000U || bit_cast<std::uint32_t>(t.view.global[std::size_t(p) * 512 + 2 * i + 1]) != 0) throw std::runtime_error("noncanonical inactive rotary coordinates");
    return t;
}
} // namespace strix::gemma4
