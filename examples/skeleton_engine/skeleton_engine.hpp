#pragma once
// Minimal model family on the shared serve/session layer. It is the smallest
// thing that serves through strix::serve::run(): a tokenizer hook, a chat
// renderer hook, a session::Engine over a session::Backend, and the
// SessionService factory the serve shell selects by registry family.
// docs/adding-an-engine.md walks through this file.
//
// The "model" is a counter: the next token is a letter chosen by the number of
// rows already consumed. Replace next_token() with a forward pass and the rest
// of the file is what a real family still has to provide.
#include "strix/serve/run.hpp"
#include "strix/serve/tokenizer.hpp"
#include "strix/session/backend_cpu.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace skeleton {
namespace ss = strix::session;
namespace sv = strix::serve;

inline constexpr std::uint32_t kEos = 256;     // ids 0..255 are bytes
inline constexpr std::uint32_t kVocab = 257;
inline constexpr std::uint32_t kContext = 256; // rows per slot
inline constexpr unsigned kSlots = 2;

// Hook 1: tokenizer. A real family usually binds serve::BpeTokenizer instead.
class ByteTokenizer final : public sv::ApiTokenizer {
public:
    std::vector<std::uint32_t> encode(std::string_view text, bool = true) const override {
        std::vector<std::uint32_t> ids;
        for (const unsigned char c : text) ids.push_back(c);
        return ids;
    }
    std::string decode(std::span<const std::uint32_t> ids, bool = false) const override {
        std::string text;
        for (const auto id : ids) if (id < 256) text.push_back(static_cast<char>(id));
        return text;
    }
    std::string token_bytes(std::uint32_t id) const override {
        return id < 256 ? std::string(1, static_cast<char>(id)) : std::string("<eos>");
    }
    std::size_t vocab_size() const override { return kVocab; }
    std::uint32_t find(std::string_view text) const override {
        if (text == "<eos>") return kEos;
        if (text.size() == 1) return static_cast<unsigned char>(text[0]);
        throw std::invalid_argument("unknown token");
    }
    std::uint32_t eos_token() const override { return kEos; }
};

// Hook 2: the family adapter. State rows live in buffers of the neutral
// Backend; this file never names an accelerator API. The same class runs on
// any Backend implementation that the service constructs.
class Engine final : public ss::Engine {
    using Rows = std::vector<std::uint32_t>;

    ss::Backend& backend_;
    ss::Queue queue_;
    std::array<ss::Buffer, kSlots> rows_;
    std::array<std::shared_ptr<const ss::EmbedTable>, kSlots> embeds_; // per lane, see bind_embeds
    ss::Capabilities limits_;

public:
    ss::KvStore store; // declared after the members its hooks use

    explicit Engine(ss::Backend& backend)
        : backend_(backend), queue_(backend.create_queue()), store(kSlots, 1, hooks()) {
        for (auto& rows : rows_) rows = backend_.allocate(kContext * sizeof(std::uint32_t));
        limits_.max_context = kContext;
        limits_.max_slots = kSlots;
        limits_.max_batch = 1;
        limits_.prefix_cache = true;
        limits_.embed_rows = true;
    }

    ss::Capabilities caps() const override { return limits_; }
    strix::profile::ExecutionReceipt receipt() const override {
        strix::profile::ExecutionReceipt r;
        r.engine = "skeleton";
        r.implementation = std::string(backend_.backend_name());
        return r;
    }

    void prefill(std::span<const std::uint32_t> ids, ss::KvHandle& k) override {
        anchor(k);
        consume(k, ids);
    }

    // Embedding rows (image input): the table that resolves the virtual ids of
    // the rows this lane consumes next.
    void bind_embeds(ss::KvHandle& k, std::shared_ptr<const ss::EmbedTable> table) override {
        embeds_.at(k.slot) = std::move(table);
    }

    ss::StepResult decode(ss::KvHandle& k, const ss::SamplerState& s) override {
        ss::StepResult r;
        r.before = {store.consumed(k), store.pending(k)};
        r.rng_draws = s.rng_draws;
        const auto used = store.consumed(k) + (store.pending(k) ? 1U : 0U);
        if (s.cancellation.stop_requested()) r.finish = ss::FinishReason::cancelled;
        else if (used >= kContext) r.finish = ss::FinishReason::context;
        else if (!s.max_tokens) r.finish = ss::FinishReason::length;
        else {
            anchor(k);
            const auto id = next_token(k);
            store.set_pending(k, id);
            r.tokens.push_back(id);
            if (std::find(s.stop_tokens.begin(), s.stop_tokens.end(), id) != s.stop_tokens.end())
                r.finish = ss::FinishReason::stop;
            // A step that uses up the allowance or the context is terminal.
            else if (r.tokens.size() == s.max_tokens) r.finish = ss::FinishReason::length;
            else if (used + 1 >= kContext) r.finish = ss::FinishReason::context;
        }
        r.after = {store.consumed(k), store.pending(k)};
        return r;
    }

private:
    // The stand-in for a forward pass over the rows of one slot.
    std::uint32_t next_token(ss::KvHandle k) const {
        const auto n = store.consumed(k);
        return n % 32 == 31 ? kEos : static_cast<std::uint32_t>('a' + n % 26);
    }
    void consume(ss::KvHandle k, std::span<const std::uint32_t> ids) {
        if (ids.empty()) return;
        const auto at = static_cast<std::size_t>(store.consumed(k));
        if (at + ids.size() > kContext) throw std::length_error("skeleton context exceeded");
        // Prevalidation of virtual ids, before any write. A real family then
        // fills its embed stage: token rows by lookup, virtual rows by copying
        // table->find(embed_handle(id))->row(embed_row(id)), plus its per-row
        // extras (role, positions, real_token_id). The stand-in keeps the ids.
        if (ss::has_embed_ids(ids)) {
            const auto& table = embeds_.at(k.slot);
            if (!table) throw std::invalid_argument("embedding rows without a bound table");
            table->validate(ids);
        }
        backend_.upload(queue_, slice(k, at, ids.size()), std::as_bytes(ids));
        backend_.synchronize(queue_);
        store.append_consumed(k, ids);
    }
    // The pending token becomes a consumed row before anything follows it.
    void anchor(ss::KvHandle k) {
        if (const auto pending = store.pending(k)) {
            const std::array<std::uint32_t, 1> id{*pending};
            consume(k, id);
            store.set_pending(k, {});
        }
    }
    ss::BufferSlice slice(ss::KvHandle k, std::size_t first, std::size_t count) const {
        return {rows_.at(k.slot), first * sizeof(std::uint32_t), count * sizeof(std::uint32_t)};
    }
    ss::RankState snapshot(ss::KvHandle k) {
        auto copy = std::make_shared<Rows>(static_cast<std::size_t>(store.consumed(k)));
        if (!copy->empty()) {
            backend_.download(queue_, slice(k, 0, copy->size()), std::as_writable_bytes(std::span(*copy)));
            backend_.synchronize(queue_);
        }
        const auto bytes = copy->size() * sizeof(std::uint32_t);
        return {std::shared_ptr<const Rows>(std::move(copy)), bytes};
    }
    // Hook 3: what the shared KvStore and PrefixCache need to save and restore
    // a slot. Truncation needs no work here because rows past the frontier are
    // never read; a real family drops or masks its KV rows.
    ss::Hooks hooks() {
        ss::Hooks h;
        h.initialize = [](ss::KvHandle) {};
        h.reset = [this](ss::KvHandle k) { embeds_.at(k.slot).reset(); };
        h.release = [this](ss::KvHandle k) { embeds_.at(k.slot).reset(); };
        h.capture = [this](ss::KvHandle k, std::uint32_t) { return snapshot(k); };
        h.view = [this](ss::KvHandle k, std::uint32_t) { return snapshot(k); };
        h.restore = [this](ss::KvHandle k, std::uint32_t, const ss::RankState& state) {
            const auto& rows = *std::static_pointer_cast<const Rows>(state.owner);
            if (rows.empty()) return;
            backend_.upload(queue_, slice(k, 0, rows.size()), std::as_bytes(std::span(rows)));
            backend_.synchronize(queue_);
        };
        h.truncate = [](ss::KvHandle, std::uint32_t, std::uint64_t) {};
        h.partial_restore_supported = true;
        return h;
    }
};

// Hook 4: the service. It owns backend, engine, queue, runtime and the shared
// OpenAI front end, in the order they must outlive each other.
class Service final : public sv::SessionService {
public:
    explicit Service(const sv::ServiceSelection& selection)
        : engine_(backend_), runtime_(queue_, engine_, engine_.store, {}, session_hooks()) {
        sv::ApiConfig config;
        config.model_id = selection.model.name;
        config.sampling_policy = "reject"; // the counter has no sampler
        config.stop_tokens = {kEos};
        config.default_thinking = false;
        // Hook 5: chat template. Here: the last message, verbatim.
        config.chat_renderer = [](const sv::ChatTemplateInput& in) {
            return in.messages.as_array().back().find("content")->as_string();
        };
        runtime_.configure_api(config);
        api_ = std::make_unique<sv::ApiService>(std::move(config),
            sv::ApiTokenizerBinding(std::make_shared<ByteTokenizer>()), queue_, kContext, false);
    }
    sv::SessionRuntime& runtime() override { return runtime_; }
    void handle(const sv::HttpRequest& request, sv::HttpResponseWriter& writer) override {
        api_->handle(request, writer);
    }

private:
    static sv::SessionHooks session_hooks() {
        sv::SessionHooks h;
        h.identity = {"skeleton-weights-v1", "skeleton-profile-v1", "skeleton-state-v1", 1};
        return h;
    }
    ss::CpuBackend backend_; // the only line that picks a backend
    Engine engine_;
    sv::GenerationQueue queue_;
    sv::SessionRuntime runtime_;
    std::unique_ptr<sv::ApiService> api_;
};

// Hook 6: registration under the registry `family` value.
inline void install_session_service(sv::ServiceFactories& factories) {
    factories.install("skeleton", {
        [](const sv::ServiceSelection&) {},
        [](const sv::ServiceSelection& selection, std::ostream& out) {
            out << "skeleton family: model " << selection.model.name << ", cpu backend, "
                << kSlots << " slots, context " << kContext << "\n";
        },
        [](const sv::ServiceSelection& selection) -> std::unique_ptr<sv::SessionService> {
            return std::make_unique<Service>(selection);
        }});
}
} // namespace skeleton
