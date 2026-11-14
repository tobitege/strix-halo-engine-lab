#include "strix/qwen_prefill_pair.hpp"
#include "strix/qwen_prefill_balance.hpp"
#include "xport/crc32c_fold.hpp"
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>

namespace {
struct Wire {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::vector<std::byte>> messages;
    bool closed = false;
    void send(std::span<const std::byte> bytes) {
        std::lock_guard lock(mutex);
        messages.emplace_back(bytes.begin(), bytes.end()); cv.notify_all();
    }
    void receive(std::span<std::byte> bytes) {
        std::unique_lock lock(mutex);
        if (!cv.wait_for(lock, std::chrono::seconds(5), [&] { return closed || !messages.empty(); }))
            throw std::runtime_error("fake transport timed out");
        if (messages.empty()) throw std::runtime_error("fake transport closed");
        auto message = std::move(messages.front()); messages.pop_front();
        if (message.size() != bytes.size()) throw std::runtime_error("fake transport byte count mismatch");
        std::copy(message.begin(), message.end(), bytes.begin());
    }
    void stop() { std::lock_guard lock(mutex); closed = true; cv.notify_all(); }
};
void require(bool value) { if (!value) throw std::runtime_error("pair prefill schedule assertion failed"); }
struct MockStaging final : strix::qwen_pp::PipeStaging {
    std::vector<std::byte> storage;
    std::mutex& mutex;
    std::condition_variable& cv;
    std::uint64_t& received;
    MockStaging(std::mutex& lock, std::condition_variable& condition, std::uint64_t& count)
        : mutex(lock), cv(condition), received(count) {}
    std::span<std::byte> resize(std::size_t bytes) override { storage.resize(bytes); return storage; }
    void copy(bool, std::size_t, void*, std::size_t) override {}
    void seal(std::size_t) override {}
    void await_piece(std::size_t) override {}
    void received_piece(std::size_t, std::size_t) override {}
    void finish_receive() override { std::lock_guard lock(mutex); ++received; cv.notify_all(); }
};
void balance_checks() {
    using namespace strix::qwen_pp;
    LegalCuts legal{}; legal.fill(true);
    require(parse_cut(nullptr, legal) == measured_default_cut && measured_default_cut == 23U);
    require(parse_cut("24", legal) == 24U);
    {
        LegalCuts no_default = legal; no_default[measured_default_cut] = false;
        require(parse_cut(nullptr, no_default) == 24U);
    }
    for (const char* bad : {"0", "48", "-1", "24x", "", " 24"}) {
        bool rejected = false;
        try { (void)parse_cut(bad, legal); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected);
    }
    legal[25] = false;
    bool rejected = false;
    try { (void)parse_cut("25", legal); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
    LayerCosts costs; costs.layers.fill(1); costs.lm_head = 8;
    require(choose_cut(costs, legal).cut == 28U);
    legal[28] = false; require(choose_cut(costs, legal).cut == 27U);
    auto prediction = predict_cut(costs, 24);
    require(prediction.a == 24 && prediction.b == 32);
    std::string layers = "{";
    for (unsigned layer = 0; layer < 48U; ++layer) {
        if (layer != 0U) layers += ',';
        layers += '"' + std::to_string(layer) + "\":1";
    }
    layers += '}';
    const auto bucket = [&](unsigned context, unsigned head) {
        return "{\"context\":" + std::to_string(context) + ",\"layers\":" + layers + ",\"lm_head\":" + std::to_string(head) + '}';
    };
    const auto json = "{\"buckets\":[" + bucket(131072, 8) + ',' + bucket(32768, 0) + "]}";
    require(parse_costs(json, 32000).context == 32768U);
    require(parse_costs(json, 32769).lm_head == 8);
    require(parse_costs(json, 262144).context == 131072U);
    require(choose_cut(parse_costs(json, 32769), legal).cut == 27U);
    for (const std::string& bad : {std::string("{\"buckets\":[]}"),
        "{\"buckets\":[" + bucket(1, 0) + ',' + bucket(1, 0) + "]}",
        std::string("{\"buckets\":[{\"context\":1,\"layers\":{},\"lm_head\":0}]}")}) {
        rejected = false;
        try { (void)parse_costs(bad, 1); } catch (const std::exception&) { rejected = true; }
        require(rejected);
    }
    legal.fill(false); rejected = false;
    try { (void)choose_cut(costs, legal); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
}
void half_cut_checks() {
    using namespace strix::qwen_pp;
    using strix::pair::StageCut;
    require(StageCut::parse("23") == StageCut(23U) && StageCut::parse("22.5") == StageCut(22U, true));
    require(StageCut(22U, true).text() == "22.5" && StageCut(23U).text() == "23");
    require(StageCut(23U).wire() == 23U && StageCut(22U, true).wire() == ((1ULL << 32U) | 22U));
    require(StageCut(22U, true).state_begin() == 23U && StageCut(23U).state_begin() == 23U);
    require(StageCut(22U).units() < StageCut(22U, true).units() && StageCut(22U, true).units() < StageCut(23U).units());
    for (const char* bad : {"", "x", "22.", "22.50", "22.5x", ".5", "22,5"}) {
        bool rejected = false;
        try { (void)StageCut::parse(bad); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected);
    }
    LegalCuts legal{}; legal.fill(true);
    LegalCuts legal_half{}; legal_half[22] = true;
    require(parse_stage_cut(nullptr, legal, legal_half) == StageCut(measured_default_cut));
    require(parse_stage_cut("24", legal, legal_half) == StageCut(24U));
    require(parse_stage_cut("22.5", legal, legal_half) == StageCut(22U, true));
    for (const char* bad : {"21.5", "0.5", "48.5", "22.25", "22.5 "}) {
        bool rejected = false;
        try { (void)parse_stage_cut(bad, legal, legal_half); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected);
    }
    // Step topology: whole cuts need the attention mix first; half cuts need the feed-forward mix before its inject.
    struct Step { std::optional<std::uint32_t> layer; int op; };
    const std::vector<Step> steps{{1U, 0}, {1U, 1}, {1U, 2}, {1U, 3}, {2U, 1}, {2U, 0}, {2U, 3}, {2U, 2}, {std::nullopt, 2}};
    const auto whole = legal_cuts(steps, 0, 1, true), half = legal_cuts(steps, 2, 3, false);
    require(whole[1] && !whole[2] && half[1] && !half[2] && !half[3]);
    // Costs: 48 equal layers of 2 units whose attention sub-block is 1 unit, tail 1: whole cuts give 48/49 at best,
    // the half cut 24.5 gives 49/48; with a cheaper attention (0.5) the half cut balances to 48.5/48.5.
    LayerCosts costs; costs.layers.fill(2.0); costs.lm_head = 1.0;
    legal_half.fill(true);
    require(!choose_cut(costs, legal, legal_half).half);            // no attention costs: whole cuts only
    require(choose_cut(costs, legal).cut == 24U);
    costs.attention.fill(0.5); costs.has_attention = true;
    const auto chosen = choose_cut(costs, legal, legal_half);
    require(chosen.cut == 24U && chosen.half && chosen.a == 48.5 && chosen.b == 48.5);
    require(chosen.stage() == StageCut(24U, true));
    const auto explicit_half = predict_cut(costs, StageCut(22U, true));
    require(explicit_half.half && explicit_half.a == 44.5 && explicit_half.b == 52.5);
    costs.has_attention = false;                                     // explicit half cut without attention costs
    require(predict_cut(costs, StageCut(22U, true)).a == 45.0);
    std::string json = "{\"buckets\":[{\"context\":32768,\"lm_head\":1,\"layers\":{";
    for (int layer = 0; layer < 48; ++layer) json += (layer ? ",\"" : "\"") + std::to_string(layer) + "\":2";
    json += "},\"attention\":{";
    for (int layer = 0; layer < 48; ++layer) json += (layer ? ",\"" : "\"") + std::to_string(layer) + "\":0.5";
    const auto parsed = parse_costs(json + "}}]}", 32768);
    require(parsed.has_attention && parsed.attention[47] == 0.5 && choose_cut(parsed, legal, legal_half).half);
    bool rejected = false;
    try { (void)parse_costs(json + ",\"48\":1}}]}", 32768); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
}
void chunk_checks() {
    using namespace strix::qwen_pp;
    LayerCosts costs; costs.layers.fill(0.001); costs.has_transfer = true;
    costs.transfer_ms_per_row = 0.01; costs.fixed_ms = 20;
    const auto prediction = predict_cut(costs, 24);
    require(choose_chunk("auto", 16384, 32768, &costs, prediction).rows == 2048);
    require(choose_chunk(nullptr, 16384, 32768, &costs, prediction).rows == 4096);   // measured default
    require(choose_chunk(nullptr, 16384, 131072, nullptr, prediction).rows == 4096);
    require(choose_chunk(nullptr, 16384, 4096, nullptr, prediction).rows == 16384);  // 4096 not < prompt: legacy
    require(choose_chunk(nullptr, 2048, 32768, nullptr, prediction).rows == 2048);   // above admitted tile: legacy
    require(choose_chunk("auto", 8192, 131072, &costs, prediction).rows == 2048);
    require(choose_chunk("legacy", 16384, 32768, &costs, prediction).rows == 16384);
    require(choose_chunk("auto", 16384, 32768, nullptr, prediction).rows == 16384);
    costs.has_transfer = false;
    require(choose_chunk("auto", 8192, 32768, &costs, prediction).rows == 8192);
    costs.has_transfer = true; costs.fixed_ms = 1e6;
    require(choose_chunk("auto", 8192, 32768, &costs, prediction).rows == 8192);
    for (const char* value : {"2048", "4096", "8192", "16", "32"})
        require(valid_chunk(choose_chunk(value, 16384, 32768, nullptr, prediction).rows, 32768));
    for (const char* value : {"", "0", "8", "17", "-2048", "4096x", "32768"}) {
        bool rejected = false;
        try { (void)choose_chunk(value, 16384, 32768, nullptr, prediction); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected);
    }
    require(!valid_chunk(2048, 32776)); // eight-row terminal tail cannot execute
    require(!valid_chunk(2048, 0));
    costs.fixed_ms = 20;
    require(legal_prefix(1193, 4096) == 1192);
    require(legal_prefix(32777, 4096) == 32768);
    require(legal_prefix(32793, 4096) == 32792);
    require(choose_chunk(nullptr, 16384, 32769, nullptr, prediction).rows == 4096);
    require(choose_prompt_route(1193, 4096, nullptr, prediction).prefix == 0);
    require(choose_prompt_route(32771, 4096, nullptr, prediction).prefix == 32768);
    require(choose_prompt_route(24576, 4096, nullptr, prediction).prefix == 0);
    require(choose_prompt_route(28672, 4096, nullptr, prediction).prefix == 28672);
    require(choose_prompt_route(32777, 4096, nullptr, prediction).prefix == 32768);
    require(choose_prompt_route(32793, 4096, nullptr, prediction).prefix == 32792);
    auto route_costs = costs;
    route_costs.fixed_ms = 0; route_costs.transfer_ms_per_row = 0;
    require(choose_prompt_route(131072, 4096, &route_costs, CutPrediction{24U, 0.3, 0.3}).prefix != 0);
    route_costs.fixed_ms = 1e9;
    require(choose_prompt_route(131072, 4096, &route_costs, CutPrediction{24U, 0.3, 0.3}).prefix == 0);
    const auto bucket = [&](const std::string& extra) {
        std::string text = "{\"buckets\":[{\"context\":32768,\"layers\":{";
        for (unsigned i = 0; i < 48; ++i) text += (i ? "," : "") + std::string("\"") + std::to_string(i) + "\":0.001";
        return text + "},\"lm_head\":0" + extra + "}]}";
    };
    require(parse_costs(bucket(",\"transfer_ms_per_row\":0.01,\"fixed_ms\":20"), 32768).has_transfer);
    for (const auto extra : {",\"fixed_ms\":20", ",\"transfer_ms_per_row\":-1,\"fixed_ms\":20"}) {
        bool rejected = false;
        try { (void)parse_costs(bucket(extra), 32768); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected);
    }
}
void blocked_send_overlap_checks() {
    using namespace strix::qwen_pp;
    Wire forward, backward;
    std::mutex gate; std::condition_variable cv;
    bool sending = false, posted_next = false;
    const auto abort = [&] { forward.stop(); backward.stop(); };
    Pipe producer(0, 2, [&](auto bytes) {
        if (bytes.size() == sizeof(PipeFrame::header)) {
            std::unique_lock lock(gate);
            if (!sending) {
                sending = true; cv.notify_all();
                if (!cv.wait_for(lock, std::chrono::seconds(5), [&] { return posted_next; }))
                    throw std::runtime_error("stage A did not post next chunk during blocked send");
            }
        }
        forward.send(bytes);
    }, [&](auto bytes) { backward.receive(bytes); }, abort);
    Pipe consumer(1, 2, [&](auto bytes) { backward.send(bytes); }, [&](auto bytes) { forward.receive(bytes); }, abort);
    for (std::uint64_t i = 0; i < 2U; ++i) {
        auto& frame = producer.acquire(i);
        frame.header[0] = Seam::magic; frame.header[1] = i;
        frame.header[3] = 16; frame.header[4] = 24; frame.header[41] = 2;
        frame.bytes.assign(16U * Seam::row_bytes, static_cast<std::byte>(i));
        producer.release(i);
        std::unique_lock lock(gate);
        if (i == 0U) require(cv.wait_for(lock, std::chrono::seconds(5), [&] { return sending; }));
        else { posted_next = true; cv.notify_all(); }
    }
    for (std::uint64_t i = 0; i < 2U; ++i) {
        auto& frame = consumer.acquire(i);
        require(frame.header[1] == i);
        require(std::all_of(frame.payload.begin(), frame.payload.end(), [&](auto byte) { return byte == static_cast<std::byte>(i); }));
        consumer.release(i);
    }
    consumer.finish(2); producer.finish(2);
}
void crc_checks() {
    using namespace strix::qwen_pp;
    const std::string known = "123456789";
    SeamCrc check; check.update(std::as_bytes(std::span(known)));
    require(check.value() == 0xe3069283U);
    std::vector<std::byte> data(seam_piece_bytes * 2U + 17U);
    for (std::size_t index = 0; index < data.size(); ++index) data[index] = static_cast<std::byte>(index % 251U);
    SeamCrc whole, pieces, combined; whole.update(data);
    for (std::size_t offset = 0; offset < data.size(); offset += seam_piece_bytes) {
        const auto piece = std::span(data).subspan(offset, std::min(seam_piece_bytes, data.size() - offset));
        pieces.update(piece); SeamCrc checksum; checksum.update(piece); combined.append(checksum.value(), piece.size());
    }
    require(whole.value() == pieces.value() && whole.value() == combined.value());
    SeamCrc empty; combined.append(empty.value(), 0); require(whole.value() == combined.value());
}
void fault_checks() {
    using namespace strix::qwen_pp;
    // Exercise actual multi-piece wire framing, and refuse corruption and peer cut drift.
    for (unsigned fault = 0; fault < 4U; ++fault) {
        Wire forward, backward;
        const auto abort = [&] { forward.stop(); backward.stop(); };
        unsigned messages = 0;
        Pipe producer(0, 1, [&](auto bytes) {
            ++messages;
            if ((fault == 1U && messages == 3U) || (fault == 3U && messages == 6U)) {
                std::vector<std::byte> corrupt(bytes.begin(), bytes.end()); corrupt[0] ^= std::byte{1}; forward.send(corrupt);
            } else forward.send(bytes);
        }, [&](auto bytes) { backward.receive(bytes); }, abort, 28);
        Pipe consumer(1, 1, [&](auto bytes) { backward.send(bytes); }, [&](auto bytes) { forward.receive(bytes); }, abort,
            fault == 2U ? 24U : 28U);
        auto& sent = producer.acquire(0);
        sent.header[0] = Seam::magic; sent.header[1] = 0; sent.header[3] = 700; sent.header[4] = 28; sent.header[41] = 1;
        sent.bytes.assign(700U * Seam::row_bytes, std::byte{37}); producer.release(0);
        bool rejected = false;
        try {
            auto& received = consumer.acquire(0);
            require(received.payload.size() == sent.bytes.size());
            require(std::all_of(received.payload.begin(), received.payload.end(), [](auto byte) { return byte == std::byte{37}; }));
            consumer.release(0); consumer.finish(1); producer.finish(1);
        } catch (const std::exception&) { rejected = true; }
        require(rejected == (fault != 0U));
    }
}

void routed_piece_checks() {
    using namespace strix::qwen_pp;
    struct Staging final : PipeStaging {
        std::vector<std::byte> host, device;
        unsigned uploads = 0;
        std::span<std::byte> resize(std::size_t size) override { host.resize(size); device.resize(size); return host; }
        void copy(bool, std::size_t, void*, std::size_t) override {}
        void seal(std::size_t) override { device = host; }
        void await_piece(std::size_t) override {}
        void received_piece(std::size_t offset, std::size_t bytes) override {
            ++uploads; std::copy_n(host.data() + offset, bytes, device.data() + offset);
        }
        void finish_receive() override { require(host == device); }
        void* device_piece(std::size_t offset) override { return device.data() + offset; }
        void read_device_piece(std::size_t offset, std::size_t bytes) override {
            std::copy_n(device.data() + offset, bytes, host.data() + offset);
        }
    };
    // Same framing and CRC checks through a separate payload channel. Exercise
    // a full piece + tail, callback decline, corruption, and callback failure.
    for (unsigned mode = 0; mode < 4U; ++mode) {
        Wire forward, backward, payload;
        const auto abort = [&] { forward.stop(); backward.stop(); payload.stop(); };
        std::atomic<unsigned> sent_pieces{0}, received_pieces{0};
        const auto routed = [&](bool sending, PipeStaging& staging, std::size_t offset, std::size_t size) {
            if (mode == 1U) return false;
            if (mode == 3U) throw std::runtime_error("injected routed exchange failure");
            auto bytes = std::span(static_cast<std::byte*>(staging.device_piece(offset)), size);
            if (sending) {
                ++sent_pieces;
                if (mode == 2U && offset != 0U) bytes[0] ^= std::byte{1};
                payload.send(bytes);
            } else {
                ++received_pieces; payload.receive(bytes); staging.read_device_piece(offset, size);
            }
            return true;
        };
        Pipe producer(0, 1, [&](auto bytes) { forward.send(bytes); }, [&](auto bytes) { backward.receive(bytes); }, abort,
            24, [] { return std::make_shared<Staging>(); }, {}, routed);
        Pipe consumer(1, 1, [&](auto bytes) { backward.send(bytes); }, [&](auto bytes) { forward.receive(bytes); }, abort,
            24, [] { return std::make_shared<Staging>(); }, {}, routed);
        auto& sent = producer.acquire(0);
        sent.header[0] = Seam::magic; sent.header[1] = 0; sent.header[3] = 700; sent.header[4] = 24; sent.header[41] = 1;
        sent.resize(700U * Seam::row_bytes);
        std::fill(sent.payload.begin(), sent.payload.end(), std::byte{71});
        sent.staging->seal(sent.payload.size()); producer.release(0);
        bool rejected = false;
        try {
            auto& received = consumer.acquire(0);
            require(received.payload.size() == sent.payload.size());
            require(std::all_of(received.payload.begin(), received.payload.end(), [](auto b) { return b == std::byte{71}; }));
            const auto& staging = static_cast<const Staging&>(*received.staging);
            require(staging.uploads == (mode == 1U ? 2U : 0U));
            consumer.release(0); consumer.finish(1); producer.finish(1);
        } catch (const std::exception&) { rejected = true; }
        require(rejected == (mode >= 2U));
        if (mode == 0U) require(sent_pieces == 2U && received_pieces == 2U);
    }
}

// The device CRC fold (span digests + tail) is the seam CRC-32C bit for bit, at every length class.
std::uint32_t folded_crc(std::span<const std::byte> bytes) {
    using namespace strix::xport;
    std::vector<std::uint32_t> digests;
    const std::size_t whole = bytes.size() / crc32c_fold_span_bytes;
    for (std::size_t n = 0; n < whole; ++n) digests.push_back(crc32c_span_digest(bytes.subspan(n * crc32c_fold_span_bytes, crc32c_fold_span_bytes)));
    return crc32c_fold(digests, bytes.subspan(whole * crc32c_fold_span_bytes));
}
void crc_fold_checks() {
    using namespace strix::xport;
    const char* text = "123456789";
    require(crc32c_bytes(std::as_bytes(std::span(text, 9))) == 0xe3069283U);
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (const std::size_t size : {std::size_t{1}, std::size_t{255}, std::size_t{4096}, crc32c_fold_span_bytes - 1U,
             crc32c_fold_span_bytes, crc32c_fold_span_bytes + 1U, 3U * crc32c_fold_span_bytes + 777U, 16U * crc32c_fold_span_bytes}) {
        std::vector<std::byte> data(size);
        for (auto& value : data) { state = state * 6364136223846793005ULL + 1442695040888963407ULL; value = static_cast<std::byte>(state >> 56U); }
        strix::qwen_pp::SeamCrc reference; reference.update(data);
        require(folded_crc(data) == reference.value());
        require(crc32c_bytes(data) == reference.value());
        // Appending two runs: the shift factor of the second run's length.
        const std::size_t cut = size / 3U;
        const auto head = std::span(data).first(cut), tail = std::span(data).subspan(cut);
        require(crc32c_append(crc32c_bytes(head), crc32c_bytes(tail), crc32c_shift(tail.size())) == reference.value());
        if (size > 8U) { data[size / 2U] ^= std::byte{4}; require(folded_crc(data) != reference.value()); }
    }
}
// Device-checked pieces: the CRC comes from the device snapshot at both ends and the piece never visits the host.
void device_checked_piece_checks() {
    using namespace strix::qwen_pp;
    struct Staging final : PipeStaging {
        std::vector<std::byte> host, device;
        std::function<bool(std::size_t)> rule;
        std::atomic<unsigned> host_reads{0}, device_crcs{0}, uploads{0};
        std::atomic<unsigned>* received_crcs = nullptr;
        std::span<std::byte> resize(std::size_t size) override { host.assign(size, std::byte{0}); device.assign(size, std::byte{0}); return host; }
        void copy(bool, std::size_t, void*, std::size_t) override {}
        void seal(std::size_t) override {}
        void await_piece(std::size_t) override { ++host_reads; }
        void received_piece(std::size_t offset, std::size_t bytes) override {
            ++uploads; std::copy_n(host.data() + offset, bytes, device.data() + offset);
        }
        void finish_receive() override {}
        void* device_piece(std::size_t offset) override { return device.data() + offset; }
        void read_device_piece(std::size_t, std::size_t) override { throw std::logic_error("device-checked piece read back to the host"); }
        void route_device(std::function<bool(std::size_t)> routed) override { rule = std::move(routed); }
        bool device_routed(std::size_t, std::size_t bytes) override { return rule && rule(bytes); }
        std::uint32_t device_crc(std::size_t offset, std::size_t bytes, bool received) override {
            if (received) ++*received_crcs;
            ++device_crcs; return folded_crc(std::span(device).subspan(offset, bytes));
        }
    };
    const auto rule = [](std::size_t bytes) { return bytes > 65536U; };
    for (unsigned mode = 0; mode < 3U; ++mode) {
        Wire forward, backward, payload;
        const auto abort = [&] { forward.stop(); backward.stop(); payload.stop(); };
        std::atomic<unsigned> moved{0}, send_attempts{0}, receive_attempts{0}, received_crcs{0};
        const auto routed = [&](bool sending, PipeStaging& staging, std::size_t offset, std::size_t size) {
            if (!rule(size)) return false;
            if (sending) ++send_attempts; else ++receive_attempts;
            if (mode == 2U) return false; // a transport that declines a device-checked piece is a logic error
            auto bytes = std::span(static_cast<std::byte*>(staging.device_piece(offset)), size);
            if (sending) {
                std::vector<std::byte> wire(bytes.begin(), bytes.end());
                if (mode == 1U && offset == 0U) wire[size / 2U] ^= std::byte{1};
                payload.send(wire);
            } else { payload.receive(bytes); ++moved; }
            return true;
        };
        const auto make = [&] {
            auto staging = std::make_shared<Staging>();
            staging->received_crcs = &received_crcs;
            staging->route_device(rule); return staging;
        };
        Pipe producer(0, 1, [&](auto bytes) { forward.send(bytes); }, [&](auto bytes) { backward.receive(bytes); }, abort,
            24, make, {}, routed);
        Pipe consumer(1, 1, [&](auto bytes) { backward.send(bytes); }, [&](auto bytes) { forward.receive(bytes); }, abort,
            24, make, {}, routed);
        auto& sent = producer.acquire(0);
        sent.header[0] = Seam::magic; sent.header[1] = 0; sent.header[3] = 700; sent.header[4] = 24; sent.header[41] = 1;
        sent.resize(700U * Seam::row_bytes);
        auto& source = static_cast<Staging&>(*sent.staging);
        for (std::size_t n = 0; n < source.device.size(); ++n) source.device[n] = static_cast<std::byte>(n * 131U + (n >> 9U));
        const auto expected = source.device;
        SeamCrc reference; reference.update(expected);
        producer.release(0);
        PipeFrame* received = nullptr;
        std::exception_ptr acquisition_error;
        try { received = &consumer.acquire(0); }
        catch (...) { acquisition_error = std::current_exception(); }
        // Assertions must never be caught as evidence of pipe rejection.
        if (mode == 0U) {
            if (acquisition_error) std::rethrow_exception(acquisition_error);
            require(received != nullptr);
            const auto& staging = static_cast<const Staging&>(*received->staging);
            require(staging.device == expected);
            require(received->header[40] == reference.value()); // the whole-frame CRC is the host CRC of the same bytes
            const std::size_t pieces = (expected.size() + seam_piece_bytes - 1U) / seam_piece_bytes;
            const std::size_t last = expected.size() - (pieces - 1U) * seam_piece_bytes;
            const unsigned device_pieces = static_cast<unsigned>(pieces - (rule(last) ? 0U : 1U));
            require(moved == device_pieces && staging.device_crcs == device_pieces && source.device_crcs == device_pieces);
            require(send_attempts == device_pieces && receive_attempts == device_pieces && received_crcs == device_pieces);
            require(source.host_reads == pieces - device_pieces && staging.uploads == pieces - device_pieces);
            consumer.release(0); consumer.finish(1); producer.finish(1);
        } else {
            require(received == nullptr && acquisition_error != nullptr);
            std::string reason;
            try { std::rethrow_exception(acquisition_error); }
            catch (const std::runtime_error& error) { reason = error.what(); }
            if (mode == 1U) {
                require(reason == "pair prefill seam piece CRC mismatch");
                require(moved == 1U && receive_attempts == 1U && received_crcs == 1U);
                require(send_attempts >= 1U && source.device_crcs >= 1U);
            } else {
                require(reason == "fake transport closed");
                // The decline originates on the producer, before it sends a payload.
                // A peer abort alone cannot prove that the guard ran.
                bool declined = false;
                try { producer.finish(1); }
                catch (const std::logic_error& error) {
                    declined = std::string_view(error.what()) ==
                        "pair prefill device-checked piece was not moved by the transport";
                }
                require(declined);
                require(send_attempts == 1U && source.device_crcs == 1U && source.host_reads == 0U);
                require(moved == 0U && received_crcs == 0U);
            }
        }
    }
}

}
// Default strided seam copy is exactly the per-row copy sequence (offsets, device addresses, lengths).
void copy_rows_checks() {
    struct Recorder final : strix::qwen_pp::PipeStaging {
        std::vector<std::array<std::size_t, 3>> calls;
        std::span<std::byte> resize(std::size_t) override { return {}; }
        void copy(bool, std::size_t offset, void* device, std::size_t bytes) override {
            calls.push_back({offset, reinterpret_cast<std::uintptr_t>(device), bytes});
        }
        void seal(std::size_t) override {}
        void await_piece(std::size_t) override {}
        void received_piece(std::size_t, std::size_t) override {}
        void finish_receive() override {}
    } recorder;
    std::array<std::byte, 64> device{};
    recorder.copy_rows(true, 5U, 100U, device.data(), 12U, 7U, 4U);
    require(recorder.calls.size() == 4U);
    for (std::size_t row = 0; row < 4U; ++row)
        require(recorder.calls[row][0] == 5U + row * 100U &&
            recorder.calls[row][1] == reinterpret_cast<std::uintptr_t>(device.data()) + row * 12U && recorder.calls[row][2] == 7U);
}
// Shared pair timeline: OFF records nothing; ON accumulates per step, and the pipe fills its progress columns.
void timeline_checks() {
    strix::dist::PipelineTimeline off("t", 0, strix::qwen_pp::pair_timing_columns(), false);
    off.add_ms(3, strix::qwen_pp::pt_wire, 1.0);
    require(!off.enabled() && off.measure(0, 0, [] { return 7; }) == 7);
    strix::dist::PipelineTimeline on("t", 0, strix::qwen_pp::pair_timing_columns(), true);
    on.add_ms(1, strix::qwen_pp::pt_wire, 1.5); on.add_ms(1, strix::qwen_pp::pt_wire, 0.5);
    on.add_ms(1, 999U, 1.0); // unknown column ignored
    std::FILE* file = std::tmpfile(); require(file != nullptr);
    on.report(10.0, file); std::rewind(file);
    std::string text; char buffer[512];
    while (std::fgets(buffer, sizeof buffer, file)) text += buffer;
    std::fclose(file);
    require(text.find("step=1 ") != std::string::npos && text.find("step=0 ") == std::string::npos);
    require(text.find("sum steps=1 wall-ms=10.000") != std::string::npos && text.find(" wire=2.000") != std::string::npos);
    Wire forward, backward;
    const auto abort = [&] { forward.stop(); backward.stop(); };
    strix::dist::PipelineTimeline t0("t", 0, strix::qwen_pp::pair_timing_columns(), true);
    strix::dist::PipelineTimeline t1("t", 1, strix::qwen_pp::pair_timing_columns(), true);
    strix::qwen_pp::Pipe producer(0, 3, [&](auto bytes) { forward.send(bytes); }, [&](auto bytes) { backward.receive(bytes); }, abort,
        24, {}, {}, {}, &t0);
    strix::qwen_pp::Pipe consumer(1, 3, [&](auto bytes) { backward.send(bytes); }, [&](auto bytes) { forward.receive(bytes); }, abort,
        24, {}, {}, {}, &t1);
    std::thread stage_a([&] {
        for (std::uint64_t sequence = 0; sequence < 3U; ++sequence) {
            auto& frame = producer.acquire(sequence);
            frame.header[0] = strix::qwen_pp::Seam::magic; frame.header[1] = sequence; frame.header[41] = 3U;
            frame.header[2] = sequence * 16U; frame.header[3] = 1U; frame.header[4] = 24U;
            frame.bytes.assign(strix::qwen_pp::Seam::row_bytes, std::byte{1});
            producer.release(sequence);
        }
        producer.finish(3);
    });
    for (std::uint64_t sequence = 0; sequence < 3U; ++sequence) { consumer.acquire(sequence); consumer.release(sequence); }
    consumer.finish(3);
    stage_a.join();
    std::FILE* rows = std::tmpfile(); require(rows != nullptr);
    t0.report(-1.0, rows); t1.report(-1.0, rows); std::rewind(rows);
    std::string lines;
    while (std::fgets(buffer, sizeof buffer, rows)) lines += buffer;
    std::fclose(rows);
    for (const char* needle : {"rank=0 step=2 ", "rank=1 step=2 ", "rank=0 sum steps=3", "rank=1 sum steps=3"})
        require(lines.find(needle) != std::string::npos);
}
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--device-checked-only") {
            device_checked_piece_checks(); return 0;
        }
        balance_checks(); half_cut_checks(); chunk_checks(); blocked_send_overlap_checks(); crc_checks(); fault_checks(); routed_piece_checks(); crc_fold_checks(); device_checked_piece_checks(); timeline_checks(); copy_rows_checks();
        Wire forward, backward;
        std::mutex received_mutex; std::condition_variable received_cv; std::uint64_t received_chunks = 0;
        const auto abort = [&] { forward.stop(); backward.stop(); };
        strix::qwen_pp::Pipe producer(0, 7, [&](auto bytes) { forward.send(bytes); }, [&](auto bytes) { backward.receive(bytes); }, abort);
        strix::qwen_pp::Pipe consumer(1, 7, [&](auto bytes) { backward.send(bytes); }, [&](auto bytes) { forward.receive(bytes); }, abort, 24,
            [&] { return std::make_shared<MockStaging>(received_mutex, received_cv, received_chunks); });
        std::exception_ptr producer_error;
        std::array<const void*, 2> addresses{};
        std::mutex observed_mutex;
        std::condition_variable observed_cv;
        std::uint64_t submitted = 0;
        std::thread stage_a([&] {
            try {
                for (std::uint64_t sequence = 0; sequence < 7U; ++sequence) {
                    auto& frame = producer.acquire(sequence);
                    const auto slot = static_cast<std::size_t>(sequence % 2U);
                    if (sequence < 2U) addresses[slot] = &frame;
                    else require(addresses[slot] == &frame);
                    const auto rows = static_cast<std::size_t>(sequence + 1U);
                    frame.header[0] = strix::qwen_pp::Seam::magic; frame.header[1] = sequence;
                    frame.header[41] = 7U;
                    frame.header[2] = sequence * 16U; frame.header[3] = rows; frame.header[4] = 24U;
                    frame.bytes.assign(rows * strix::qwen_pp::Seam::row_bytes, static_cast<std::byte>(sequence));
                    producer.release(sequence);
                    { std::lock_guard lock(observed_mutex); submitted = sequence + 1U; observed_cv.notify_all(); }
                }
                producer.finish(7);
            } catch (...) { producer_error = std::current_exception(); abort(); }
        });
        std::size_t total_bytes = 0;
        try {
            for (std::uint64_t sequence = 0; sequence < 7U; ++sequence) {
                auto& frame = consumer.acquire(sequence);
                if (sequence == 0U) {
                    std::unique_lock lock(observed_mutex);
                    require(observed_cv.wait_for(lock, std::chrono::seconds(5), [&] { return submitted == 2U; }));
                    {
                        std::unique_lock received_lock(received_mutex);
                        require(received_cv.wait_for(received_lock, std::chrono::seconds(5), [&] { return received_chunks == 2U; }));
                    }
                    // Stage A has completed the next chunk while stage B owns chunk 0.
                    // Slot 0 cannot be acquired for chunk 2 before this consumer releases it.
                    require(submitted == 2U);
                }
                require(frame.header[1] == sequence);
                require(frame.payload.size() == static_cast<std::size_t>(sequence + 1U) * strix::qwen_pp::Seam::row_bytes);
                require(std::all_of(frame.payload.begin(), frame.payload.end(), [&](auto value) { return value == static_cast<std::byte>(sequence); }));
                total_bytes += frame.payload.size();
                consumer.release(sequence);
            }
            consumer.finish(7);
        } catch (...) { abort(); stage_a.join(); throw; }
        stage_a.join();
        if (producer_error) std::rethrow_exception(producer_error);
        require(total_bytes == 28U * strix::qwen_pp::Seam::row_bytes);
        std::cout << "PASS: chunk policy/transfer schema, blocked-send overlap, cut/model/bucket validation, multi-piece CRC/corruption/cut drift, H2D lookahead before B release, 7 ordered chunks, 2 slots per rank, consumer acknowledgement before reuse, " << total_bytes << " seam bytes\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
