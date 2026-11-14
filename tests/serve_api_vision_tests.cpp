// Media input through the three API front doors, proven with a fake adapter on the example engine:
// no adapter keeps the old refusals, text requests do not move, image parts become virtual ids that the
// engine resolves through the request's EmbedTable, and usage, receipts, /tokenize and capabilities follow.
#include "../examples/skeleton_engine/skeleton_engine.hpp"

#include "strix/serve/media.hpp"

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

namespace {
namespace sv = strix::serve;
namespace ss = strix::session;

int failures = 0;
void check(bool ok, const char* what, int line) {
    if (ok) return;
    ++failures;
    std::cerr << "FAILED line " << line << ": " << what << "\n";
}
#define CHECK(x) check((x), #x, __LINE__)

struct Writer final : sv::HttpResponseWriter {
    int status = 0;
    std::string body;
    void send(int s, std::string_view, std::string_view b,
              const std::vector<std::pair<std::string, std::string>>&) override { status = s; body = b; }
    bool begin_stream(int s, std::string_view,
                      const std::vector<std::pair<std::string, std::string>>&) override { status = s; return true; }
    bool write(std::string_view b) override { body += b; return true; }
    void end_stream() override {}
    bool client_gone() const override { return false; }
};

std::string base64(std::string_view bytes) {
    static constexpr char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (std::size_t i = 0; i + 3 <= bytes.size(); i += 3) { // callers pass a multiple of three bytes
        const unsigned v = (static_cast<unsigned>(static_cast<unsigned char>(bytes[i])) << 16U) |
                           (static_cast<unsigned>(static_cast<unsigned char>(bytes[i + 1])) << 8U) |
                           static_cast<unsigned>(static_cast<unsigned char>(bytes[i + 2]));
        out += {table[v >> 18U], table[(v >> 12U) & 63U], table[(v >> 6U) & 63U], table[v & 63U]};
    }
    return out;
}
// A header-only PNG (33 bytes): the media layer reads the size, the fake adapter never decodes.
std::string png(unsigned w, unsigned h) {
    std::string s;
    const auto put = [&](std::initializer_list<unsigned> bytes) {
        for (const auto b : bytes) s.push_back(static_cast<char>(static_cast<unsigned char>(b)));
    };
    put({0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n', 0, 0, 0, 13, 'I', 'H', 'D', 'R'});
    put({w >> 24U, (w >> 16U) & 255U, (w >> 8U) & 255U, w & 255U});
    put({h >> 24U, (h >> 16U) & 255U, (h >> 8U) & 255U, h & 255U});
    put({8, 2, 0, 0, 0, 0, 0, 0, 0});
    return s;
}
std::string data_url(unsigned w, unsigned h) { return "data:image/png;base64," + base64(png(w, h)); }

// The adapter of the test: rows = width x height of the file, spliced between two real ids.
struct FakeAdapter : sv::ApiMediaAdapter {
    explicit FakeAdapter(std::string name = "fake-tower-v1", std::string media_kind = "image", std::string mark = "\x1f",
                         std::uint32_t capacity = ss::kEmbedMaxHandles)
        : interner(capacity), name_(std::move(name)), kind_(std::move(media_kind)), mark_(std::move(mark)) {}
    std::string kind() const override { return kind_; }
    std::string identity() const override { return name_; }
    std::string placeholder() const override { return mark_; }
    std::uint32_t max_rows() const override { return 64; }
    std::uint64_t epoch() const override { return epoch_; }
    sv::ApiMediaSpan encode(const sv::media::MediaItem& item, std::uint32_t row_budget) override {
        ++calls;
        last_budget = row_budget;
        const auto rows = item.width * item.height;
        if (rows > row_budget) throw sv::media::MediaError(400, "image needs more rows than the request may spend", "");
        if (item.width == 5) throw std::invalid_argument("image shape is not representable");
        ss::EmbedRows r;
        r.digest = item.digest;
        r.rows = rows;
        r.width = 2;
        r.real_token_id = '#';
        r.data.assign(static_cast<std::size_t>(rows) * 2U, 0x3f80);
        sv::ApiMediaSpan span;
        span.cache_hit = !seen_.insert(item.digest).second;
        if (split) others.push_back(std::make_unique<ss::EmbedInterner>());
        span.lease = (split ? *others.back() : interner).intern(std::move(r));
        span.layout.push_back('<');
        for (const auto id : span.lease.ids()) span.layout.push_back(id);
        span.layout.push_back('>');
        span.grid_h = item.height;
        span.grid_w = item.width;
        span.preprocess_ms = 0.25;
        span.encode_ms = span.cache_hit ? 0.0 : 1.5;
        return span;
    }
    ss::EmbedInterner interner;
    std::vector<std::unique_ptr<ss::EmbedInterner>> others;
    bool split = false; // a new handle space per item: what two adapters with their own interners would do
    int calls = 0;
    std::uint32_t last_budget = 0;
    std::uint64_t epoch_ = 0;

private:
    std::string name_, kind_, mark_;
    std::set<sv::media::Digest> seen_;
};

// Text of a message the way the family renderers read it: strings and text parts; any other part is refused
// with the text the shipped renderers use.
std::string render(const sv::ChatTemplateInput& in) {
    std::string out;
    for (const auto& message : in.messages.as_array()) {
        const auto* content = message.find("content");
        if (!content) continue;
        if (content->is_string()) { out += content->as_string(); continue; }
        if (!content->is_array()) continue;
        for (const auto& part : content->as_array()) {
            const auto* type = part.find("type");
            const auto* text = part.find("text");
            if (type && type->is_string() && type->as_string() == "text" && text && text->is_string()) out += text->as_string();
            else throw sv::ChatTemplateError("Image input is not supported by this engine.");
        }
    }
    return out;
}

struct Service {
    explicit Service(std::vector<std::shared_ptr<sv::ApiMediaAdapter>> adapters = {}, std::uint32_t request_rows = 0,
                     std::function<void(std::uint64_t)> on_epoch = {})
        : engine(backend), runtime(queue, engine, engine.store, {}, hooks()) {
        sv::ApiConfig config;
        config.model_id = "skeleton";
        config.sampling_policy = "reject";
        config.stop_tokens = {skeleton::kEos};
        config.default_thinking = false;
        config.chat_renderer = render;
        config.vision.adapters = std::move(adapters);
        config.vision.max_request_rows = request_rows;
        config.vision.media.limits.max_images = 3;
        config.vision.on_epoch = std::move(on_epoch);
        runtime.configure_api(config);
        api = std::make_unique<sv::ApiService>(std::move(config),
            sv::ApiTokenizerBinding(std::make_shared<skeleton::ByteTokenizer>()), queue, skeleton::kContext, false);
        owner = std::thread([this] { runtime.run(); });
    }
    ~Service() {
        runtime.shutdown();
        owner.join();
    }
    Writer call(const std::string& method, const std::string& path, const std::string& body = {}) {
        sv::HttpRequest request;
        request.method = method;
        request.path = path;
        request.target = path;
        request.body = body;
        Writer writer;
        api->handle(request, writer);
        return writer;
    }
    Writer post(const std::string& path, const std::string& body) { return call("POST", path, body); }
    static sv::SessionHooks hooks() {
        sv::SessionHooks h;
        h.identity = {"skeleton-weights-v1", "skeleton-profile-v1", "skeleton-state-v1", 1};
        return h;
    }
    ss::CpuBackend backend;
    skeleton::Engine engine;
    sv::GenerationQueue queue;
    sv::SessionRuntime runtime;
    std::unique_ptr<sv::ApiService> api;
    std::thread owner;
};

// Members that differ between two runs of the same request (ids, clocks) are dropped before a comparison.
sv::Json scrub(const sv::Json& value) {
    if (value.is_array()) {
        sv::Json out = sv::Json::array();
        for (const auto& item : value.as_array()) out.push(scrub(item));
        return out;
    }
    if (!value.is_object()) return value;
    sv::Json out = sv::Json::object();
    for (const auto& [key, member] : value.as_object()) {
        if (key == "id" || key == "created" || key == "created_at" || key == "completed_at" || key == "request_id" ||
            key == "timings" || key == "timing" || key == "call_id" || key == "item_id" || key == "response_id")
            continue;
        if (key == "queue_ms" || key == "prefill_ms" || key == "decode_ms" || key == "decode_tokens_per_second")
            continue;
        out.set(key, scrub(member));
    }
    return out;
}
std::string stable(const Writer& w) {
    try { return std::to_string(w.status) + " " + scrub(sv::parse_json(w.body)).dump(); }
    catch (const std::exception&) { return std::to_string(w.status) + " " + w.body; }
}
const sv::Json& at(const sv::Json& value, std::initializer_list<const char*> path) {
    static const sv::Json none;
    const sv::Json* current = &value;
    for (const auto* key : path) {
        current = current->find(key);
        if (!current) return none;
    }
    return *current;
}
double number(const sv::Json& value, std::initializer_list<const char*> path) {
    const auto& v = at(value, path);
    return v.is_number() ? v.as_double() : -1.0;
}
std::size_t embed_ids(const std::vector<std::uint32_t>& ids) { return ss::count_embed_ids(ids); }
// The events of an SSE body in order: "event name|scrubbed data" per event, ids and clocks removed.
std::vector<std::string> sse_events(const std::string& body) {
    std::vector<std::string> out;
    std::string name;
    std::size_t at_line = 0;
    while (at_line < body.size()) {
        auto end = body.find('\n', at_line);
        if (end == std::string::npos) end = body.size();
        std::string line = body.substr(at_line, end - at_line);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        at_line = end + 1;
        if (line.rfind("event:", 0) == 0) name = line.substr(6);
        else if (line.rfind("data:", 0) == 0) {
            auto data = line.substr(5);
            if (!data.empty() && data[0] == ' ') data.erase(0, 1);
            try { data = scrub(sv::parse_json(data)).dump(); } catch (const std::exception&) {}
            out.push_back(name + "|" + data);
            name.clear();
        }
    }
    return out;
}

std::string chat_body(const std::string& parts, const std::string& more = {}) {
    return R"({"model":"skeleton","max_tokens":4)" + more + R"(,"messages":[{"role":"user","content":[)" + parts + "]}]}";
}
std::string image_part(unsigned w, unsigned h) {
    return R"({"type":"image_url","image_url":{"url":")" + data_url(w, h) + R"("}})";
}

// The comparison must retain generated content, token ids, finish reasons, usage and event order.
void sse_comparison_contract() {
    const std::string original = R"({"id":"a","created":1,"delta":"hello","token_ids":[1,2],"finish_reason":"stop","usage":{"total_tokens":2}})";
    const auto stream = [](const std::string& data) { return "event: delta\ndata: " + data + "\n\nevent: done\ndata: [DONE]\n\n"; };
    const auto baseline = sse_events(stream(original));
    auto changed = sv::parse_json(original);
    changed.set("id", sv::Json("b")); changed.set("created", sv::Json(2));
    CHECK(sse_events(stream(changed.dump())) == baseline);
    for (const auto* field : {"delta", "token_ids", "finish_reason", "usage"}) {
        changed = sv::parse_json(original); changed.set(field, sv::Json("changed"));
        CHECK(sse_events(stream(changed.dump())) != baseline);
    }
    CHECK(sse_events("event: done\ndata: [DONE]\n\nevent: delta\ndata: " + original + "\n\n") != baseline);
    CHECK(sse_events("event: delta\ndata: " + original + "\n\n") != baseline);
}

void prepared_prompt_lifetime() {
    sv::ApiChatPrompt prepared;
    {
        auto adapter = std::make_shared<FakeAdapter>("small-interner", "image", "\x1f", 2);
        Service service({adapter});
        auto body = sv::parse_json(chat_body(image_part(2, 2)));
        body.set("tool_choice", sv::Json("none"));
        prepared = service.api->prepare_chat_prompt(body);
        CHECK(prepared.tools_off && prepared.embeds && embed_ids(prepared.tokens) == 4U);
        // Force repeated wrap/reuse of the other handle while the returned table pins this image.
        for (unsigned i = 1; i <= 8; ++i) {
            ss::EmbedRows rows; rows.digest[0] = static_cast<std::uint8_t>(i);
            rows.rows = 1; rows.width = 1; rows.data = {0x4000};
            const auto other = adapter->interner.intern(std::move(rows));
            CHECK(other.handle() != prepared.embeds->leases()[0].handle());
            prepared.embeds->validate(prepared.tokens);
        }
        CHECK(adapter->interner.epoch() > 0);
    }
    // No service, adapter, or interner remains: only the public result owns the rows.
    CHECK(static_cast<bool>(prepared.embeds));
    if (prepared.embeds) {
        prepared.embeds->validate(prepared.tokens);
        const auto* rows = prepared.embeds->find(prepared.embeds->leases()[0].handle());
        CHECK(rows && rows->rows == 4 && rows->width == 2 && rows->data == std::vector<std::uint16_t>(8, 0x3f80));
    }
}

// 1. No adapter: every front door answers what it answered before the media layer existed.
void without_adapter() {
    Service s;
    const auto chat = s.post("/v1/chat/completions", chat_body(R"({"type":"text","text":"hi"},)" + image_part(2, 2)));
    CHECK(chat.status == 400);
    CHECK(chat.body.find("Image input is not supported by this engine.") != std::string::npos);
    const auto messages = s.post("/v1/messages",
        R"({"model":"skeleton","max_tokens":4,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":")" +
            base64(png(2, 2)) + R"("}}]}]})");
    CHECK(messages.status == 400);
    CHECK(messages.body.find("unsupported content block type image") != std::string::npos);
    const auto responses = s.post("/v1/responses",
        R"({"model":"skeleton","max_output_tokens":4,"input":[{"role":"user","content":[{"type":"input_image","image_url":")" + data_url(2, 2) + R"("}]}]})");
    CHECK(responses.status == 400);
    CHECK(responses.body.find("only plaintext parts are supported") != std::string::npos);
    const auto caps = sv::parse_json(s.call("GET", "/v1/capabilities").body);
    CHECK(at(caps, {"image_input"}).is_bool() && !at(caps, {"image_input"}).as_bool());
    CHECK(!caps.find("vision"));
    CHECK(s.call("GET", "/v1/models").body.find("media_adapters") == std::string::npos);
}

// 2. Text requests: the same bytes with and without a loaded adapter.
void text_is_unchanged() {
    Service plain;
    Service vision({std::make_shared<FakeAdapter>()});
    const std::vector<std::pair<std::string, std::string>> bodies = {
        {"/v1/chat/completions", R"({"model":"skeleton","max_tokens":4,"messages":[{"role":"user","content":"hi"}]})"},
        {"/v1/chat/completions", R"({"model":"skeleton","max_tokens":4,"messages":[{"role":"system","content":"be brief"},{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]})"},
        {"/v1/chat/completions", R"({"model":"skeleton","max_tokens":3,"stream":true,"messages":[{"role":"user","content":"stream"}]})"},
        {"/v1/messages", R"({"model":"skeleton","max_tokens":4,"messages":[{"role":"user","content":"hi"}]})"},
        {"/v1/messages", R"({"model":"skeleton","max_tokens":4,"system":"s","messages":[{"role":"user","content":[{"type":"text","text":"block"}]}]})"},
        {"/v1/responses", R"({"model":"skeleton","max_output_tokens":4,"input":"hi"})"},
        {"/v1/responses", R"({"model":"skeleton","max_output_tokens":4,"input":[{"role":"user","content":[{"type":"input_text","text":"parts"}]}]})"},
        {"/v1/messages", R"({"model":"skeleton","max_tokens":3,"stream":true,"messages":[{"role":"user","content":"stream"}]})"},
        {"/v1/responses", R"({"model":"skeleton","max_output_tokens":3,"stream":true,"input":"stream"})"},
        // the placeholder byte in plain text is text while the request holds no media
        {"/v1/chat/completions", std::string(R"({"model":"skeleton","max_tokens":4,"messages":[{"role":"user","content":"a\u001fb"}]})")},
    };
    for (const auto& [path, body] : bodies) {
        const auto a = plain.post(path, body), b = vision.post(path, body);
        CHECK(a.status == 200);
        if (body.find("\"stream\":true") == std::string::npos) {
            if (stable(a) != stable(b)) std::cerr << path << "\n" << stable(a) << "\n" << stable(b) << "\n";
            CHECK(stable(a) == stable(b));
        } else {
            // the same events in the same order with the same payloads
            const auto ea = sse_events(a.body), eb = sse_events(b.body);
            CHECK(b.status == 200);
            CHECK(ea.size() >= 2U);
            if (ea != eb) {
                std::cerr << path << ": " << ea.size() << " vs " << eb.size() << " events\n";
                for (std::size_t i = 0; i < std::min(ea.size(), eb.size()); ++i)
                    if (ea[i] != eb[i]) { std::cerr << ea[i] << "\n" << eb[i] << "\n"; break; }
            }
            CHECK(ea == eb);
        }
        CHECK(b.body.find("\"vision\"") == std::string::npos);
        if (path == "/v1/chat/completions") {
            const auto json = sv::parse_json(body);
            CHECK(plain.api->chat_prompt(json) == vision.api->chat_prompt(json));
            const auto ta = plain.post("/tokenize", body), tb = vision.post("/tokenize", body);
            CHECK(ta.status == 200 && ta.body == tb.body); // byte-equal: no id, no clock
            CHECK(tb.body.find("images") == std::string::npos);
        }
    }
    // old refusals outside the accepted turns stay the renderer's
    const auto system = vision.post("/v1/chat/completions",
        R"({"model":"skeleton","max_tokens":4,"messages":[{"role":"system","content":[)" + image_part(2, 2) + R"(]},{"role":"user","content":"hi"}]})");
    CHECK(system.status == 400);
    CHECK(system.body.find("Image input is not supported by this engine.") != std::string::npos);
}

// 3. Capability and model listing carry the adapter identity; the hash moves with it.
void capabilities() {
    Service plain;
    Service one({std::make_shared<FakeAdapter>("fake-tower-v1")});
    Service two({std::make_shared<FakeAdapter>("fake-tower-v2")});
    Service both({std::make_shared<FakeAdapter>("fake-tower-v1"), std::make_shared<FakeAdapter>("fake-sound-v1", "audio", "\x1e")});
    const auto doc = [](Service& s) { return sv::parse_json(s.call("GET", "/v1/capabilities").body); };
    const auto a = doc(plain), b = doc(one), c = doc(two), d = doc(both);
    CHECK(at(b, {"image_input"}).as_bool());
    CHECK(at(b, {"vision", "adapter"}).as_string() == "fake-tower-v1");
    CHECK(at(b, {"vision", "remote_fetch"}).as_string() == "off");
    CHECK(number(b, {"vision", "max_items"}) == 3.0);
    CHECK(number(b, {"vision", "max_rows_per_item"}) == 64.0);
    const auto id = [](const sv::Json& j) { return at(j, {"capabilities_id"}).as_string(); };
    CHECK(id(a) != id(b) && id(b) != id(c) && id(a) != id(c) && id(d) != id(b));
    // another media kind is one more registered adapter: its own capability and member, no front door edit
    CHECK(at(d, {"audio_input"}).as_bool());
    CHECK(at(d, {"audio", "adapter"}).as_string() == "fake-sound-v1");
    const auto models = sv::parse_json(one.call("GET", "/v1/models").body);
    CHECK(at(models.find("data")->as_array()[0], {"media_adapters", "image"}).as_string() == "fake-tower-v1");
    CHECK(at(models.find("data")->as_array()[0], {"capabilities_id"}).as_string() == id(b));
    // bad registrations fail at construction
    bool refused = false;
    try { Service bad({std::make_shared<FakeAdapter>("x", "image", "two bytes")}); } catch (const std::invalid_argument&) { refused = true; }
    CHECK(refused);
    refused = false;
    try { Service bad({std::make_shared<FakeAdapter>("x"), std::make_shared<FakeAdapter>("y")}); } catch (const std::invalid_argument&) { refused = true; }
    CHECK(refused);
}

// 4. The three front doors with an image: virtual ids reach the engine, usage counts them, the receipt says so.
void images() {
    auto adapter = std::make_shared<FakeAdapter>();
    Service s({adapter});
    const auto text_only = sv::parse_json(chat_body(R"({"type":"text","text":"hi"})"));
    const auto with_image = chat_body(R"({"type":"text","text":"hi"},)" + image_part(2, 2));
    const auto base = s.api->chat_prompt(text_only);
    // the vector-only helper refuses media: virtual ids are returned only together with their table
    bool refused = false;
    try { (void)s.api->chat_prompt(sv::parse_json(with_image)); } catch (const std::invalid_argument&) { refused = true; }
    CHECK(refused);
    const auto prepared = s.api->prepare_chat_prompt(sv::parse_json(with_image));
    CHECK(prepared.embeds && prepared.embeds->size() == 1U);
    if (prepared.embeds) prepared.embeds->validate(prepared.tokens); // the returned table resolves the returned ids
    const auto& ids = prepared.tokens;
    CHECK(ids.size() == base.size() + 6U); // '<' + 4 rows + '>'
    CHECK(embed_ids(ids) == 4U);
    CHECK(ids[base.size()] == '<' && ids.back() == '>' && ss::is_embed_id(ids[base.size() + 1U]));
    CHECK(ss::embed_row(ids[base.size() + 4U]) == 3U);
    CHECK(s.api->prepare_chat_prompt(sv::parse_json(with_image)).tokens == ids); // same image -> same ids
    const auto other = s.api->prepare_chat_prompt(sv::parse_json(chat_body(R"({"type":"text","text":"hi"},)" + image_part(1, 4)))).tokens;
    CHECK(embed_ids(other) == 4U && other != ids); // another image -> another handle

    const auto chat = s.post("/v1/chat/completions", with_image);
    if (chat.status != 200) std::cerr << chat.body << "\n";
    CHECK(chat.status == 200);
    const auto chat_json = sv::parse_json(chat.body);
    CHECK(number(chat_json, {"usage", "prompt_tokens"}) == static_cast<double>(ids.size()));
    CHECK(at(chat_json, {"strix", "vision", "adapter"}).as_string() == "fake-tower-v1");
    CHECK(number(chat_json, {"strix", "vision", "rows"}) == 4.0);
    const auto* listed = at(chat_json, {"strix", "vision"}).find("images");
    CHECK(listed && listed->is_array() && listed->size() == 1U);
    if (listed && listed->size() == 1U) {
        const auto& image = listed->as_array()[0];
        CHECK(number(image, {"rows"}) == 4.0 && number(image, {"tokens"}) == 6.0);
        CHECK(image.find("grid")->as_array()[0].as_double() == 2.0 && image.find("grid")->as_array()[1].as_double() == 2.0);
        CHECK(image.find("cache_hit")->as_bool()); // prepare_chat_prompt above encoded it first
        CHECK(image.find("digest")->as_string().size() == 16U);
        CHECK(image.find("encode_ms") && image.find("preprocess_ms"));
    }

    // two images in one turn, one of them twice: order is prompt order
    const auto two = s.post("/v1/chat/completions", chat_body(image_part(2, 2) + R"(,{"type":"text","text":"x"},)" + image_part(3, 1)));
    CHECK(two.status == 200);
    const auto two_json = sv::parse_json(two.body);
    CHECK(number(two_json, {"strix", "vision", "rows"}) == 7.0);
    CHECK(number(two_json, {"usage", "prompt_tokens"}) == 1.0 + 6.0 + 5.0);

    // streaming carries the receipt as well
    const auto stream = s.post("/v1/chat/completions", chat_body(image_part(2, 2), R"(,"stream":true)"));
    CHECK(stream.status == 200);
    CHECK(stream.body.find("\"vision\"") != std::string::npos && stream.body.find("fake-tower-v1") != std::string::npos);

    // Messages: an image block, and an image inside a tool result
    const std::string block = R"({"type":"image","source":{"type":"base64","media_type":"image/png","data":")" + base64(png(2, 2)) + R"("}})";
    const auto messages = s.post("/v1/messages",
        R"({"model":"skeleton","max_tokens":4,"messages":[{"role":"user","content":[{"type":"text","text":"hi"},)" + block + "]}]}");
    if (messages.status != 200) std::cerr << messages.body << "\n";
    CHECK(messages.status == 200);
    const auto messages_json = sv::parse_json(messages.body);
    CHECK(number(messages_json, {"strix", "vision", "rows"}) == 4.0);
    CHECK(number(messages_json, {"usage", "input_tokens"}) == static_cast<double>(ids.size()));
    const auto tool = s.post("/v1/messages",
        R"({"model":"skeleton","max_tokens":4,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"see"},)" +
            block + "]}]}]}");
    if (tool.status != 200) std::cerr << tool.body << "\n";
    CHECK(tool.status == 200);
    CHECK(number(sv::parse_json(tool.body), {"strix", "vision", "rows"}) == 4.0);

    // Responses: input_image
    const auto responses = s.post("/v1/responses",
        R"({"model":"skeleton","max_output_tokens":4,"input":[{"role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_image","image_url":")" +
            data_url(2, 2) + R"("}]}]})");
    if (responses.status != 200) std::cerr << responses.body << "\n";
    CHECK(responses.status == 200);
    const auto responses_json = sv::parse_json(responses.body);
    CHECK(number(responses_json, {"strix", "vision", "rows"}) == 4.0);
    CHECK(number(responses_json, {"usage", "input_tokens"}) == static_cast<double>(ids.size()));

    // /tokenize: real ids only, the rows are listed
    const auto tokenize = s.post("/tokenize", with_image);
    CHECK(tokenize.status == 200);
    const auto tokenize_json = sv::parse_json(tokenize.body);
    CHECK(number(tokenize_json, {"count"}) == static_cast<double>(ids.size()));
    std::size_t marks = 0;
    for (const auto& id : tokenize_json.find("tokens")->as_array()) {
        CHECK(id.as_double() < 2147483648.0);
        if (id.as_double() == static_cast<double>('#')) ++marks;
    }
    CHECK(marks == 4U);
    const auto* images = tokenize_json.find("images");
    CHECK(images && images->size() == 1U);
    if (images && images->size() == 1U) {
        const auto& image = images->as_array()[0];
        CHECK(number(image, {"rows"}) == 4.0 && number(image, {"tokens"}) == 6.0);
        CHECK(number(image, {"offset"}) == static_cast<double>(base.size()));
        CHECK(image.find("type")->as_string() == "image");
    }
}

// 5. Refusals: forged parts, a placeholder in the text, limits, row budgets, a wrapped handle space.
void refusals() {
    auto adapter = std::make_shared<FakeAdapter>();
    Service s({adapter});
    const auto status = [&](const std::string& body) { return s.post("/v1/chat/completions", body).status; };
    // the placeholder byte in the text of a request that holds an image
    const auto mark = s.post("/v1/chat/completions", chat_body(R"({"type":"text","text":"a\u001fb"},)" + image_part(2, 2)));
    CHECK(mark.status == 400 && mark.body.find("reserved media placeholder") != std::string::npos);
    // a canonical part sent by the client: in an accepted turn the media layer refuses it, elsewhere the wiring does
    CHECK(status(chat_body(R"({"type":"image","media":0})")) == 400);
    CHECK(status(R"({"model":"skeleton","max_tokens":4,"messages":[{"role":"assistant","content":[{"type":"image","media":0}]},{"role":"user","content":[)" +
                 image_part(2, 2) + "]}]}") == 400);
    // limits of the media layer (3 images in this service) and of the context (256 rows on this engine)
    CHECK(status(chat_body(image_part(1, 1) + "," + image_part(1, 2) + "," + image_part(2, 1) + "," + image_part(2, 2))) == 400);
    const auto big = s.post("/v1/chat/completions", chat_body(image_part(8, 9))); // 72 rows > max_rows 64
    CHECK(big.status == 400 && big.body.find("more rows") != std::string::npos);
    CHECK(adapter->last_budget == 64U);
    const auto shape = s.post("/v1/chat/completions", chat_body(image_part(5, 1)));
    CHECK(shape.status == 400 && shape.body.find("not representable") != std::string::npos &&
          shape.body.find("messages[0].content[0]") != std::string::npos);
    CHECK(status(chat_body(image_part(8, 8) + "," + image_part(8, 8))) == 200); // one digest, 2 x 66 ids fit
    const std::string filler(120, 'x');
    const auto full = s.post("/v1/chat/completions", chat_body(R"({"type":"text","text":")" + filler + R"("},)" + image_part(8, 8) + "," + image_part(7, 9) + "," + image_part(9, 7)));
    CHECK(full.status == 400); // 120 + 3 x 65 ids do not fit 256
    // remote URLs stay off
    CHECK(status(chat_body(R"({"type":"image_url","image_url":{"url":"https://example.com/a.png"}})")) == 400);

    Service capped({std::make_shared<FakeAdapter>()}, 5);
    CHECK(capped.post("/v1/chat/completions", chat_body(image_part(2, 2))).status == 200);
    CHECK(capped.post("/v1/chat/completions", chat_body(image_part(2, 2) + "," + image_part(1, 2))).status == 400);

    // two handle spaces hand out the same handle for different content: refused, never resolved to the wrong rows
    auto divided = std::make_shared<FakeAdapter>();
    divided->split = true;
    Service two_spaces({divided});
    CHECK(two_spaces.post("/v1/chat/completions", chat_body(image_part(2, 2))).status == 200);
    const auto mixed = two_spaces.post("/v1/chat/completions", chat_body(image_part(2, 2) + "," + image_part(1, 3)));
    CHECK(mixed.status == 500 && mixed.body.find("share one interner") != std::string::npos);

    // epoch: without an owner of the cached prefixes media requests stop; text goes on
    adapter->epoch_ = 1;
    CHECK(status(chat_body(image_part(2, 2))) == 503);
    CHECK(status(R"({"model":"skeleton","max_tokens":4,"messages":[{"role":"user","content":"hi"}]})") == 200);
    auto wrapped = std::make_shared<FakeAdapter>();
    std::vector<std::uint64_t> seen;
    Service owned({wrapped}, 0, [&](std::uint64_t epoch) { seen.push_back(epoch); });
    wrapped->epoch_ = 4;
    CHECK(owned.post("/v1/chat/completions", chat_body(image_part(2, 2))).status == 200);
    CHECK(owned.post("/v1/chat/completions", chat_body(image_part(2, 2))).status == 200);
    CHECK(seen.size() == 1U && seen[0] == 4U);
}
} // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif
    try {
        without_adapter();
        text_is_unchanged();
        sse_comparison_contract();
        prepared_prompt_lifetime();
        capabilities();
        images();
        refusals();
    } catch (const std::exception& e) {
        std::cerr << "FAILED: exception " << e.what() << "\n";
        return 1;
    }
    std::cout << "serve api vision tests: " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
