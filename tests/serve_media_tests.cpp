#include "strix/serve/media.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

using namespace strix::serve;
using namespace strix::serve::media;

namespace {

int failures = 0;

void check(bool ok, const char* what, int line) {
    if (ok) return;
    ++failures;
    std::cerr << "FAIL line " << line << ": " << what << '\n';
}
#define CHECK(expr) check((expr), #expr, __LINE__)

// Runs fn and reports the MediaError status (0 when nothing was thrown); param goes to *where.
template <typename Fn> int status_of(Fn&& fn, std::string* where = nullptr, std::string* message = nullptr) {
    try {
        fn();
    } catch (const MediaError& e) {
        if (where) *where = e.param;
        if (message) *message = e.what();
        return e.status;
    }
    return 0;
}

std::string base64(std::string_view bytes) {
    static constexpr char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    std::size_t i = 0;
    for (; i + 3 <= bytes.size(); i += 3) {
        const unsigned v = (static_cast<unsigned>(static_cast<unsigned char>(bytes[i])) << 16U) |
                           (static_cast<unsigned>(static_cast<unsigned char>(bytes[i + 1])) << 8U) |
                           static_cast<unsigned>(static_cast<unsigned char>(bytes[i + 2]));
        out += {table[v >> 18U], table[(v >> 12U) & 63U], table[(v >> 6U) & 63U], table[v & 63U]};
    }
    if (const auto rest = bytes.size() - i; rest == 1) {
        const unsigned v = static_cast<unsigned>(static_cast<unsigned char>(bytes[i])) << 16U;
        out += {table[v >> 18U], table[(v >> 12U) & 63U], '=', '='};
    } else if (rest == 2) {
        const unsigned v = (static_cast<unsigned>(static_cast<unsigned char>(bytes[i])) << 16U) |
                           (static_cast<unsigned>(static_cast<unsigned char>(bytes[i + 1])) << 8U);
        out += {table[v >> 18U], table[(v >> 12U) & 63U], table[(v >> 6U) & 63U], '='};
    }
    return out;
}

void put(std::string& s, std::initializer_list<unsigned> bytes) {
    for (const auto b : bytes) s.push_back(static_cast<char>(static_cast<unsigned char>(b)));
}
void be32(std::string& s, unsigned v) { put(s, {v >> 24U, (v >> 16U) & 255U, (v >> 8U) & 255U, v & 255U}); }

// Header-only files: the normalizer reads sizes, it never decodes.
std::string png(unsigned w, unsigned h) {
    std::string s;
    put(s, {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n', 0, 0, 0, 13, 'I', 'H', 'D', 'R'});
    be32(s, w);
    be32(s, h);
    put(s, {8, 2, 0, 0, 0, 0, 0, 0, 0});
    return s;
}
std::string jpeg(unsigned w, unsigned h, unsigned sof = 0xC0) {
    std::string s;
    put(s, {0xFF, 0xD8, 0xFF, 0xE0, 0, 16, 'J', 'F', 'I', 'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0});
    put(s, {0xFF, 0xC4, 0, 4, 0, 0});       // a table segment whose marker is in the SOF range
    put(s, {0xFF, 0xFF, 0xFF, sof, 0, 11, 8}); // fill bytes before the frame header
    put(s, {h >> 8U, h & 255U, w >> 8U, w & 255U, 1, 1, 0x11, 0});
    return s;
}
std::string webp(std::string_view kind, unsigned w, unsigned h) {
    std::string s = "RIFF";
    put(s, {0, 0, 0, 0});
    s += "WEBP";
    s += kind;
    put(s, {0, 0, 0, 0});
    if (kind == "VP8 ") {
        put(s, {0x10, 0, 0, 0x9D, 0x01, 0x2A, w & 255U, w >> 8U, h & 255U, h >> 8U});
    } else if (kind == "VP8L") {
        const unsigned packed = (w - 1U) | ((h - 1U) << 14U);
        put(s, {0x2F, packed & 255U, (packed >> 8U) & 255U, (packed >> 16U) & 255U, packed >> 24U});
    } else {
        put(s, {0, 0, 0, 0, (w - 1U) & 255U, ((w - 1U) >> 8U) & 255U, (w - 1U) >> 16U, (h - 1U) & 255U, ((h - 1U) >> 8U) & 255U, (h - 1U) >> 16U});
    }
    return s;
}

std::string data_url(std::string_view bytes, std::string_view type = "image/png") {
    return "data:" + std::string(type) + ";base64," + base64(bytes);
}
std::string quoted(std::string_view s) { return Json(s).dump(); }

std::string chat_body(const std::string& parts, std::string_view role = "user") {
    return R"({"model":"m","messages":[{"role":"system","content":"s"},{"role":)" + quoted(role) + R"(,"content":[)" + parts + "]}]}";
}
std::string chat_image(const std::string& url) { return R"({"type":"image_url","image_url":{"url":)" + quoted(url) + "}}"; }

void digests() {
    CHECK(hex(sha256("")) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(hex(sha256("abc")) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(hex(sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(hex(sha256(std::string(1000, 'a'))) == "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
}

void base64_cases() {
    CHECK(decode_base64("aGVsbG8=", 100) == "hello");
    CHECK(decode_base64("aGVsbG8", 100) == "hello");
    CHECK(decode_base64("aGVs\r\nbG8h\n", 100) == "hello!");
    CHECK(decode_base64("/+8=", 100) == std::string("\xFF\xEF", 2));
    for (const auto& bytes : {std::string("a"), std::string("ab"), std::string("abc"), std::string("\0\x01\xFE\xFF", 4)})
        CHECK(decode_base64(base64(bytes), 100) == bytes);
    CHECK(status_of([] { (void)decode_base64("aGVs*G8=", 100); }) == 400);
    CHECK(status_of([] { (void)decode_base64("aGVsb", 100); }) == 400);      // one dangling symbol
    CHECK(status_of([] { (void)decode_base64("aGVsbG8==", 100); }) == 400);  // wrong padding count
    CHECK(status_of([] { (void)decode_base64("aGVsbG8=x", 100); }) == 400);  // data after padding
    CHECK(status_of([] { (void)decode_base64("a===", 100); }) == 400);
    CHECK(status_of([] { (void)decode_base64("aGVs-G8_", 100); }) == 400);   // url-safe alphabet is not accepted
    CHECK(status_of([] { (void)decode_base64("", 100); }) == 400);
    CHECK(status_of([] { (void)decode_base64(" \n", 100); }) == 400);
    CHECK(decode_base64("aGVsbG8=", 5) == "hello");
    CHECK(status_of([] { (void)decode_base64("aGVsbG8=", 4); }) == 413);
}

void probes() {
    auto info = probe_image(png(640, 480));
    CHECK(info.format == Format::png && info.width == 640 && info.height == 480);
    info = probe_image(jpeg(1920, 1080));
    CHECK(info.format == Format::jpeg && info.width == 1920 && info.height == 1080);
    info = probe_image(jpeg(33, 65535, 0xC2));
    CHECK(info.format == Format::jpeg && info.width == 33 && info.height == 65535);
    info = probe_image(webp("VP8 ", 300, 200));
    CHECK(info.format == Format::webp && info.width == 300 && info.height == 200);
    info = probe_image(webp("VP8L", 16384, 1));
    CHECK(info.format == Format::webp && info.width == 16384 && info.height == 1);
    info = probe_image(webp("VP8X", 70000, 9));
    CHECK(info.format == Format::webp && info.width == 70000 && info.height == 9);
    CHECK(media_type(Format::png) == "image/png" && media_type(Format::jpeg) == "image/jpeg" && media_type(Format::webp) == "image/webp");

    CHECK(status_of([] { (void)probe_image("GIF89a\x01\x00\x01\x00"); }) == 400);
    CHECK(status_of([] { (void)probe_image("<svg xmlns='http://www.w3.org/2000/svg'/>"); }) == 400);
    CHECK(status_of([] { (void)probe_image(""); }) == 400);
    CHECK(status_of([] { (void)probe_image(png(0, 10)); }) == 400);
    CHECK(status_of([] { (void)probe_image(jpeg(10, 0)); }) == 400);
    // Every cut before the size field is complete is refused; trailing = bytes after that field.
    const std::vector<std::pair<std::string, std::size_t>> files = {{png(8, 8), 9}, {jpeg(8, 8), 4}, {webp("VP8 ", 8, 8), 0}, {webp("VP8L", 8, 8), 0}, {webp("VP8X", 8, 8), 0}};
    for (const auto& [file, trailing] : files) {
        for (std::size_t cut = 1; cut < file.size() - trailing; ++cut)
            CHECK(status_of([&, cut] { (void)probe_image(std::string_view(file).substr(0, cut)); }) == 400);
        CHECK(status_of([&] { (void)probe_image(std::string_view(file).substr(0, file.size() - trailing)); }) == 0);
    }
    std::string no_frame;
    put(no_frame, {0xFF, 0xD8, 0xFF, 0xDA, 0, 4, 0, 0});
    CHECK(status_of([&] { (void)probe_image(no_frame); }) == 400);
    std::string wrong_start = webp("VP8 ", 8, 8);
    wrong_start[23] = 0;
    CHECK(status_of([&] { (void)probe_image(wrong_start); }) == 400);
}

void addresses() {
    for (const char* ok : {"8.8.8.8", "1.1.1.1", "172.15.0.1", "172.32.0.1", "100.63.255.255", "100.128.0.1", "192.167.1.1", "223.255.255.255",
                           "2606:4700:4700::1111", "[2a00:1450:4001:81b::200e]", "2001:4860:4860:0:0:0:0:8888", "::ffff:8.8.8.8", "64:ff9b::8.8.8.8"})
        CHECK(address_is_public(ok));
    for (const char* no : {"127.0.0.1", "127.255.255.254", "0.0.0.0", "10.0.0.1", "172.16.0.1", "172.31.255.255", "192.168.1.1", "169.254.169.254",
                           "100.64.0.1", "100.127.255.255", "192.0.0.8", "192.0.2.1", "198.18.0.1", "198.51.100.7", "203.0.113.9", "224.0.0.1",
                           "255.255.255.255", "::", "::1", "[::1]", "fc00::1", "fd12:3456::1", "fe80::1", "fec0::1", "ff02::1", "::ffff:127.0.0.1",
                           "::ffff:7f00:1", "::ffff:10.1.2.3", "::127.0.0.1", "64:ff9b::10.0.0.1", "2002:7f00:1::1", "2002:c0a8:101::", "2001:db8::1",
                           "2001:0:1::1", "fe80::1%eth0", "", "1.2.3", "1.2.3.4.5", "01.2.3.4", "256.1.1.1", "1.2.3.4 ", "0x7f.0.0.1", "2130706433",
                           "1:2:3:4:5:6:7", "1:2:3:4:5:6:7:8:9", "1::2::3", "12345::1", ":1::2", "1:", "g::1", "example.com"})
        CHECK(!address_is_public(no));
}

void policies() {
    CHECK(parse_fetch_policy("", "").mode == FetchMode::off);
    CHECK(parse_fetch_policy("off", "a.example").mode == FetchMode::off && parse_fetch_policy("off", "a.example").hosts.empty());
    const auto policy = parse_fetch_policy("allowlist", " Img.Example.com ,cdn.example.org,,");
    CHECK(policy.mode == FetchMode::allowlist && policy.hosts == (std::vector<std::string>{"img.example.com", "cdn.example.org"}));
    const auto throws = [](std::string_view mode, std::string_view hosts) {
        try { (void)parse_fetch_policy(mode, hosts); } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    CHECK(throws("on", "a.example"));
    CHECK(throws("allowlist", ""));
    CHECK(throws("allowlist", "*.example.com"));
    CHECK(throws("allowlist", "https://a.example"));
    CHECK(throws("allowlist", "a.example:8443"));
    CHECK(Options{}.fetch.mode == FetchMode::off && Options{}.limits.max_images == 16 && Options{}.limits.max_image_bytes == 20U * 1024U * 1024U &&
          Options{}.limits.max_pixels == 40'000'000);
}

void text_only_is_untouched() {
    const std::vector<std::pair<Api, std::string>> bodies = {
        {Api::chat_completions, R"({"model":"m","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":null},{"role":"user","content":[{"type":"text","text":"a"}]}],"temperature":0.50})"},
        {Api::chat_completions, R"({"model":"m","messages":[{"role":"user","content":[{"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAA"}},{"type":"image"}]}]})"},
        {Api::chat_completions, R"({"model":"m","prompt":"x"})"},
        {Api::chat_completions, R"({"model":"m","messages":"oops"})"},
        {Api::messages, R"({"model":"m","system":[{"type":"text","text":"s"}],"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"r"}]},{"type":"tool_result","tool_use_id":"u","content":"plain"}]}]})"},
        {Api::responses, R"({"model":"m","input":"hello"})"},
        {Api::responses, R"({"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"a"}]},{"type":"function_call_output","call_id":"c","output":"done"}]})"},
    };
    for (const auto& [api, text] : bodies) {
        const auto body = parse_json(text);
        const auto out = normalize(body, api);
        CHECK(out.items.empty());
        CHECK(out.body.dump() == body.dump());
        CHECK(out.body.dump() == text);
    }
}

void chat() {
    const auto first = png(64, 48), second = jpeg(800, 600);
    const auto text = chat_body(R"({"type":"text","text":"look"},)" + chat_image(data_url(first)) + R"(,{"type":"image_url","image_url":)" +
                                quoted(data_url(second, "image/png")) + R"(},{"type":"text","text":"end"})");
    const auto body = parse_json(text);
    const auto out = normalize(body, Api::chat_completions);
    CHECK(body.dump() == text); // the input value is not edited
    CHECK(out.body.dump() == chat_body(R"({"type":"text","text":"look"},{"type":"image","media":0},{"type":"image","media":1},{"type":"text","text":"end"})"));
    CHECK(out.items.size() == 2);
    if (out.items.size() == 2) {
        CHECK(out.items[0].bytes == first && out.items[0].format == Format::png && out.items[0].media_type == "image/png");
        CHECK(out.items[0].width == 64 && out.items[0].height == 48 && out.items[0].digest == sha256(first) && out.items[0].detail.empty());
        CHECK(out.items[0].param == "messages[1].content[1]");
        // The declared type said PNG; the magic bytes decide.
        CHECK(out.items[1].bytes == second && out.items[1].format == Format::jpeg && out.items[1].media_type == "image/jpeg");
        CHECK(out.items[1].param == "messages[1].content[2]");
    }

    const auto detailed = normalize(parse_json(chat_body(R"({"type":"image_url","image_url":{"url":)" + quoted(data_url(first)) + R"(,"detail":"low"}})", "tool")), Api::chat_completions);
    CHECK(detailed.items.size() == 1 && detailed.items[0].detail == "low");
    CHECK(normalize(parse_json(chat_body(chat_image("DATA:image/png;BASE64," + base64(first)))), Api::chat_completions).items.size() == 1);
    CHECK(normalize(parse_json(chat_body(chat_image("data:;base64," + base64(first)))), Api::chat_completions).items.size() == 1);

    // Same image twice: two items, one digest.
    const auto twice = normalize(parse_json(chat_body(chat_image(data_url(first)) + "," + chat_image(data_url(first)))), Api::chat_completions);
    CHECK(twice.items.size() == 2 && twice.items[0].digest == twice.items[1].digest);
    CHECK(normalize(parse_json(chat_body(chat_image(data_url(webp("VP8L", 9, 9))))), Api::chat_completions).items.at(0).digest != twice.items.at(0).digest);

    // Other roles keep the renderer's own handling.
    for (const char* role : {"system", "assistant", "developer"}) {
        const auto untouched = chat_body(chat_image(data_url(first)), role);
        const auto result = normalize(parse_json(untouched), Api::chat_completions);
        CHECK(result.items.empty() && result.body.dump() == untouched);
    }

    const auto fails = [](const std::string& parts, std::string* where = nullptr, const Options& options = {}) {
        return status_of([&] { (void)normalize(parse_json(chat_body(parts)), Api::chat_completions, options); }, where);
    };
    std::string where;
    CHECK(fails(chat_image("https://example.com/a.png"), &where) == 400 && where == "messages[1].content[0].image_url.url");
    CHECK(fails(chat_image("http://example.com/a.png")) == 400);
    CHECK(fails(chat_image("file:///etc/passwd")) == 400);
    CHECK(fails(chat_image("ftp://example.com/a.png")) == 400);
    CHECK(fails(chat_image("/tmp/a.png")) == 400);
    CHECK(fails(chat_image("")) == 400);
    CHECK(fails(chat_image("data:image/png;base64")) == 400);
    CHECK(fails(chat_image("data:image/svg+xml,%3Csvg/%3E")) == 400);
    CHECK(fails(chat_image("data:image/png;base64,!!!!")) == 400);
    CHECK(fails(chat_image(data_url("GIF89a\x01\x00\x01\x00", "image/gif"))) == 400);
    CHECK(fails(R"({"type":"image_url"})", &where) == 400 && where == "messages[1].content[0].image_url");
    CHECK(fails(R"({"type":"image_url","image_url":{"url":7}})") == 400);
    CHECK(fails(R"({"type":"image_url","image_url":{"url":)" + quoted(data_url(first)) + R"(,"detail":3}})") == 400);
    CHECK(fails(R"({"type":"text","text":"a"},{"type":"image","media":0})", &where) == 400 && where == "messages[1].content[1]");

    // Limits.
    std::string many;
    for (int i = 0; i < 17; ++i) many += (i ? "," : "") + chat_image(data_url(first));
    CHECK(fails(many, &where) == 400 && where == "messages[1].content[16]");
    Options options;
    options.limits.max_images = 17;
    CHECK(fails(many, nullptr, options) == 0);
    options = {};
    options.limits.max_image_bytes = first.size();
    CHECK(fails(chat_image(data_url(first)), nullptr, options) == 0);
    options.limits.max_image_bytes = first.size() - 1;
    CHECK(fails(chat_image(data_url(first)), &where, options) == 413 && where == "messages[1].content[0].image_url.url");
    // The size comes from the header: a few bytes can claim a huge canvas.
    CHECK(fails(chat_image(data_url(png(8000, 5000)))) == 0);
    CHECK(fails(chat_image(data_url(png(8000, 5001))), &where) == 400 && where == "messages[1].content[0]");
    CHECK(fails(chat_image(data_url(png(0x7FFFFFFF, 0x7FFFFFFF)))) == 400);
    CHECK(fails(chat_image(data_url(webp("VP8X", 16777216, 16777216)))) == 400);
}

void messages() {
    const auto first = png(10, 20), second = webp("VP8 ", 30, 40), third = jpeg(50, 60);
    const std::string head = R"({"model":"m","max_tokens":8,"messages":[{"role":"assistant","content":[{"type":"text","text":"t"}]},{"role":"user","content":[)";
    const auto text = head + R"({"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":)" + quoted(base64(first)) +
                      R"(}},{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"shot"},{"type":"image","source":{"type":"url","url":)" +
                      quoted(data_url(second)) + R"(}}]},{"type":"image","source":{"type":"base64","media_type":"x","data":)" + quoted(base64(third)) + "}}]}]}";
    const auto out = normalize(parse_json(text), Api::messages);
    CHECK(out.body.dump() == head + R"({"type":"image","media":0},{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"shot"},{"type":"image","media":1}]},{"type":"image","media":2}]}]})");
    CHECK(out.items.size() == 3);
    if (out.items.size() == 3) {
        CHECK(out.items[0].bytes == first && out.items[0].media_type == "image/png" && out.items[0].param == "messages[1].content[0]");
        CHECK(out.items[1].bytes == second && out.items[1].format == Format::webp && out.items[1].width == 30 && out.items[1].height == 40);
        CHECK(out.items[1].param == "messages[1].content[1].content[1]");
        CHECK(out.items[2].bytes == third && out.items[2].format == Format::jpeg);
    }

    const auto wrap = [](const std::string& block, std::string_view role = "user") {
        return R"({"model":"m","messages":[{"role":)" + quoted(role) + R"(,"content":[)" + block + "]}]}";
    };
    const auto image = R"({"type":"image","source":{"type":"base64","media_type":"image/png","data":)" + quoted(base64(first)) + "}}";
    const auto assistant = wrap(image, "assistant");
    const auto untouched = normalize(parse_json(assistant), Api::messages);
    CHECK(untouched.items.empty() && untouched.body.dump() == assistant);
    // An OpenAI-shaped part is not a Messages image.
    const auto foreign = wrap(chat_image(data_url(first)));
    CHECK(normalize(parse_json(foreign), Api::messages).body.dump() == foreign);

    const auto fails = [&](const std::string& block, std::string* where = nullptr) {
        return status_of([&] { (void)normalize(parse_json(wrap(block)), Api::messages); }, where);
    };
    std::string where;
    CHECK(fails(R"({"type":"image"})", &where) == 400 && where == "messages[0].content[0].source");
    CHECK(fails(R"({"type":"image","source":{"type":"file","file_id":"f"}})", &where) == 400 && where == "messages[0].content[0].source.type");
    CHECK(fails(R"({"type":"image","source":{"type":"base64","media_type":"image/png"}})") == 400);
    CHECK(fails(R"({"type":"image","source":{"type":"base64","data":"data:image/png;base64,AAAA"}})", &where) == 400 && where == "messages[0].content[0].source.data");
    CHECK(fails(R"({"type":"image","source":{"type":"url","url":"https://example.com/a.png"}})", &where) == 400 && where == "messages[0].content[0].source.url");
    CHECK(fails(R"({"type":"image","media":0})") == 400);
    CHECK(fails(R"({"type":"tool_result","tool_use_id":"t","content":[{"type":"image","media":0}]})", &where) == 400 && where == "messages[0].content[0].content[0]");
}

void responses() {
    const auto first = png(10, 20), second = jpeg(30, 40);
    const std::string head = R"({"model":"m","input":[{"role":"system","content":[{"type":"input_text","text":"s"}]},{"type":"message","role":"user","content":[{"type":"input_text","text":"q"},)";
    const auto text = head + R"({"type":"input_image","image_url":)" + quoted(data_url(first)) + R"(,"detail":"high"}]},{"type":"function_call_output","call_id":"c","output":[{"type":"input_image","image_url":)" +
                      quoted(data_url(second)) + R"(,"file_id":null}]},{"role":"user","content":"plain"}]})";
    const auto out = normalize(parse_json(text), Api::responses);
    CHECK(out.body.dump() == head + R"({"type":"image","media":0}]},{"type":"function_call_output","call_id":"c","output":[{"type":"image","media":1}]},{"role":"user","content":"plain"}]})");
    CHECK(out.items.size() == 2);
    if (out.items.size() == 2) {
        CHECK(out.items[0].bytes == first && out.items[0].detail == "high" && out.items[0].param == "input[1].content[1]");
        CHECK(out.items[1].bytes == second && out.items[1].param == "input[2].output[0]");
    }

    const auto wrap = [](const std::string& part, std::string_view role = "user") {
        return R"({"model":"m","input":[{"role":)" + quoted(role) + R"(,"content":[)" + part + "]}]}";
    };
    const auto image = R"({"type":"input_image","image_url":)" + quoted(data_url(first)) + "}";
    for (const char* role : {"system", "developer", "assistant"}) {
        const auto result = normalize(parse_json(wrap(image, role)), Api::responses);
        CHECK(result.items.empty() && result.body.dump() == wrap(image, role));
    }
    const auto fails = [&](const std::string& part, std::string* where = nullptr) {
        return status_of([&] { (void)normalize(parse_json(wrap(part)), Api::responses); }, where);
    };
    std::string where;
    CHECK(fails(R"({"type":"input_image","file_id":"file-1"})", &where) == 400 && where == "input[0].content[0].file_id");
    CHECK(fails(R"({"type":"input_image"})", &where) == 400 && where == "input[0].content[0].image_url");
    CHECK(fails(R"({"type":"input_image","image_url":{"url":"data:image/png;base64,AAAA"}})") == 400);
    CHECK(fails(R"({"type":"input_image","image_url":"https://example.com/a.png"})") == 400);
    CHECK(fails(R"({"type":"input_image","image_url":)" + quoted(data_url(first)) + R"(,"detail":false})") == 400);
}

void remote() {
    const auto file = png(12, 34);
    int calls = 0;
    RemoteUrl seen;
    FetchRules rules;
    Options options;
    options.fetch = parse_fetch_policy("allowlist", "img.example.com,127.0.0.1,localhost,2130706433");
    options.fetcher = [&](const RemoteUrl& url, const FetchRules& r) {
        ++calls;
        seen = url;
        rules = r;
        return file;
    };
    const auto run = [&](const std::string& url, std::string* message = nullptr) {
        return status_of([&] { (void)normalize(parse_json(chat_body(chat_image(url))), Api::chat_completions, options); }, nullptr, message);
    };
    CHECK(run("https://IMG.example.com:8443/a/b.png?x=1#frag") == 0 && calls == 1);
    CHECK(seen.host == "img.example.com" && seen.port == 8443 && seen.target == "/a/b.png?x=1");
    CHECK(rules.max_bytes == options.limits.max_image_bytes && rules.timeout_ms == 5000 && !rules.follow_redirects);
    CHECK(run("HTTPS://img.example.com") == 0 && calls == 2 && seen.port == 443 && seen.target == "/");
    CHECK(run("https://img.example.com?x=1") == 0 && calls == 3 && seen.target == "/?x=1");
    const auto out = normalize(parse_json(chat_body(chat_image("https://img.example.com/a.png"))), Api::chat_completions, options);
    CHECK(out.items.size() == 1 && out.items[0].bytes == file && out.items[0].width == 12 && out.items[0].height == 34);
    calls = 0;

    // Refusals happen before the fetcher runs.
    CHECK(run("http://img.example.com/a.png") == 400);
    CHECK(run("https://other.example.com/a.png") == 400);
    CHECK(run("https://img.example.com.evil.test/a.png") == 400);
    CHECK(run("https://img.example.com@evil.test/a.png") == 400);
    CHECK(run("https://user:pw@img.example.com/a.png") == 400);
    CHECK(run("https://img.example.com:0/a.png") == 400);
    CHECK(run("https://img.example.com:70000/a.png") == 400);
    CHECK(run("https://img.example.com:8a/a.png") == 400);
    CHECK(run("https://img.example.com/a b.png") == 400);
    CHECK(run("https://img.example.com\\@evil.test/") == 400);
    CHECK(run("https:///a.png") == 400);
    // Address literals: private ranges are refused even when someone put them on the list.
    std::string message;
    CHECK(run("https://127.0.0.1/a.png", &message) == 400 && message.find("public") != std::string::npos);
    CHECK(run("https://2130706433/a.png", &message) == 400 && message.find("public") != std::string::npos);
    CHECK(run("https://0x7f.0.0.1/a.png") == 400);
    CHECK(run("https://127.1/a.png") == 400);
    CHECK(run("https://10.0.0.5/a.png") == 400);
    CHECK(run("https://169.254.169.254/latest/meta-data/") == 400);
    CHECK(run("https://[::1]/a.png") == 400);
    CHECK(run("https://[::ffff:192.168.0.1]:443/a.png") == 400);
    CHECK(run("https://[fe80::1]/a.png") == 400);
    CHECK(run("https://[::1/a.png") == 400);
    CHECK(run("https://8.8.8.8/a.png", &message) == 400 && message.find("allowlist") != std::string::npos); // public but not listed
    CHECK(calls == 0);
    // A name is checked by the fetcher against the address it resolves to; this layer passes it through.
    CHECK(run("https://localhost/a.png") == 0 && calls == 1);

    options.fetcher = [](const RemoteUrl&, const FetchRules&) -> std::string { throw std::runtime_error("redirect refused"); };
    CHECK(run("https://img.example.com/a.png", &message) == 400 && message.find("redirect refused") != std::string::npos);
    options.fetcher = [](const RemoteUrl&, const FetchRules& r) { return std::string(r.max_bytes + 1, 'x'); };
    options.limits.max_image_bytes = 64;
    CHECK(run("https://img.example.com/a.png") == 413);
    options.fetcher = [](const RemoteUrl&, const FetchRules&) { return std::string("not an image"); };
    CHECK(run("https://img.example.com/a.png") == 400);
    options.fetcher = nullptr;
    CHECK(run("https://img.example.com/a.png", &message) == 400 && message.find("not available") != std::string::npos);
    // data: URLs do not depend on the fetch policy.
    CHECK(run(data_url(file)) == 0);
}

} // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        digests(); base64_cases(); probes(); addresses(); policies(); text_only_is_untouched(); chat(); messages(); responses(); remote();
    } catch (const std::exception& e) {
        std::cerr << "unexpected exception: " << e.what() << '\n';
        return 1;
    }
    std::cout << "serve media tests: " << failures << " failures\n";
    return failures ? 1 : 0;
}
