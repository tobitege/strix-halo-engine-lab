// The example family of docs/adding-an-engine.md serves a completion and a chat
// request through the shared serve shell with the CPU reference backend.
#include "../examples/skeleton_engine/skeleton_engine.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
namespace sv = strix::serve;

void check(bool ok, const char* what) {
    if (ok) return;
    std::cerr << "FAILED: " << what << "\n";
    std::exit(1);
}
#define CHECK(x) check((x), #x)

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

Writer post(const sv::HttpHandler& handler, const std::string& path, const std::string& body) {
    sv::HttpRequest request;
    request.method = "POST";
    request.path = path;
    request.target = path;
    request.body = body;
    Writer writer;
    handler(request, writer);
    return writer;
}
} // namespace

// Embedding rows through the engine contract: a bound table resolves virtual
// ids, everything else is refused before a write.
void embed_rows() {
    namespace ss = strix::session;
    ss::CpuBackend backend;
    skeleton::Engine engine(backend);
    CHECK(engine.caps().embed_rows);
    ss::EmbedInterner interner;
    ss::EmbedRows rows;
    rows.digest[0] = 7;
    rows.rows = 3;
    rows.width = 2;
    rows.data.assign(6, 0x3f80);
    const auto lease = interner.intern(std::move(rows));
    auto table = std::make_shared<const ss::EmbedTable>(std::vector<ss::EmbedLease>{lease});
    std::vector<std::uint32_t> prompt{'a', 'b'};
    for (const auto id : lease.ids()) prompt.push_back(id);
    prompt.push_back('c');

    auto lane = engine.store.acquire();
    const auto refused = [&](std::span<const std::uint32_t> ids) {
        try {
            engine.prefill(ids, lane);
        } catch (const std::invalid_argument&) {
            return engine.store.consumed(lane) == 0;
        }
        return false;
    };
    CHECK(refused(prompt)); // no table bound
    engine.bind_embeds(lane, table);
    const std::vector<std::uint32_t> outside{'a', ss::embed_id(lease.handle(), 3)};
    CHECK(refused(outside)); // row outside the span
    const std::vector<std::uint32_t> unknown{'a', ss::embed_id(lease.handle() + 1, 0)};
    CHECK(refused(unknown)); // handle outside the request
    engine.prefill(prompt, lane);
    CHECK(engine.store.consumed(lane) == prompt.size());
    engine.store.release(lane);
    CHECK(interner.status().live == 1); // lease + table still hold it
}

int main(int argc, char** argv) {
    CHECK(argc == 2);
    embed_rows();
    sv::ServiceFactories factories;
    skeleton::install_session_service(factories);

    // The shell seam replaces only listen/accept: selection, validation,
    // construction and cleanup are the production path of strix::serve::run().
    bool served = false;
    sv::ServeShell shell;
    shell.serve = [&](const sv::HttpServerConfig&, sv::HttpHandler handler, sv::SessionRuntime& runtime) {
        std::thread owner([&] { runtime.run(); });
        const auto completion = post(handler, "/v1/completions",
            R"({"model":"skeleton","prompt":"hi","max_tokens":4})");
        const auto chat = post(handler, "/v1/chat/completions",
            R"({"model":"skeleton","messages":[{"role":"user","content":"hi"}],"max_tokens":4})");
        runtime.shutdown();
        owner.join();
        if (completion.status != 200 || chat.status != 200)
            std::cerr << completion.status << " " << completion.body << "\n" << chat.status << " " << chat.body << "\n";
        CHECK(completion.status == 200);
        // Two prompt rows: the counter continues with c, d, e, f.
        CHECK(completion.body.find("cdef") != std::string::npos);
        CHECK(chat.status == 200);
        CHECK(chat.body.find("cdef") != std::string::npos);
        served = true;
    };

    std::vector<std::string> arguments{"serve", "--registry-root", argv[1], "--registry", "models/registry.toml",
                                       "--model", "skeleton", "--port", "0"};
    std::vector<char*> pointers;
    for (auto& a : arguments) pointers.push_back(a.data());
    std::ostringstream out, error;
    const auto code = sv::run(static_cast<int>(pointers.size()), pointers.data(), factories, out, error, shell);
    if (code != 0) std::cerr << "run: " << code << " " << error.str() << "\n";
    CHECK(code == 0);
    CHECK(served);

    auto inspect = arguments;
    inspect.resize(7);
    inspect.push_back("--family-inspect");
    pointers.clear();
    for (auto& a : inspect) pointers.push_back(a.data());
    std::ostringstream inspected, inspect_error;
    CHECK(sv::run(static_cast<int>(pointers.size()), pointers.data(), factories, inspected, inspect_error) == 0);
    CHECK(inspected.str().find("cpu backend") != std::string::npos);
    std::cout << "skeleton engine: ok\n";
    return 0;
}
