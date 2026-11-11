// Host-only deployment tests using the serving harness's recording writer and real ApiService.
#include "strix/serve/api.hpp"
#include "strix/deepseek41_batch_forward.hpp"
#include <map>
#include <array>
#include <iostream>
using namespace strix::serve;
namespace {
int failures = 0;
void check(bool value, const char* label) {
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
const Json& at(const Json& j, const char* key) {
    const auto* value = j.find(key);
    if (!value) throw std::runtime_error(std::string("missing ") + key);
    return *value;
}
struct Writer final : HttpResponseWriter {
    int status = 0;
    std::string body;
    void send(int code, std::string_view, std::string_view text,
        const std::vector<std::pair<std::string, std::string>>&) override { status = code; body = text; }
    bool begin_stream(int, std::string_view, const std::vector<std::pair<std::string, std::string>>&) override { return false; }
    bool write(std::string_view) override { return false; }
    void end_stream() override {}
    bool client_gone() const override { return false; }
};
Writer get(ApiService& api, const char* path = "/health", const char* token = nullptr) {
    HttpRequest request; request.method = "GET"; request.path = path;
    if (token) request.headers.emplace_back("authorization", std::string("Bearer ") + token);
    Writer out; api.handle(request, out); return out;
}
void kv_pool_tests() {
    using namespace strix::deepseek41;
    struct Role { bool owns_kv; unsigned ratio; };
    const std::array<Role,3> roles{{{true,1},{true,2},{false,0}}};
    std::map<std::uint64_t,std::uint64_t> free{{0,32},{64,16}};
    constexpr std::uint64_t entry=352; // fixture KV + index row, independent of model assets
    const auto total=batch_pool_storage_bytes(80,roles,3,entry);
    const auto fragmented=batch_pool_free_bytes(free,roles,3,entry);
    check(fragmented==72*entry && fragmented<total,"free bytes sum all arena holes across KV-source layers");
    free.erase(0);
    check(batch_pool_free_bytes(free,roles,3,entry)==24*entry,"reserve subtracts actual compressed ownership");
    free.clear();
    check(batch_pool_free_bytes(free,roles,3,entry)==0,"exhausted pool is zero rather than null");
    free.emplace(0,80);
    check(batch_pool_free_bytes(free,roles,3,entry)==total,"release/coalescing restores complete arena bytes");
    GenerationQueue queue;
    ApiConfig config;
    config.deployment.capacity_provider=[&] { return ServeCapacity{4,0,batch_pool_free_bytes(free,roles,3,entry),0}; };
    ApiService api(config,{},queue,8192,false);
    check(at(at(parse_json(get(api,"/api/status").body),"capacity"),"kv_bytes_free").as_int()==static_cast<std::int64_t>(total),
          "allocator byte snapshot reaches HTTP capacity");
    queue.shutdown();
}
void deployment_tests() {
    GenerationQueue queue;
    ApiConfig config; config.http_slots = 3; config.api_key = "test-key";
    ApiService absent(config, {}, queue, 8192, false);
    check(get(absent, "/v1/deployment", "test-key").status == 404, "deployment absent is 404");
    check(!parse_json(get(absent).body).find("deployment_id"), "legacy health omits absent deployment");
    const std::string plan = "{ \"schema\": \"strix-deployment-v1\", \"context\": 8192 }";
    config.deployment.deployment_json = [&] { return plan; };
    config.deployment.deployment_id = [] { return "0123456789abcdef"; };
    config.deployment.capacity_provider = [] { return ServeCapacity{3, 2, 123456789012ULL, 4}; };
    unsigned notified = 0;
    config.on_drain = [&] { ++notified; };
    ApiService api(config, {}, queue, 8192, false);
    check(get(api, "/v1/deployment").status == 401, "deployment respects authentication");
    const auto deployment = get(api, "/v1/deployment", "test-key");
    check(deployment.status == 200 && deployment.body == plan, "deployment returns exact engine JSON bytes");
    for (const char* path : {"/health", "/v1/health"})
        check(at(parse_json(get(api, path).body), "deployment_id").as_string() == "0123456789abcdef", "health deployment identity");
    auto record = std::make_shared<ServeHealthRecord>(); record->context_tokens = 8192;
    config.health_record = record;
    ApiService enriched(config, {}, queue, 8192, false);
    check(at(parse_json(get(enriched).body), "deployment_id").as_string() == "0123456789abcdef", "enriched health deployment identity");
    const auto status = parse_json(get(api, "/api/status").body);
    const auto& c = at(status, "capacity");
    check(at(c, "slots_total").as_int() == 3 && at(c, "slots_busy").as_int() == 2 &&
        at(c, "kv_bytes_free").as_int() == 123456789012LL && at(c, "queue_depth").as_int() == 4,
        "capacity is supplied by callback without changing values");
    auto first = queue.submit({}), second = queue.submit({});
    auto running = queue.take(false);
    check(first && second && running && queue.snapshot().running == 1, "accepted work before drain");
    HttpRequest request; request.method = "POST"; request.path = "/admin/drain";
    Writer denied; api.handle(request, denied);
    check(denied.status == 401 && !queue.draining(), "unauthorized drain has no effect");
    request.headers.emplace_back("authorization", "Bearer test-key");
    Writer started; api.handle(request, started);
    check(started.status == 200 && !at(parse_json(started.body), "drained").as_bool(), "drain reports in-flight work");
    check(!queue.submit({}) && !queue.stopping(), "drain closes admission without stopping owner");
    check(queue.snapshot().queued == 1 && queue.snapshot().running == 1, "drain preserves queued and running jobs");
    check(get(api).status == 503 && get(enriched).status == 503, "both health renderers report draining");
    Writer repeat; api.handle(request, repeat);
    check(notified == 1 && repeat.status == 200, "drain notification is idempotent");
    queue.finish(*running, {});
    auto waiting = queue.take(false);
    check(bool(waiting), "accepted queued job still runs");
    queue.finish(*waiting, {});
    Writer finished; api.handle(request, finished);
    bool first_done = false, second_done = false;
    (void)first->wait(std::chrono::milliseconds(0), first_done);
    (void)second->wait(std::chrono::milliseconds(0), second_done);
    check(at(parse_json(finished.body), "drained").as_bool() && first_done && second_done, "drained only after completion");
    check(at(parse_json(get(api, "/api/status").body), "drained").as_bool(), "status reports drained");
    GenerationQueue plain_queue;
    ApiService plain({}, {}, plain_queue, 8192, false);
    Writer default_drain; HttpRequest plain_request; plain_request.method = "POST"; plain_request.path = "/admin/drain";
    plain.handle(plain_request, default_drain);
    check(default_drain.status == 200 && plain_queue.draining(), "drain exists without engine hook");
    check(at(at(parse_json(get(plain, "/api/status").body), "capacity"), "kv_bytes_free").is_null(), "unmeasured KV remains unknown");
    GenerationQueue external;
    {
        QueueExternalWork work(external);
        check(bool(work) && external.begin_drain() && !external.drained(), "direct backend work keeps drain pending");
        QueueExternalWork refused(external);
        check(!refused && external.snapshot().running == 0, "direct work refuses admission without changing generation counters");
    }
    check(external.drained(), "direct backend completion releases drain");
    queue.shutdown(); plain_queue.shutdown();
}

} // namespace
int main() try {
    kv_pool_tests(); deployment_tests();
    std::cout << "deployment failures: " << failures << '\n';
    return failures ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
}
