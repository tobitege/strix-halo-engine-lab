#include "strix/model_input.hpp"
#include "strix/session/checkpoint_io.hpp"
#include "strix/ds41_file_reader.hpp"
#include "strix/ds41_dspark_vocab.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/deepseek41_session_cache.hpp"
#include "strix/serve/session_api.hpp"
#include "strix/deepseek41_attention_host.hpp"
#include "strix/ds41_attention.hpp"
#include "strix/ds41_native_fp8.hpp"
#include "strix/ds41_trellis.hpp"
#include "strix/model_content_hash.hpp"
#include <array>
#include <algorithm>
#include <cstdlib>
#include <condition_variable>
#include <mutex>
#include <fstream>
#include <iostream>
#include <chrono>
#include <iterator>
#include <thread>

using namespace strix::deepseek41;
using Sessions = PrefixCache::Sessions;
namespace io = strix::session::checkpoint_io;
#ifdef NDEBUG
#error DS41 host tests require active assertions
#endif
namespace {
void require(bool ok, const char *why) {
    if (!ok)
        throw std::runtime_error(why);
}
template <class F> void refuses(F f) {
    bool bad = false;
    try {
        f();
    } catch (const std::exception &) {
        bad = true;
    }
    require(bad, "expected refusal");
}
SessionCompat compat(unsigned rank = 0) {
    SessionCompat c;
    c.fields = {{"rank", std::to_string(rank)}, {"world", "2"}, {"selector", "exact"}, {"dspark", "off"}, {"schema", "test-v1"}};
    return c;
}
SessionCheckpoint sample(unsigned rank, const std::string &id = "aa", const std::string &gen = "generation1", unsigned layers = 4) {
    SessionCheckpoint c;
    c.id = id;
    c.generation = gen;
    c.rank = rank;
    c.compat_record = compat(rank).record();
    c.attention.position = 3;
    c.attention.next_token = 17;
    c.attention.tokens = {1, 2, 3};
    c.attention.layers.resize(layers);
    const auto plan = strix::ds41::attn::released_plan();
    for (unsigned i = 0; i < layers; ++i) {
        auto &l = c.attention.layers[i];
        const auto &r = plan[i];
        l.owns_kv = r.owns_kv;
        l.owns_index = r.owns_index;
        l.ratio = r.ratio;
        l.ring.resize(136U * 528U, static_cast<std::uint8_t>(i));
        l.topk_known = false;
        if (r.owns_kv) {
            l.kv.resize((3U / r.ratio) * 288U, 5);
            l.idx_keys.resize((3U / r.ratio) * 68U, 7);
            if (r.ratio == 2) {
                l.carry.resize(1024, 0.25F);
                l.carry_valid = true;
            }
        }
    }
    c.frontier.consumed = c.frontier.accepted = 3;
    c.frontier.emitted = 2;
    c.frontier.pending = 17;
    return c;
}
struct Pair {
    std::mutex mutex;
    std::condition_variable cv;
    unsigned arrived = 0, departed = 0, round = 0;
    bool broken = false;
    std::array<SessionBytes, 2> packets;
    std::array<std::exception_ptr, 2> errors;
    SessionExchange exchange(unsigned r) {
        return [this, r](std::span<const std::uint8_t> b) {
            std::unique_lock<std::mutex> lock(mutex);
            const auto epoch = round;
            const auto wait = [&](auto ready) {
                if (!cv.wait_for(lock, std::chrono::seconds(5), [&] { return broken || ready(); })) {
                    broken = true; cv.notify_all();
                }
                require(!broken, "paired exchange deadline (never an expected refusal)");
            };
            packets[r].assign(b.begin(), b.end());
            ++arrived; cv.notify_all(); wait([&] { return arrived == 2; });
            auto out = packets[1 - r];
            if (++departed == 2) { arrived = departed = 0; ++round; cv.notify_all(); }
            else wait([&] { return round != epoch; });
            return out;
        };
    }
    SessionHostExchange host_exchange(unsigned r) {
        return [ex = exchange(r)](std::span<const std::byte> out, std::span<std::byte> in) {
            const auto peer = ex({reinterpret_cast<const std::uint8_t*>(out.data()), out.size()});
            require(peer.size() == in.size(), "startup exchange size");
            std::copy(peer.begin(), peer.end(), reinterpret_cast<std::uint8_t*>(in.data()));
        };
    }
    template <class F> void run(F f, bool success = true) {
        errors = {};
        std::thread peer([&] {
            try {
                f(1, exchange(1));
            } catch (...) {
                errors[1] = std::current_exception();
            }
        });
        try {
            f(0, exchange(0));
        } catch (...) {
            errors[0] = std::current_exception();
        }
        peer.join();
        if (success) {
            for (auto e : errors)
                if (e)
                    std::rethrow_exception(e);
        } else
            require(errors[0] && errors[1], "group failure must refuse BOTH ranks");
    }
};
void startup_failures() {
    for (const auto* reason : {"model files changed during startup", "duplicate tensor name", "asset directory unavailable", "hash read failed"}) {
        for (unsigned failed = 0; failed < 2; ++failed) {
            for (int explicit_rank = -1; explicit_rank < 2; ++explicit_rank) {
                Pair pair;
                pair.run([&](unsigned rank, const auto&) {
                    std::string diagnostic;
                    auto session = admit_session_startup([&] {
                        if (rank == failed) throw std::runtime_error(reason);
                        return std::make_unique<int>(1);
                    }, [&](const char* why) { diagnostic = why; });
                    if (rank == failed)
                        require(diagnostic == reason, "setup failure reason");
                    const bool enabled = session_startup_ready(bool(session), static_cast<int>(rank) == explicit_rank, pair.host_exchange(rank));
                    if (explicit_rank < 0) require(!enabled, "one-rank setup failure enabled sessions");
                }, explicit_rank < 0);
            }
        }
    }
    require(!admit_session_startup([]() -> std::unique_ptr<int> { throw 1; }, [](const char* line) {
        require(std::string_view(line) == "unknown session setup failure", "unknown setup failure reason");
    }), "unknown setup failure enabled sessions");
    Pair pair;
    pair.run([&](unsigned rank, const auto&) {
        require(session_startup_ready(true, false, pair.host_exchange(rank)), "successful session setup refused");
    });
}
void draft_envelopes() {
    auto cfg = compat();
    cfg.fields["dspark"] = "on";
    auto c = sample(0);
    c.compat_record = cfg.record();
    c.draft.emplace();
    auto& d = *c.draft;
    d.context.start = 1; d.context.next = 3; d.context.core = 100;
    d.context.extra = {127000, 101, 129000}; // order is head split identity
    for (unsigned i = 0; i < 3; ++i) {
        d.context.rings[i].resize(128U * 528U);
        for (std::size_t j = 0; j < d.context.rings[i].size(); ++j)
            d.context.rings[i][j] = static_cast<std::uint8_t>((j + i) % 251);
    }
    d.controller.costs.verify[1] = 0.125F;
    d.controller.costs.samples[1] = 7;
    d.controller.costs.draft = 0.03125F;
    d.controller.seen[4][7] = 11; d.controller.hits[4][7] = 9;
    d.controller.decisions = 31; d.controller.sweep = 4; d.controller.explore = true;
    d.exchange_request = 65; d.windows = 13; d.verified = 22; d.accepted = 9;
    c.frontier.drafter = 3;
    c.frontier.lineage = "ds41-dspark-r1-committed-turn-v2";
    const auto bytes = serialize_session(c);
    const auto copy = deserialize_session(bytes, cfg, 2, 0, bytes.size());
    require(serialize_session(copy) == bytes, "DSpark envelope byte roundtrip");
    require(serialize_session_draft(*copy.draft) == serialize_session_draft(d), "DSpark full state roundtrip");
    const auto digest = session_state_sha256(c);
    auto bad = c;
    bad.draft->context.rings[2][99] ^= 1;
    require(session_state_sha256(bad) != digest, "drafter ring missing from canonical digest");
    bad = c; bad.draft->controller.decisions++;
    require(session_state_sha256(bad) != digest, "controller missing from canonical digest");
    bad = c; std::swap(bad.draft->context.extra[0], bad.draft->context.extra[1]);
    require(session_state_sha256(bad) != digest, "vocabulary order missing from digest");
    bad = c; bad.draft->context.start = 4;
    refuses([&] { (void)serialize_session(bad); });
    bad = c; bad.draft->context.next = 2;
    refuses([&] { (void)serialize_session(bad); });
    bad = c; bad.draft->context.rings[0].pop_back();
    refuses([&] { (void)serialize_session(bad); });
    bad = c; bad.draft->context.extra.push_back(101);
    refuses([&] { (void)serialize_session(bad); });
    bad = c; bad.draft->controller.hits[0][0] = 1;
    refuses([&] { (void)serialize_session(bad); });
    auto raw = serialize_session_draft(d);
    raw.back() = 1;
    refuses([&] { (void)deserialize_session_draft(raw); });
    raw = serialize_session_draft(d); raw.resize(raw.size() - 1);
    refuses([&] { (void)deserialize_session_draft(raw); });
    refuses([&] { (void)deserialize_session(bytes, compat(), 2, 0, bytes.size()); });
    auto corrupt = bytes; corrupt[corrupt.size() / 2] ^= 1;
    refuses([&] { (void)deserialize_session(corrupt, cfg, 2, 0, bytes.size()); });
    std::cout << "PASS DSpark v2 rings/cursors/vocabulary/controller/frontier/negative cases\n";
}

void draft_groups(const std::filesystem::path& dir) {
    Pair pair;
    std::array<SessionCompat, 2> keys{compat(0), compat(1)};
    for (auto& key : keys) key.fields["dspark"] = "on";
    const auto make = [&](unsigned r) {
        auto c = sample(r);
        c.compat_record = keys[r].record();
        c.draft.emplace();
        c.draft->context.next = 3;
        for (auto& ring : c.draft->context.rings) ring.resize(128U * 528U, 17);
        c.frontier.drafter = 3;
        c.frontier.lineage = "ds41-dspark-r1-committed-turn-v2";
        return c;
    };
    std::array<std::unique_ptr<Sessions>, 2> caches;
    for (unsigned r = 0; r < 2; ++r)
        caches[r] = std::make_unique<Sessions>(keys[r], r, 4U << 20U, dir / ("draft-r" + std::to_string(r)));
    pair.run([&](unsigned r, const auto& ex) { caches[r]->capture(make(r), ex); });
    pair.run([&](unsigned r, const auto& ex) { caches[r]->checkpoint("aa", ex); });
    // Fresh cache objects, same durable group, not a RAM hit.
    for (unsigned r = 0; r < 2; ++r)
        caches[r] = std::make_unique<Sessions>(keys[r], r, 4U << 20U, dir / ("draft-r" + std::to_string(r)));
    pair.run([&](unsigned r, const auto& ex) {
        const auto restored = caches[r]->resume("aa", true, ex);
        require(serialize_session(restored) == serialize_session(make(r)), "DSpark group restart state");
    });
    // A rank-local valid but different replicated ring must fail on BOTH ranks.
    pair.run([&](unsigned r, const auto& ex) {
        auto c = make(r);
        if (r == 1) c.draft->context.rings[2][123] ^= 1;
        caches[r]->capture(c, ex);
    }, false);
    for (std::uint64_t next : {127ULL, 128ULL, 129ULL, 8193ULL}) {
        auto d = *make(0).draft;
        d.context.next = next; d.context.start = next - 9;
        const auto bytes = serialize_session_draft(d);
        require(serialize_session_draft(deserialize_session_draft(bytes)) == bytes, "DSpark ring-wrap/skip cursor");
    }
    std::cout << "PASS DSpark durable group restart/replicated divergence/ring-wrap cursors\n";
}

void envelopes() {
    const auto c = sample(0);
    auto bytes = serialize_session(c);
    const auto copy = deserialize_session(bytes, compat(), 2, 0, bytes.size());
    require(serialize_session(copy) == bytes, "byte preserving envelope");
    require(copy.frontier.pending == 17 && copy.attention.tokens == c.attention.tokens, "pending/history roundtrip");
    for (auto at : {std::size_t{0}, bytes.size() / 2, bytes.size() - 1}) {
        auto bad = bytes;
        bad[at] ^= 1;
        refuses([&] { (void)deserialize_session(bad, compat(), 2, 0, bad.size()); });
    }
    for (auto size : {std::size_t{0}, std::size_t{7}, bytes.size() / 2, bytes.size() - 1})
        refuses([&] { (void)deserialize_session(std::span(bytes).first(size), compat(), 2, 0, bytes.size()); });
    auto other = compat();
    other.fields["selector"] = "approximate";
    refuses([&] { (void)deserialize_session(bytes, other, 2, 0, bytes.size()); });
    other = compat();
    other.fields["exe"] = "another-build";
    refuses([&] { (void)deserialize_session(bytes, other, 2, 0, bytes.size()); });
    other = compat();
    other.fields["dspark"] = "on";
    refuses([&] { (void)deserialize_session(bytes, other, 2, 0, bytes.size()); });
    refuses([&] { (void)deserialize_session(bytes, compat(), 2, 1, bytes.size()); });
    refuses([&] { (void)deserialize_session(bytes, compat(), 1, 0, bytes.size()); });
    refuses([&] { (void)deserialize_session(bytes, compat(), 2, 0, bytes.size() - 1); });
    auto bad = c;
    bad.frontier.drafter = 1;
    refuses([&] { (void)serialize_session(bad); });
    bad = c;
    bad.frontier.pending = 99;
    refuses([&] { (void)serialize_session(bad); });
    bad = c;
    bad.attention.layers[0].ring.resize(1);
    refuses([&] { (void)serialize_session(bad); });
    for (const auto &id : {"../escape", "a/b", "", "a.b"})
        refuses([&] { validate_session_id(id); });
    auto x = c.attention, y = x;
    y.layers[0].ring[100 * 528] = 99;
    require(session_state_sha256(x) == session_state_sha256(y), "ignore inactive ring slots");
    y.layers[0].ring[0] ^= 1;
    require(session_state_sha256(x) != session_state_sha256(y), "bind live ring bytes");
    std::cout << "PASS envelope/bounds/compat/frontier/canonical-state\n";
}
void loader_retarget([[maybe_unused]] const std::filesystem::path& dir) {
#ifdef __linux__
    const auto root = dir / "startup-retarget";
    std::filesystem::create_directory(root);
    const auto one = root / "one", two = root / "two", weights = root / "weights";
    { std::ofstream(one, std::ios::binary).put(1); std::ofstream(two, std::ios::binary).put(2); }
    const auto retarget = [&](const auto& target) {
        const auto next = root / "next";
        std::filesystem::create_symlink(target.filename(), next);
        std::filesystem::rename(next, weights);
    };
    retarget(one);
    const auto old = session_compat({}, {{"model", weights}}, one);
    auto checkpoint = sample(0); checkpoint.compat_record = old.record();
    const auto encoded = serialize_session(checkpoint);
    retarget(two);
    strix::LoaderCapture capture;
    strix::ds41::FileReader reader(weights, false, 1);
    std::uint8_t loaded = 0; reader.read_range(0, 1, &loaded);
    require(loaded == 2, "production reader loaded second content");
    const auto snapshot = strix::LoaderCapture::snapshot();
    require(snapshot.size() == 1 && snapshot[0].identity.ino != 0, "fd identity recorded");
    const auto current = session_compat({}, {{"model", weights}}, one, &snapshot);
    require(current.key() != old.key(), "loaded content changes key");
    refuses([&] { (void)deserialize_session(encoded, current, 2, 0, encoded.size()); });
    const auto before = strix::file_hash_cache_stats();
    require(session_compat({}, {{"model", weights}}, one, &snapshot).key() == current.key(), "warm loader key");
    require(strix::file_hash_cache_stats().misses == before.misses, "warm loader compatibility misses");
    require(current.key() == session_compat({}, {{"model", weights}}, one).key(), "prewarm/live key equivalence");
    retarget(one);
    reader.read_range(0, 1, &loaded);
    require(loaded == 2, "retained reader still loads second content");
    refuses([&] { (void)session_compat({}, {{"model", weights}}, one, &snapshot); });
    std::string diagnostic;
    const auto factory = [&] {
        (void)session_compat({}, {{"model", weights}}, one, &snapshot);
        return std::make_unique<int>(1);
    };
    const auto log = [&](const char* line) { diagnostic = line; };
    require(!admit_session_startup(factory, log), "automatic changed-model session enabled");
    require(diagnostic == "model files changed during startup", "automatic startup diagnostic");
    // The identity overload cannot hash the current path for an earlier loader,
    // even when the persistent cache is disabled.
    const char* setting = std::getenv("STRIX_HASH_CACHE");
    const bool had_setting = setting != nullptr;
    const std::string saved = setting ? setting : "";
    setenv("STRIX_HASH_CACHE", "0", 1);
    refuses([&] { (void)strix::sha256_file_content(weights, snapshot[0].identity); });
    if (had_setting) setenv("STRIX_HASH_CACHE", saved.c_str(), 1); else unsetenv("STRIX_HASH_CACHE");
    retarget(two);
    { std::fstream out(two, std::ios::in | std::ios::out | std::ios::binary); out.put(3); }
    refuses([&] { strix::verify_loader_snapshot(snapshot); });
    std::cout << "PASS startup retarget loaded=2 disk=1 old_key=" << old.key()
              << " loaded_key=" << current.key() << " warm_and_cold=refused in_place=refused\n";

    // Every separate safetensors index/header/data open contributes an identity.
    const auto pack = root / "pack"; std::filesystem::create_directory(pack);
    const auto shard = pack / "shard";
    const std::string header = R"({"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})";
    { std::ofstream out(shard, std::ios::binary); const auto n = header.size();
      for (unsigned k = 0; k < 8; ++k) out.put(static_cast<char>(static_cast<std::uint64_t>(n) >> (8U*k)));
      out << header; out.put(7); }
    { std::ofstream(pack / "model.safetensors.index.json") << R"({"weight_map":{"x":"shard"}})"; }
    strix::ds41::SafetensorsCheckpoint safetensors(pack);
    const auto* tensor = safetensors.find("x"); require(tensor != nullptr, "fixture tensor");
    safetensors.read(*tensor, &loaded); require(loaded == 7, "safetensors payload");
    const auto files = strix::LoaderCapture::snapshot();
    require(std::any_of(files.begin(), files.end(), [&](const auto& f) { return f.path == shard; }), "shard identity");
    require(std::any_of(files.begin(), files.end(), [&](const auto& f) { return f.path == pack / "model.safetensors.index.json"; }), "index identity");
    { std::ofstream out(pack / "model.safetensors.index.json", std::ios::app); out << " "; }
    refuses([&] { strix::verify_loader_snapshot(files); });

    const auto vocab = root / "vocab.bin";
    { std::ofstream out(vocab, std::ios::binary); for (int i = 0; i < 4; ++i) out.put(i == 0 ? 1 : 0); }
    require(strix::ds41::dspark::read_vocabulary_file(vocab.string()).at(0) == 1, "vocabulary payload");
    const auto gguf = root / "model.gguf";
    { std::ofstream out(gguf, std::ios::binary); out << "GGUF"; out.put(3);
      for (int i = 0; i < 27; ++i) out.put(0); }
    require(strix::gguf::read_file(gguf).version == 3, "GGUF fixture");
    const auto all = strix::LoaderCapture::snapshot();
    for (const auto& path : {vocab, gguf})
        require(std::any_of(all.begin(), all.end(), [&](const auto& f) { return f.path == path; }), "vocab/GGUF fd identity");
    std::cout << "PASS startup safetensors index/header/data, GGUF and vocabulary fd snapshots\n";
#endif
}
void full_hash(const std::filesystem::path &dir) {
    const auto file = dir / "weights.bin";
    std::string bytes(40U << 20U, 'a');
    {
        std::ofstream f(file, std::ios::binary);
        f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    const auto first = session_file_sha256(file);
    require(first == session_file_sha256(file), "cached full hash stable");
    {
        std::fstream f(file, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(20U << 20U);
        f.put('b');
    }
    std::filesystem::last_write_time(file, std::filesystem::last_write_time(file) + std::chrono::seconds(2));
    const auto second = session_file_sha256(file);
    require(first != second, "middle content must affect strict identity");
    require(second == strix::model_content_identity_hex(strix::sha256_file_content(file)), "full SHA matches existing SHA implementation");
    const auto cfg = session_compat({{"rank", "0"}}, {{"weights", file}}, file);
    auto changed = cfg;
    changed.fields["env:STRIX_DS41_IDX_DIRECT"] = "1";
    require(changed.key() != cfg.key(), "selector key binding");
    const auto assets = dir / "assets";
    std::filesystem::create_directories(assets / ".metadata" / "nested");
    std::ofstream(assets / "regular") << "weights";
    const auto key = [&] { return session_compat({}, {{"pack", assets}}, file).key(); };
    const auto original = key();
    std::ofstream(assets / ".gitattributes") << "hidden change";
    std::ofstream(assets / ".metadata" / "nested" / "regular") << "hidden subtree";
    require(key() == original, "hidden asset metadata must not affect key");
    std::ofstream(assets / ".gitattributes", std::ios::app) << "more metadata";
    std::ofstream(assets / ".metadata" / "nested" / "regular", std::ios::app) << "more hidden metadata";
    require(key() == original, "hidden asset changes must not affect key");
    std::ofstream(assets / "regular", std::ios::app) << "changed";
    require(key() != original, "regular asset change must affect key");
    const auto environment = [](const char* name, const char* value) {
#ifdef _WIN32
        _putenv_s(name, value ? value : "");
#else
        if (value) setenv(name, value, 1); else unsetenv(name);
#endif
    };
    // The prepared-loader OFF sentinel is not an asset named "0".
    {
        const char* prior = std::getenv("STRIX_DS41_PREPARED_LOAD");
        const bool present = prior != nullptr;
        const std::string saved = prior ? prior : "";
        environment("STRIX_DS41_PREPARED_LOAD", "0");
        const auto off = session_compat({}, {{"model", file}}, file);
        require(!off.fields.contains("asset:env:STRIX_DS41_PREPARED_LOAD"), "prepared OFF is not a file asset");
        environment("STRIX_DS41_PREPARED_LOAD", present ? saved.c_str() : nullptr);
    }
    for (const auto* name : {"GPU_TURN_ID", "GPU_TURN_DEADLINE", "GPU_WINDOW", "STRIX_HASH_CACHE",
             "STRIX_SESSION_DIR", "STRIX_SESSION_CACHE", "STRIX_SESSION_DISK", "STRIX_SESSION_DISK_BYTES",
             "STRIX_SESSION_RAM_BYTES", "STRIX_SESSION_MAX_PARKED", "STRIX_SESSION_POOL_IDENTITY_TEST",
             "STRIX_DS41_CB4", "AMD_SESSION_TEST_NUMERICS", "GPU_SESSION_TEST_NUMERICS",
             "HIP_SESSION_TEST_NUMERICS", "HSA_SESSION_TEST_NUMERICS", "ROCR_SESSION_TEST_NUMERICS",
             "STRIX_SESSION_FUTURE_NUMERICS"}) {
        const char* prior = std::getenv(name);
        const bool present = prior != nullptr;
        const std::string saved = prior ? prior : "";
        environment(name, "0"); const auto before = key();
        environment(name, "1"); const auto after = key();
        const std::string_view n(name);
        const bool numerical = n == "STRIX_DS41_CB4" || n.ends_with("NUMERICS");
        require((before != after) == numerical, "environment numerical/operational binding");
        environment(name, present ? saved.c_str() : nullptr);
    }
    const auto shard = [](const std::filesystem::path& path, const std::string& name, char byte) {
        const std::string header = "{\"" + name + "\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}";
        std::ofstream out(path, std::ios::binary);
        const auto n = static_cast<std::uint64_t>(header.size());
        for (unsigned k = 0; k < 8; ++k) out.put(static_cast<char>(n >> (8U*k)));
        out << header; out.put(byte);
    };
    // The compatibility inventory is exactly the checkpoint loader's resolution.
    const auto pack = dir / "checkpoint";
    std::filesystem::create_directories(pack);
    shard(pack / ".hidden.safetensors", "x", 80);
    shard(pack / "visible.safetensors", "y", 81);
    const auto pack_key = [&] { return session_compat({}, {{"env:STRIX_DS41_TRELLIS", pack}}, file).key(); };
    const auto mutate = [&](const std::filesystem::path& path) {
        std::ofstream(path, std::ios::app) << "mutation";
        std::filesystem::last_write_time(path, std::filesystem::last_write_time(path) + std::chrono::seconds(2));
    };
    auto prior_key = pack_key();
    mutate(pack / ".hidden.safetensors");
    require(pack_key() != prior_key, "unindexed hidden shard bound");
    prior_key = pack_key(); mutate(pack / "visible.safetensors");
    require(pack_key() != prior_key, "visible checkpoint shard bound");
    prior_key = pack_key(); mutate(pack / ".notes");
    require(pack_key() == prior_key, "unconsumed dotfile excluded");
    std::ofstream(pack / "model.safetensors.index.json") << R"({"weight_map":{"x":".hidden.safetensors","y":"../outside.bin"}})";
    shard(dir / "outside.bin", "y", 82);
    prior_key = pack_key(); mutate(pack / ".hidden.safetensors");
    require(pack_key() != prior_key, "indexed hidden shard bound");
    prior_key = pack_key(); mutate(dir / "outside.bin");
    require(pack_key() != prior_key, "indexed external shard bound");
    prior_key = pack_key(); mutate(pack / "visible.safetensors");
    require(pack_key() == prior_key, "unreferenced shard excluded with index");
#ifndef _WIN32
    std::filesystem::create_symlink(dir / "outside.bin", pack / ".link");
    std::ofstream(pack / "model.safetensors.index.json") << R"({"weight_map":{"x":".link"}})";
    prior_key = pack_key(); mutate(dir / "outside.bin");
    require(pack_key() != prior_key, "symlink target content bound");
    const auto alias = dir / "checkpoint-alias";
    std::filesystem::create_directory_symlink(pack, alias);
    const auto alias_key = [&] { return session_compat({}, {{"env:STRIX_DS41_TRELLIS", alias}}, file).key(); };
    const auto before_alias = alias_key(); mutate(dir / "outside.bin");
    require(alias_key() != before_alias, "directory symlink resolved");
#endif
    const char* native_prior = std::getenv("STRIX_DS41_NATIVE_FP8");
    const bool native_present = native_prior != nullptr;
    const std::string native_saved = native_prior ? native_prior : "";
    environment("STRIX_DS41_NATIVE_FP8", pack.string().c_str());
    prior_key = key(); mutate(dir / "outside.bin");
    require(key() != prior_key, "native FP8 directory contents bound");
    environment("STRIX_DS41_NATIVE_FP8", native_present ? native_saved.c_str() : nullptr);
    const auto duplicate = dir / "duplicate";
    std::filesystem::create_directory(duplicate);
    shard(duplicate / "a.safetensors", "x", 80);
    shard(duplicate / "b.safetensors", "x", 81);
    const auto duplicate_key = [&] { return session_compat({}, {{"env:STRIX_DS41_NATIVE_FP8", duplicate}}, file).key(); };
    strix::ds41::SafetensorsCheckpoint ordinary(duplicate);
    require(ordinary.find("x") != nullptr, "ordinary duplicate loading changed");
    refuses(duplicate_key);
    // Recreate the first shard with identical bytes, changing directory scan order.
    std::filesystem::rename(duplicate / "a.safetensors", dir / "archived-a.safetensors");
    shard(duplicate / "a.safetensors", "x", 80);
    refuses(duplicate_key);
    std::ofstream(duplicate / "model.safetensors.index.json") << R"({"weight_map":{"x":"a.safetensors","other":"b.safetensors"}})";
    refuses(duplicate_key);
    const auto suffix = dir / "head#v1";
    std::filesystem::create_directory(suffix);
    shard(suffix / "head.safetensors", "head", 83);
    environment("STRIX_DS41_TRELLIS", (dir / "nonexistent-unused-base").string().c_str());
    const auto mix = "default=" + pack.string() + ",head=" + suffix.string();
    environment("STRIX_DS41_TRELLIS_MIX", mix.c_str());
    strix::ds41::TrellisPack loaded(strix::ds41::trellis_env_spec());
    require(loaded.class_dir("head") == suffix && loaded.class_dir("q_a") == pack, "effective mix loader resolution");
    prior_key = key(); mutate(suffix / "head.safetensors");
    require(key() != prior_key, "inline # suffix path not bound");
    const auto manifest = dir / "mix.txt";
    std::ofstream(manifest) << " # comment\n default=" << pack.string() << "  # trailing comment\n";
    environment("STRIX_DS41_TRELLIS_MIX", manifest.string().c_str());
    strix::ds41::TrellisPack manifest_loaded(strix::ds41::trellis_env_spec());
    require(manifest_loaded.class_dir("head") == pack, "manifest parser differs");
    (void)key();
    environment("STRIX_DS41_TRELLIS_MIX", nullptr);
    environment("STRIX_DS41_TRELLIS", nullptr);
    std::cout << "PASS session duplicate shard refusal (indexed/unindexed), effective mix/default/inline suffix/manifest\n";
    std::cout << "PASS full-file hash and identity invalidation\n";
}
void group_tests(const std::filesystem::path &dir) {
    const auto d0 = dir / "rank0", d1 = dir / "rank1";
    const std::array paths{d0, d1};
    const auto budget = 8U << 20U;
    std::array<std::unique_ptr<Sessions>, 2> cache;
    for (unsigned r = 0; r < 2; ++r)
        cache[r] = std::make_unique<Sessions>(compat(r), r, budget, paths[r]);
    Pair pair;
    pair.run([&](unsigned r, const auto &ex) { cache[r]->capture(sample(r), ex); });
    // A capture is not durable and must not publish a group manifest.
    require(!std::filesystem::exists(d0 / "aa.rank0.group") && !std::filesystem::exists(d1 / "aa.rank1.group"),
            "RAM capture published disk manifest");
    pair.run([&](unsigned r, const auto &ex) { cache[r]->checkpoint("aa", ex); });
    const auto bytes0 = std::filesystem::file_size(d0 / "aa.rank0.group"), bytes1 = std::filesystem::file_size(d1 / "aa.rank1.group");
    require(bytes0 == bytes1 && bytes0 > 0, "both group manifests published");
    for (unsigned r = 0; r < 2; ++r)
        cache[r] = std::make_unique<Sessions>(compat(r), r, budget, paths[r]); // restart, no RAM
    pair.run([&](unsigned r, const auto &ex) {
        const auto c = cache[r]->resume("aa", true, ex);
        require(serialize_session(c) == serialize_session(sample(r)), "restart restored other bytes");
    });
    for (unsigned r = 0; r < 2; ++r) {
        const auto info = cache[r]->inspect("aa");
        require(!info.ram && info.disk && info.bytes > 0 && info.checkpoint.frontier.pending == 17, "disk inventory metadata");
        require(cache[r]->disk_bytes() > info.bytes, "disk accounting includes manifest");
    }
    pair.run([&](unsigned r, const auto &ex) { (void)cache[r]->resume("unknown", true, ex); }, false);
    for (const auto &error : pair.errors) {
        try { std::rethrow_exception(error); } catch (const std::exception &e) {
            require(std::string(e.what()).find("session_not_found") != std::string::npos, "missing error category on both ranks");
        }
    }
    pair.run([&](unsigned r, const auto &ex) { cache[r]->capture(cache[r]->resume("aa", true, ex), ex); });
    for (unsigned r = 0; r < 2; ++r)
        require(cache[r]->inspect("aa").disk && cache[r]->inspect("aa").ram, "disk restore admission retains durable generation");
    const auto file = d1 / "aa.generation1.rank1.state";
    std::ifstream f(file, std::ios::binary);
    SessionBytes original((std::istreambuf_iterator<char>(f)), {});
    f.close();
    const auto write = [&](const auto &p, const auto &b) {
        std::ofstream out(p, std::ios::binary);
        out.write(reinterpret_cast<const char *>(b.data()), static_cast<std::streamsize>(b.size()));
    };
    auto corrupt = original;
    corrupt[corrupt.size() / 2] ^= 1;
    write(file, corrupt);
    pair.run([&](unsigned r, const auto &ex) { (void)cache[r]->resume("aa", true, ex); }, false);
    for (const auto &error : pair.errors) {
        try { std::rethrow_exception(error); } catch (const std::exception &e) {
            require(std::string(e.what()).find("session_corrupt") != std::string::npos, "peer corruption error category");
        }
    }
    write(file, original);
    std::filesystem::rename(file, file.string() + ".missing");
    pair.run([&](unsigned r, const auto &ex) { (void)cache[r]->resume("aa", true, ex); }, false);
    std::filesystem::rename(file.string() + ".missing", file);
    auto short_file = original;
    short_file.resize(short_file.size() / 2);
    write(file, short_file);
    pair.run([&](unsigned r, const auto &ex) { (void)cache[r]->resume("aa", true, ex); }, false);
    write(file, original);
    // A different valid generation on only one disk cannot form a mixed restore.
    std::ifstream mf(d1 / "aa.rank1.group", std::ios::binary);
    SessionBytes old_manifest((std::istreambuf_iterator<char>(mf)), {});
    mf.close();
    pair.run([&](unsigned r, const auto &ex) {
        cache[r]->capture(sample(r, "aa", "generation2"), ex);
        require(!cache[r]->inspect("aa").disk, "new RAM generation cannot advertise stale disk durability");
        cache[r]->checkpoint("aa", ex);
    });
    write(d1 / "aa.rank1.group", old_manifest);
    pair.run([&](unsigned r, const auto &ex) { (void)cache[r]->resume("aa", true, ex); }, false);
    pair.run([&](unsigned r, const auto &ex) { cache[r]->capture(sample(r, "bb", r ? "generation2" : "generation1"), ex); }, false);
    require(!std::filesystem::exists(d0 / "bb.rank0.group"), "mismatch published manifest");
    // Durable write failure of rank 1 leaves rank 0's orphan state unpublished.
    const auto fail0 = dir / "fail0", fail1 = dir / "fail1";
    for (unsigned r = 0; r < 2; ++r)
        cache[r] = std::make_unique<Sessions>(compat(r), r, budget, r ? fail1 : fail0);
    pair.run([&](unsigned r, const auto &ex) { cache[r]->capture(sample(r), ex); });
    std::filesystem::remove(fail1);
    {
        std::ofstream blocker(fail1);
        blocker << "not a directory";
    }
    pair.run([&](unsigned r, const auto &ex) { cache[r]->checkpoint("aa", ex); }, false);
    require(!std::filesystem::exists(fail0 / "aa.rank0.group"), "manifest published before both rank files durable");
    pair.run([&](unsigned r, const auto &ex) { Sessions::agree(r == 0, ex, "import failure"); }, false);
    std::cout << "PASS restart/group atomicity/corruption/truncation/missing/mixed generation/import barrier\n";
}
void quota_tests(const std::filesystem::path& dir) {
    Pair pair;
    std::array<std::unique_ptr<Sessions>,2> cache;
    const auto n = serialize_session(sample(0)).size();
    // Enough space for the conservative publication reserve and one checkpoint,
    // but no second publication until the first ID is evicted.
    const auto cap = (4U << 20U) + n + 64U;
    for (unsigned r = 0; r < 2; ++r)
        cache[r] = std::make_unique<Sessions>(compat(r), r, 8*n, dir / ("quota"+std::to_string(r)), 1, cap);
    for (const auto& id : {"aa", "bb"}) {
        pair.run([&](unsigned r,const auto& ex) { cache[r]->capture(sample(r,id),ex); cache[r]->checkpoint(id,ex); });
    }
    for (unsigned r = 0; r < 2; ++r) {
        require(cache[r]->list() == std::vector<std::string>{"bb"}, "disk quota and max parked evict aa");
        require(cache[r]->stats().entries == 1 && cache[r]->disk_bytes() <= cap, "bounded cache accounting");
        require(!std::filesystem::exists(dir / ("quota"+std::to_string(r)) / ("aa.generation1.rank"+std::to_string(r)+".state")), "disk LRU removes payload");
    }
    pair.run([&](unsigned r,const auto& ex) { (void)cache[r]->resume("bb",true,ex); });
    // Updating an ID keeps its old committed checkpoint if quota cannot hold both.
    pair.run([&](unsigned r,const auto& ex) { cache[r]->capture(sample(r,"bb","generation2"),ex); });
    pair.run([&](unsigned r,const auto& ex) { cache[r]->checkpoint("bb",ex); },false);
    for (unsigned r = 0; r < 2; ++r)
        require(std::filesystem::exists(dir / ("quota"+std::to_string(r)) / ("bb.generation1.rank"+std::to_string(r)+".state")), "quota refusal preserves committed generation");
    std::cout << "PASS max parked and coordinated disk quota/LRU/refusal\n";
}
void incompatible_tests(const std::filesystem::path& dir) {
    Pair pair;
    const auto budget = 8U << 20U;
    std::array<std::unique_ptr<Sessions>, 2> cache;
    for (unsigned r = 0; r < 2; ++r)
        cache[r] = std::make_unique<Sessions>(compat(r), r, budget, dir / ("compat" + std::to_string(r)));
    pair.run([&](unsigned r, const auto& ex) { cache[r]->capture(sample(r), ex); cache[r]->checkpoint("aa", ex); });
    const auto check = [](auto&& f) {
        try { f(); } catch (const std::exception& e) {
            const auto error = strix::serve::session_failure(e.what());
            require(error.status == 409 && error.code == "session_incompatible" &&
                    std::string(e.what()).find("session_incompatible") != std::string::npos, "compatibility must be explicitly categorized 409");
            return;
        }
        throw std::runtime_error("expected incompatibility");
    };
    for (const auto& field : {"schema", "dspark", "model", "selector", "exe", "layers", "context", "world"}) {
        for (unsigned r = 0; r < 2; ++r) {
            auto cfg = compat(r); cfg.fields[field] = std::string(field) == "dspark" ? "on" : "changed";
            cache[r] = std::make_unique<Sessions>(cfg, r, budget, dir / ("compat" + std::to_string(r)));
            check([&] { (void)cache[r]->inspect("aa"); }); // explicit HTTP resume's preflight
        }
        pair.run([&](unsigned r, const auto& ex) { check([&] { (void)cache[r]->resume("aa", true, ex); }); });
    }
    auto envelope = serialize_session(sample(0));
    check([&] { (void)deserialize_session(envelope, compat(), 1, 0, budget); });
    check([&] { (void)deserialize_session(envelope, compat(), 2, 1, budget); });
    envelope[8] = 'X';
    check([&] { (void)deserialize_session(envelope, compat(), 2, 0, budget); });
    // An unsupported manifest schema must take the same path through inspect/resume.
    for (unsigned r = 0; r < 2; ++r) {
        cache[r] = std::make_unique<Sessions>(compat(r), r, budget, dir / ("compat" + std::to_string(r)));
        const auto path = dir / ("compat" + std::to_string(r)) / ("aa.rank" + std::to_string(r) + ".group");
        std::ifstream in(path, std::ios::binary); SessionBytes b((std::istreambuf_iterator<char>(in)), {}); in.close();
        b[8] = 'X'; { std::ofstream out(path, std::ios::binary | std::ios::trunc); out.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size())); }
        check([&] { (void)cache[r]->inspect("aa"); });
    }
    pair.run([&](unsigned r, const auto& ex) { check([&] { (void)cache[r]->resume("aa", true, ex); }); });
    std::cout << "PASS inspect/resume schema, DSpark mode, model and selector incompatibility/409\n";
}
void disk_worker(const std::filesystem::path& dir, const std::string& id, bool shared = false) {
    Pair pair;
    std::array<std::unique_ptr<Sessions>, 2> cache;
    const auto n = serialize_session(sample(0)).size();
    for (unsigned r = 0; r < 2; ++r)
        cache[r] = std::make_unique<Sessions>(compat(r), r, 8*n, dir / (shared ? "shared" : "rank"+std::to_string(r)), 1, (4U << 20U)+3*n+65536U);
    for (unsigned i = 0; i < 8; ++i) {
        pair.run([&](unsigned r, const auto& ex) {
            cache[r]->capture(sample(r, id, "generation"+std::to_string(i)), ex);
            cache[r]->checkpoint(id, ex);
            if (i % 3 == 2 || i == 7) cache[r]->erase(id, ex);
        });
    }
}
SessionBytes file_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "fixture open");
    return SessionBytes((std::istreambuf_iterator<char>(in)), {});
}
std::map<std::string, SessionBytes> publications(const std::filesystem::path& dir) {
    std::map<std::string, SessionBytes> out;
    for (const auto& e : std::filesystem::directory_iterator(dir))
        if (e.path().extension() == ".state" || e.path().extension() == ".group")
            out.emplace(e.path().filename().string(), file_bytes(e.path()));
    return out;
}
struct FaultGuard {
    io::Faults saved = io::faults();
    FaultGuard() { io::faults() = {}; }
    ~FaultGuard() { io::faults() = saved; }
};
void production_faults(const std::filesystem::path& root) {
    // Payload and manifest seams are armed in the actual rank thread. A descriptor
    // exchange follows the payload vote, and precedes manifest publication.
    unsigned index = 0;
    for (unsigned failed = 0; failed < 2; ++failed)
    for (bool manifest : {false, true})
    for (auto op : {io::Operation::create, io::Operation::write, io::Operation::sync,
                    io::Operation::replace, io::Operation::directory_sync})
    for (bool previous : {false, true}) {
        const auto dir = root / ("fault" + std::to_string(index++));
        Pair pair;
        std::array<std::filesystem::path,2> paths{dir/"r0",dir/"r1"};
        std::array<std::unique_ptr<Sessions>,2> cache;
        std::array<std::map<std::string,SessionBytes>,2> old;
        const auto fresh = [&] {
            for(unsigned r=0;r<2;++r) cache[r]=std::make_unique<Sessions>(compat(r),r,8U<<20U,paths[r]);
        };
        fresh();
        if(previous) {
            pair.run([&](unsigned r,const auto& ex){cache[r]->capture(sample(r),ex);cache[r]->checkpoint("aa",ex);});
            for(unsigned r=0;r<2;++r) old[r]=publications(paths[r]);
        }
        pair.run([&](unsigned r,const auto& ex){cache[r]->capture(sample(r,"aa","generation2"),ex);});
        std::array<bool,2> observed{};
        pair.run([&](unsigned r,const auto& ex) {
            FaultGuard guard;
            bool armed=false;
            const auto arm=[&] {
                if(r!=failed || armed) return;
                armed=true;
                if(op==io::Operation::write) {io::faults().max_write=3;io::faults().fail_write_at=7;}
                else io::faults().fail=op;
            };
            if(!manifest) arm();
            const auto wire=[&](std::span<const std::uint8_t> b) {
                const auto peer=ex(b);
                if(manifest && b.size()>1) arm();
                // Model a bounded one-shot failure. This explicitly clears the
                // injected fault before real rollback IO; it proves no persistent
                // media-failure guarantee.
                if(b.size()==1 && b[0]==0 && armed) { observed[r]=true;io::faults()={}; }
                return peer;
            };
            cache[r]->checkpoint("aa",wire);
        },false);
        require(observed[failed],"fault did not reach actual production failed IO vote");
        try {std::rethrow_exception(pair.errors[failed]);}
        catch(const io::Error& e) {require(e.operation==op,"wrong operation-specific family error");}
        catch(...) {throw std::runtime_error("lost operation-specific family error");}
        for(unsigned r=0;r<2;++r) {
            std::uint64_t used=0;
            for(const auto& e:std::filesystem::directory_iterator(paths[r])) {
                const auto name=e.path().filename().string();
                require(name.find(".tmp") == std::string::npos,"failed publication leaked its temporary");
                if(e.path().extension()==".state" || e.path().extension()==".group")used+=e.file_size();
            }
            require(cache[r]->disk_bytes()==used,"orphan/manifest quota bytes lost");
            if(previous) for(const auto& [name,bytes]:old[r])
                require(file_bytes(paths[r]/name)==bytes,"failed publication lost exact prior publication");
            else require(!std::filesystem::exists(paths[r]/("aa.rank"+std::to_string(r)+".group")),"failed fresh group became visible");
        }
        fresh();
        pair.run([&](unsigned r,const auto& ex) {
            auto c=cache[r]->resume("aa",true,ex);
            require(c.generation=="generation1" && c.frontier.pending==17,"failed replacement restored broken/new group");
        },previous);
        // Clear-fault retry restores the new generation, even with an already
        // published immutable orphan payload from the refused attempt.
        pair.run([&](unsigned r,const auto& ex){cache[r]->capture(sample(r,"aa","generation2"),ex);cache[r]->checkpoint("aa",ex);});
        fresh();
        pair.run([&](unsigned r,const auto& ex){require(cache[r]->resume("aa",true,ex).generation=="generation2","bounded retry did not restore new generation");});
    }
    // Read admission/allocation failures also stay on the same collective sequence.
    for(unsigned failed=0;failed<2;++failed) for(auto op:{io::Operation::allocation,io::Operation::read}) {
        const auto dir=root/("read"+std::to_string(index++));Pair pair;
        std::array<std::unique_ptr<Sessions>,2> cache;
        for(unsigned r=0;r<2;++r)cache[r]=std::make_unique<Sessions>(compat(r),r,8U<<20U,dir/("r"+std::to_string(r)));
        pair.run([&](unsigned r,const auto& ex){cache[r]->capture(sample(r),ex);cache[r]->checkpoint("aa",ex);});
        pair.run([&](unsigned r,const auto& ex){FaultGuard guard;if(r==failed)io::faults().fail=op;(void)cache[r]->resume("aa",true,ex);},false);
        pair.run([&](unsigned r,const auto& ex){require(cache[r]->resume("aa",true,ex).generation=="generation1","read retry changed group");});
    }
    std::cout<<"PASS production Sessions per-rank payload/manifest create/partial-write/sync/replace/directory-sync, real rollback, retry and read faults\n";
}

void quota_and_cleanup(const std::filesystem::path& root) {
    const auto n=serialize_session(sample(0)).size();
    for(bool enough:{false,true}) {
        Pair pair; const auto dir=root/(enough?"peak-enough":"peak-insufficient");
        const auto cap=(4U<<20U)+(enough?3:1)*n+64U;
        std::array<std::unique_ptr<Sessions>,2> cache;
        std::array<std::map<std::string,SessionBytes>,2> old;
        const auto fresh=[&] {for(unsigned r=0;r<2;++r)cache[r]=std::make_unique<Sessions>(compat(r),r,8*n,dir/("r"+std::to_string(r)),1,cap);};
        fresh();
        pair.run([&](unsigned r,const auto& ex){cache[r]->capture(sample(r),ex);cache[r]->checkpoint("aa",ex);});
        for(unsigned r=0;r<2;++r)old[r]=publications(dir/("r"+std::to_string(r)));
        pair.run([&](unsigned r,const auto& ex){cache[r]->capture(sample(r,"aa","generation2"),ex);cache[r]->checkpoint("aa",ex);},enough);
        for(unsigned r=0;r<2;++r) {
            require(cache[r]->disk_bytes()<=cap,"replacement inflated unchanged quota");
            if(!enough)require(publications(dir/("r"+std::to_string(r)))==old[r],"insufficient peak space changed prior bytes");
        }
        fresh();pair.run([&](unsigned r,const auto& ex){require(cache[r]->resume("aa",true,ex).generation==(enough?"generation2":"generation1"),"peak-space restart generation");});
    }
    // Shared-directory cleanup recognizes only exact ID/generation/rank grammar;
    // both spellings of orphan temporaries stay charged until family cleanup.
    Pair pair;const auto dir=root/"cleanup-shared";
    std::array<std::unique_ptr<Sessions>,2> cache;
    for(unsigned r=0;r<2;++r)cache[r]=std::make_unique<Sessions>(compat(r),r,8U<<20U,dir);
    pair.run([&](unsigned r,const auto& ex){cache[r]->capture(sample(r),ex);cache[r]->checkpoint("aa",ex);});
    const auto write=[&](const std::string& name){std::ofstream(dir/name,std::ios::binary)<<"1234567";};
    for(unsigned r=0;r<2;++r) {
        const auto rank=".rank"+std::to_string(r);
        const auto before=cache[r]->disk_bytes();
        for(const auto& stem:{"aa.old"+rank+".state","aa"+rank+".group"})
            for(const auto& tmp:{".tmp.legacy",".tmp-123-4"})write(stem+tmp);
        require(cache[r]->disk_bytes()==before+28,"historical/new temporary bytes uncharged");
        write("bb.old"+rank+".state.tmp-123-4");
        write("aa.invalid.extra"+rank+".state.tmp-123-4");
        write("aa.old"+rank+".state.tmp-123-4.extra");
    }
    write("target");
    std::error_code ec;std::filesystem::create_symlink("target",dir/"aa.link.rank0.state",ec);
    const bool symlink=!ec;
    if(!symlink)std::cout<<"NOT RUN symlink creation on this host: "<<ec.message()<<'\n';
    pair.run([&](unsigned r,const auto& ex){cache[r]->erase("aa",ex);});
    for(unsigned r=0;r<2;++r) {
        const auto rank=".rank"+std::to_string(r);
        require(cache[r]->disk_bytes()==7,"cleanup lost other ID/rank bytes or left owned temporary");
        require(std::filesystem::exists(dir/("bb.old"+rank+".state.tmp-123-4")),"cleanup deleted another ID");
        require(std::filesystem::exists(dir/("aa.invalid.extra"+rank+".state.tmp-123-4")),"cleanup accepted invalid generation");
        require(std::filesystem::exists(dir/("aa.old"+rank+".state.tmp-123-4.extra")),"cleanup accepted invalid temporary suffix");
        require(std::filesystem::exists(dir/(".session-rank"+std::to_string(r)+".lock")),"cleanup unlinked lock inode");
    }
    if(symlink)require(std::filesystem::is_symlink(dir/"aa.link.rank0.state") && file_bytes(dir/"target")==SessionBytes({'1','2','3','4','5','6','7'}),"cleanup deleted symlink/target");
    // Each rank's lock failure must vote before the next ordered acquisition.
    for(unsigned failed=0;failed<2;++failed) {
        const auto base=root/("lockfail"+std::to_string(failed));
        std::array<std::filesystem::path,2> paths{base/"r0",base/"r1"};
        for(unsigned r=0;r<2;++r)cache[r]=std::make_unique<Sessions>(compat(r),r,8U<<20U,paths[r]);
        pair.run([&](unsigned r,const auto& ex){cache[r]->capture(sample(r),ex);});
        std::filesystem::remove(paths[failed]);{std::ofstream(paths[failed])<<"block";}
        pair.run([&](unsigned r,const auto& ex){cache[r]->checkpoint("aa",ex);},false);
        std::filesystem::remove(paths[failed]);std::filesystem::create_directory(paths[failed]);
        pair.run([&](unsigned r,const auto& ex){cache[r]->checkpoint("aa",ex);});
    }
    std::cout<<"PASS unchanged peak quota, sufficient replacement, exact temporary ownership, ordered lock failures/retry\n";
}
SessionCheckpoint reference_sample(unsigned r,unsigned variant) {
    auto c=sample(r,variant==0?"aa":variant==1?"bb":"draft",variant==0?"generation1":"generation2");
    if(variant) {c.attention.next_token=c.frontier.pending=23;c.attention.tokens={4,5,6};c.frontier.emitted=3;}
    if(variant==2) {
        auto cfg=compat(r);cfg.fields["dspark"]="on";c.compat_record=cfg.record();
        c.draft.emplace();c.draft->context.next=3;c.draft->context.core=100;
        for(unsigned i=0;i<3;++i)c.draft->context.rings[i].resize(128U*528U,static_cast<std::uint8_t>(17+i));
        c.draft->context.extra={127000,101,129000};c.draft->controller.decisions=31;
        c.draft->exchange_request=65;c.draft->windows=13;c.draft->verified=22;c.draft->accepted=9;
        c.frontier.drafter=3;c.frontier.lineage="ds41-dspark-r1-committed-turn-v2";
    }
    return c;
}
void entry_publications(const std::filesystem::path& root,bool write,const std::filesystem::path& reference={}) {
    // Fixed before adoption from an independently built entry production Sessions
    // executable. Never regenerate expectations with the adopted implementation.
    const char* payload[3][2]={
        {"830389713c2f5a3d97b3e671e415731e463aadcccf167c30c631df3dabb28e9c","93e3aa0a3650314555fb3f87844403692df4c73a2a59d1e330c30a340a53f274"},
        {"f5d6734910a3c48c82271ab4ef3f61294900bf2d5a7b6ed79c4348b5c5cb454b","297b12d3515bcc56f54d5771c8df4b813af63e420977433b2ad2988db1e3f31e"},
        {"10566534f8bfb6a0492d16aa4a51a88cb0932cd825b79828e1d54a871e2852ab","4ef10070da470ae3c4374d8274d042c4573756340968945b90f1cf4d34b01558"}};
    const char* group[]={"91aaca4b8d61e604d52f17e11c2bc5b397d7a5513428a9ef1a7b8da8555c5560","beb48d1c3d2abd9fda6868cbeaf428d023518b48222d7c4609e480151cfb0650","436b27e17a6cd94a3da326428607424aaa2e540326c825cbf751b9a7359ca9ce"};
    for(unsigned v=0;v<3;++v) {
        Pair pair;std::array<std::unique_ptr<Sessions>,2> cache;
        for(unsigned r=0;r<2;++r) {
            auto cfg=compat(r);if(v==2)cfg.fields["dspark"]="on";
            cache[r]=std::make_unique<Sessions>(cfg,r,8U<<20U,root/("case"+std::to_string(v))/("rank"+std::to_string(r)));
        }
        const auto id=reference_sample(0,v).id;
        if(write)pair.run([&](unsigned r,const auto& ex){cache[r]->capture(reference_sample(r,v),ex);cache[r]->checkpoint(id,ex);});
        pair.run([&](unsigned r,const auto& ex){require(serialize_session(cache[r]->resume(id,true,ex))==serialize_session(reference_sample(r,v)),"entry reference full restored state/pending/draft");});
        for(unsigned r=0;r<2;++r)for(const auto& [name,bytes]:publications(root/("case"+std::to_string(v))/("rank"+std::to_string(r)))) {
            require(strix::model_content_identity_hex(strix::sha256_bytes(std::as_bytes(std::span(bytes))))==(name.ends_with(".group")?group[v]:payload[v][r]),"entry published filename/codec bytes changed");
            if(!reference.empty())require(bytes==file_bytes(reference/("case"+std::to_string(v))/("rank"+std::to_string(r))/name),"independent entry published bytes differ");
        }
    }
    std::cout<<"PASS entry fixed payload/group hashes, full restore and optional independent byte comparisons\n";
}

void persistent_rollback_faults(const std::filesystem::path& root) {
    for(unsigned failed=0;failed<2;++failed) {
        Pair pair;const auto dir=root/("persistent"+std::to_string(failed));
        std::array<std::unique_ptr<Sessions>,2> cache;
        for(unsigned r=0;r<2;++r)cache[r]=std::make_unique<Sessions>(compat(r),r,8U<<20U,dir/("r"+std::to_string(r)));
        pair.run([&](unsigned r,const auto& ex){cache[r]->capture(sample(r),ex);cache[r]->checkpoint("aa",ex);});
        pair.run([&](unsigned r,const auto& ex){cache[r]->capture(sample(r,"aa","generation2"),ex);});
        // The first fault occurs AFTER manifest replacement. Keep it active for
        // rollback and separately force rollback replacement to fail on that rank.
        pair.run([&](unsigned r,const auto& ex) {
            FaultGuard guard;bool manifest=false;
            const auto wire=[&](std::span<const std::uint8_t> b) {
                auto peer=ex(b);
                if(r==failed && b.size()>1) {manifest=true;io::faults().fail=io::Operation::directory_sync;}
                if(r==failed && manifest && b.size()==1 && b[0]==0)io::faults().fail=io::Operation::replace;
                return peer;
            };
            cache[r]->checkpoint("aa",wire);
        },false);
        for(auto e:pair.errors)try {std::rethrow_exception(e);}catch(const std::exception& why) {
            require(std::string(why.what()).find("durable commit uncertain")!=std::string::npos,"rollback failure lost durability uncertainty");
        }
        for(unsigned r=0;r<2;++r)cache[r]=std::make_unique<Sessions>(compat(r),r,8U<<20U,dir/("r"+std::to_string(r)));
        pair.run([&](unsigned r,const auto& ex){(void)cache[r]->resume("aa",true,ex);},false);
        // Repair uses real publication with the fault removed, under both locks.
        // This bounded retry does not certify rollback on persistently failed media.
        pair.run([&](unsigned r,const auto& ex){cache[r]->capture(sample(r,"aa","generation2"),ex);cache[r]->checkpoint("aa",ex);});
        pair.run([&](unsigned r,const auto& ex){require(cache[r]->resume("aa",true,ex).generation=="generation2","uncertain publication retry failed");});
        for(unsigned r=0;r<2;++r) for(const auto& e:std::filesystem::directory_iterator(dir/("r"+std::to_string(r))))
            require(e.path().filename().string().find(".tmp")==std::string::npos,"rollback failure leaked temporary");
    }
    std::cout<<"PASS post-replace uncertainty, failed real rollback, mixed-group rejection on both ranks and bounded repair\n";
}

void generation_and_temp_pressure(const std::filesystem::path& root) {
    const auto n=serialize_session(sample(0)).size();
    for(unsigned changed=0;changed<2;++changed) {
        Pair pair;const auto dir=root/("duplicate"+std::to_string(changed));
        std::array<std::unique_ptr<Sessions>,2> cache;
        std::array<std::map<std::string,SessionBytes>,2> old;
        const auto fresh=[&]{for(unsigned r=0;r<2;++r)cache[r]=std::make_unique<Sessions>(compat(r),r,8U<<20U,dir/("r"+std::to_string(r)));};
        fresh();pair.run([&](unsigned r,const auto& ex){cache[r]->capture(sample(r),ex);cache[r]->checkpoint("aa",ex);});
        for(unsigned r=0;r<2;++r)old[r]=publications(dir/("r"+std::to_string(r)));
        pair.run([&](unsigned r,const auto& ex){auto c=sample(r);if(r==changed)c.attention.layers[0].ring[100U*528U]^=1;cache[r]->capture(c,ex);});
        pair.run([&](unsigned r,const auto& ex){cache[r]->checkpoint("aa",ex);},false);
        try{std::rethrow_exception(pair.errors[changed]);}catch(const std::exception& why){require(std::string(why.what()).find("generation already exists with different bytes")!=std::string::npos,"duplicate generation failed for wrong reason");}
        for(unsigned r=0;r<2;++r)require(publications(dir/("r"+std::to_string(r)))==old[r],"duplicate generation replaced prior bytes");
        fresh();pair.run([&](unsigned r,const auto& ex){require(serialize_session(cache[r]->resume("aa",true,ex))==serialize_session(sample(r)),"duplicate generation damaged restore");});
    }
    for(const auto* temporary:{".tmp.legacy",".tmp-123-4"})for(unsigned pressured=0;pressured<2;++pressured) {
        Pair pair;const auto dir=root/(std::string("pressure")+temporary+std::to_string(pressured));
        const auto cap=(4U<<20U)+2*n+65536U;
        std::array<std::unique_ptr<Sessions>,2> cache;
        std::array<std::map<std::string,SessionBytes>,2> old;
        for(unsigned r=0;r<2;++r)cache[r]=std::make_unique<Sessions>(compat(r),r,8*n,dir/("r"+std::to_string(r)),1,cap);
        pair.run([&](unsigned r,const auto& ex){cache[r]->capture(sample(r),ex);cache[r]->checkpoint("aa",ex);});
        for(unsigned r=0;r<2;++r)old[r]=publications(dir/("r"+std::to_string(r)));
        const auto path=dir/("r"+std::to_string(pressured))/("orphan.generation.rank"+std::to_string(pressured)+".state"+temporary);
        const auto before=cache[pressured]->disk_bytes();
        {std::ofstream out(path,std::ios::binary);std::string bytes(70000,'x');out.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));}
        require(cache[pressured]->disk_bytes()==before+70000,"temporary forgotten before admission");
        pair.run([&](unsigned r,const auto& ex){cache[r]->capture(sample(r,"aa","generation2"),ex);cache[r]->checkpoint("aa",ex);},false);
        for(unsigned r=0;r<2;++r)require(publications(dir/("r"+std::to_string(r)))==old[r],"temporary pressure destroyed prior generation");
        require(std::filesystem::file_size(path)==70000,"refused transaction deleted another ID temporary");
        pair.run([&](unsigned r,const auto& ex){cache[r]->erase("orphan",ex);});
        pair.run([&](unsigned r,const auto& ex){cache[r]->checkpoint("aa",ex);});
        for(unsigned r=0;r<2;++r)require(cache[r]->disk_bytes()<=cap,"temporary retry changed quota");
        pair.run([&](unsigned r,const auto& ex){require(cache[r]->resume("aa",true,ex).generation=="generation2","temporary cleanup retry restore");});
    }
    std::cout<<"PASS duplicate-generation conflicts per rank and both temporary spellings under asymmetric quota pressure\n";
}

void lru() {
    const auto n = serialize_session(sample(0)).size();
    std::array<std::unique_ptr<Sessions>, 2> c;
    for (unsigned r = 0; r < 2; ++r)
        c[r] = std::make_unique<Sessions>(compat(r), r, 2 * n);
    Pair pair;
    for (const auto &id : {"aa", "bb"})
        pair.run([&](unsigned r, const auto &ex) { c[r]->capture(sample(r, id), ex); });
    pair.run([&](unsigned r, const auto &ex) { (void)c[r]->resume("aa", false, ex); });
    pair.run([&](unsigned r, const auto &ex) { c[r]->capture(sample(r, "cc"), ex); });
    for (unsigned r = 0; r < 2; ++r) {
        require(c[r]->list() == std::vector<std::string>({"aa", "cc"}), "LRU order");
        require(c[r]->stats().bytes == 2 * n && c[r]->stats().evictions == 1, "LRU byte accounting");
    }
    const auto info = c[0]->inspect("aa");
    require(info.ram && !info.disk && info.bytes == n && info.checkpoint.frontier.consumed == 3, "RAM inventory metadata");
    const auto checkpoint = sample(0);
    require(session_prompt_extends(checkpoint, std::vector<std::uint32_t>{1,2,3,17,18}), "extended history hit");
    require(session_prompt_extends(checkpoint, std::vector<std::uint32_t>{1,2,3,17}), "pending-only suffix hit");
    require(!session_prompt_extends(checkpoint, std::vector<std::uint32_t>{1,2,3}), "pending must be present");
    require(!session_prompt_extends(checkpoint, std::vector<std::uint32_t>{1,2,3,18}), "changed pending is a miss");
    require(!session_prompt_extends(checkpoint, std::vector<std::uint32_t>{9,2,3,17}), "changed history is a miss");
    pair.run([&](unsigned r, const auto &ex) { c[r]->erase("aa", ex); });
    require(c[0]->stats().bytes == n, "delete byte accounting");
    pair.run(
        [&](unsigned r, const auto &ex) {
            auto huge = sample(r, "dd", "generation1", 12);
            c[r]->capture(huge, ex);
        },
        false);
    require(c[0]->stats().bytes == n && c[1]->stats().bytes == n, "refused admission changed cache");
    std::cout << "PASS RAM LRU/accounting/oversize refusal\n";
}
} // namespace
int main(int argc, char** argv) {
    if (argc == 4 && (std::string_view(argv[1]) == "--disk-worker" || std::string_view(argv[1]) == "--disk-worker-shared")) {
        disk_worker(argv[2], argv[3], std::string_view(argv[1]) == "--disk-worker-shared"); return 0;
    }
    if (argc == 3 && (std::string_view(argv[1]) == "--codec-write" || std::string_view(argv[1]) == "--codec-read")) {
        entry_publications(argv[2], std::string_view(argv[1]) == "--codec-write"); return 0;
    }
    if (argc == 4 && std::string_view(argv[1]) == "--compat-probe") {
        auto c = session_compat({{"rank", argv[3]}, {"world", "2"}, {"dspark", "off"}},
                                {{"model", argv[2]}}, argv[0]);
        std::cout << c.key() << '\n';
        c.fields.erase("rank");
        std::cout << c.key() << '\n';
        return 0;
    }
    const auto dir = std::filesystem::temp_directory_path() / ("ds41-session-tests-" + session_generation());
    try {
        std::filesystem::create_directories(dir);
        envelopes();
        startup_failures();
        draft_envelopes();
        draft_groups(dir);
        loader_retarget(dir);
        full_hash(dir);
        group_tests(dir);
        lru();
        quota_tests(dir);
        incompatible_tests(dir);
        production_faults(dir);
        quota_and_cleanup(dir);
        persistent_rollback_faults(dir);
        generation_and_temp_pressure(dir);
        entry_publications(dir/"entry-bytes",true,argc>1?std::filesystem::path(argv[1]):std::filesystem::path{});
        std::filesystem::remove_all(dir);
        std::cout << "PASS all session cache CPU tests\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << " (artifacts " << dir << ")\n";
        return 1;
    }
}
