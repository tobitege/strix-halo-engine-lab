#include <cstdlib>
#include <source_location>
#include <string>
#include <iostream>

#include "glm/rtxpf/import/glm_import_socket.hpp"
using namespace strix::glm::import;
namespace {
unsigned checks = 0;
void check(bool ok, std::source_location at = std::source_location::current()) {
    ++checks;
    if (!ok) throw std::runtime_error("CPU assertion failed at line " + std::to_string(at.line()));
}
template <class F>
void rejects(F f, std::source_location at = std::source_location::current()) {
    bool threw = false;
    try {
        f();
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, at);
}
Bytes frame(const Component& c, std::uint64_t row = 0, std::uint64_t rows = 0) {
    if (!rows) rows = c.rows;
    Bytes payload(static_cast<std::size_t>(rows * c.stride), std::byte{0x5a});
    Bytes b;
    number(b, 2, 4);
    number(b, c.layer, 4);
    number(b, c.id, 4);
    number(b, row);
    number(b, rows);
    number(b, payload.size());
    digest(b, hash(payload));
    append(b, payload);
    return b;
}
void add(Staging& s, const Bytes& b) {
    Reader r{b};
    check(r.number(4) == 2);
    s.frame(r, b);
}
void units() {
    Begin b;
    b.frontier = 5;
    b.total = 6;
    b.producer = "glm-pf-node-replay-v1";
    b.node = "r1";
    auto bytes = b.encode();
    Reader br{bytes};
    check(br.number(4) == 1);
    auto parsed = Begin::parse(br);
    check(parsed.encode() == bytes);
    check(layout(b).size() == 116);  // 34 KDA x2 + 12 DSA x4
    // Version 2 ('DSA once'): rank 0 keeps every component, rank 1 the 68 KDA components only; same encoding.
    {
        auto v2 = b;
        v2.version = 2;
        const auto all = layout(b), r0 = layout(v2);
        check(r0.size() == all.size());
        for (std::size_t i = 0; i < all.size(); ++i)
            check(r0[i].layer == all[i].layer && r0[i].id == all[i].id && r0[i].rows == all[i].rows && r0[i].stride == all[i].stride);
        v2.rank = 1;
        const auto r1 = layout(v2);
        check(r1.size() == 68);
        for (const auto& c : r1) check(c.id < 2 && c.layer < 45 && c.layer % 4 != 3);
        auto v1r1 = b;
        v1r1.rank = 1;
        check(layout(v1r1).size() == 116);
        auto enc = v2.encode();
        Reader vr{enc};
        check(vr.number(4) == 1);
        auto back = Begin::parse(vr);
        check(back.version == 2 && back.encode() == enc);
        check(hash(v2.encode(true)) != hash(b.encode(true)));  // the version is in the ranks' binding digest
        Staging s1(v2);
        auto v2r0 = b;
        v2r0.version = 2;
        rejects([&] { validate(b, v2r0); });  // a v1 sender to a receiver expecting v2
        rejects([&] { agree(hash(v2r0.encode(true)), true, vote(hash(b.encode(true)), true)); });  // mixed-version ranks
        for (unsigned layers : {45u, 46u})
            for (std::uint64_t n : {1ULL, 3ULL, 4ULL, 8ULL}) {  // pooled rows 0 and > 0
                auto q = v2;
                q.layers = layers;
                q.frontier = n;
                q.total = n + 1;
                auto full = q;
                full.version = 1;
                std::vector<std::pair<unsigned, unsigned>> kept, want;
                for (const auto& c : layout(full))
                    if (!omitted(q.version, q.rank, c.id)) kept.push_back({c.layer, c.id});
                for (const auto& c : layout(q)) want.push_back({c.layer, c.id});
                check(kept == want);  // the producer's filter of the full export == rank 1's v2 layout
            }
    }
    // Versions 3 / 4 (index tail): ikey / igate carry only the open pool block [P - P % 4, P); everything else is v1 / v2.
    {
        namespace st = strix::state;
        check(wire_version(false, false) == 1 && wire_version(true, false) == 2 && wire_version(false, true) == 3 &&
              wire_version(true, true) == 4);
        check(!index_tail(1) && !index_tail(2) && index_tail(3) && index_tail(4) && dsa_once(2) && dsa_once(4) && !dsa_once(3));
        for (unsigned layers : {45u, 46u})
            for (std::uint64_t n : {1ULL, 2ULL, 3ULL, 4ULL, 5ULL, 6ULL, 7ULL, 8ULL, 2047ULL, 2048ULL, 2049ULL, 2050ULL, 2051ULL})
                for (unsigned rank : {0u, 1u})
                    for (bool once : {false, true}) {
                        auto full = b, tail = b;
                        full.layers = tail.layers = layers;
                        full.frontier = tail.frontier = n;
                        full.total = tail.total = n + 1;
                        full.rank = tail.rank = rank;
                        full.version = wire_version(once, false);
                        tail.version = wire_version(once, true);
                        const auto f = layout(full), t = layout(tail);
                        std::vector<Component> want;  // the full layout with the index rows cut to the open tail
                        for (const auto& c : f) {
                            if (c.id != 3 && c.id != 4) { want.push_back(c); continue; }
                            check(c.rows == n && c.first == 0);
                            if (n % 4) want.push_back({c.layer, c.id, n % 4, c.stride, n - n % 4});
                        }
                        bool same = want.size() == t.size();
                        for (std::size_t i = 0; same && i < t.size(); ++i)
                            same = t[i].layer == want[i].layer && t[i].id == want[i].id && t[i].rows == want[i].rows &&
                                   t[i].stride == want[i].stride && t[i].first == want[i].first;
                        check(same);
                        // the producer's filter of its v3 export == the rank's v4 layout (DSA once composes with the tail)
                        if (once) {
                            auto v3 = tail;
                            v3.version = 3;
                            std::vector<std::pair<unsigned, unsigned>> kept, got;
                            for (const auto& c : layout(v3))
                                if (!omitted(tail.version, rank, c.id)) kept.push_back({c.layer, c.id});
                            for (const auto& c : t) got.push_back({c.layer, c.id});
                            check(kept == got);
                        }
                        // shared descriptor: the cut equals open_tail_saving per DSA layer
                        std::uint64_t fb = 0, tb = 0;
                        for (const auto& c : f) fb += c.rows * c.stride;
                        for (const auto& c : t) tb += c.rows * c.stride;
                        const unsigned dsa_layers = omitted(tail.version, rank, 3) ? 0U : (layers == 46 ? 12U : 11U);
                        check(fb - tb == dsa_layers * st::open_tail_saving(strix::glm::state_layout::kDsaIndex, n));
                    }
        // Every frame of a v4 rank-0 layout stages; the tail frames start at row 0 of their component.
        for (std::uint64_t n : {6ULL, 8ULL}) {
            auto v4 = b;
            v4.version = 4;
            v4.frontier = n;
            v4.total = n + 1;
            Staging s(v4);
            std::uint64_t expect = 0;
            for (const auto& c : layout(v4)) {
                add(s, frame(c));
                expect += c.rows * c.stride;
            }
            check(s.raw_bytes == expect);
        }
        // MINSTATE GLM pair totals (both ranks, MTP, N = P + 1): v2 806,920,192 / v4 605,618,176 B at 32K,
        // v4 1,964,572,672 B at 128K (13,824 N + Fg for N multiple of 4; Fg = 152,633,344).
        auto pair_bytes = [&](unsigned version, std::uint64_t N) {
            std::uint64_t total = 0;
            for (unsigned rank : {0u, 1u}) {
                auto q = b;
                q.version = version;
                q.rank = rank;
                q.layers = 46;
                q.frontier = N - 1;
                q.total = N;
                for (const auto& c : layout(q)) total += c.rows * c.stride;
            }
            return total;
        };
        check(pair_bytes(2, 32768) == 806920192ULL && pair_bytes(4, 32768) == 605618176ULL);
        check(pair_bytes(4, 131072) == 1964572672ULL && pair_bytes(2, 131072) - pair_bytes(4, 131072) == 805281792ULL);
        auto v5 = b;
        v5.version = 5;
        rejects([&] { Staging s(v5); });
        auto v4 = b;
        v4.version = 4;
        check(hash(v4.encode(true)) != hash(b.encode(true)));  // the version is in the ranks' binding digest
    }
    // The same helpers size/reset/serialize/restore the engine's actual allocations.
    // Exercise all pool boundaries, including a capacity different from the frontier.
    namespace sl = strix::glm::state_layout;
    check(sl::kConvFloatsPerHead == 3 * 128 * 3);
    check(sl::conv_floats(32) == 32 * 3 * 128 * 3);
    for (std::uint64_t cap : {4ULL, 8ULL, 2048ULL}) {
        check(sl::pooled_rows(cap) == cap / strix::glm::kKpool);
        for (std::uint64_t n : {1ULL, 3ULL, 4ULL, 5ULL, 7ULL, 8ULL}) {
            if (n > cap) continue;
            auto probe = b;
            probe.frontier = n;
            probe.total = n + 1;
            unsigned conv = 0, pooled = 0;
            for (const auto& c : layout(probe)) {
                if (c.id == 1) {
                    check(c.rows == 32 && c.stride == sl::kConvFloatsPerHead * sizeof(float));
                    check(c.rows * c.stride == sl::conv_floats(32) * sizeof(float));
                    ++conv;
                }
                if (c.id == 5) {
                    check(c.rows == sl::pooled_rows(n) && c.rows <= sl::pooled_rows(cap));
                    check(c.stride == strix::glm::kIdxDim * sizeof(float));
                    // Slot one begins beyond the whole capacity, not beyond the serialized prefix.
                    const auto slot_bytes = sl::pooled_rows(cap) * c.stride;
                    check(slot_bytes + c.rows * c.stride <= 2 * slot_bytes);
                    ++pooled;
                }
            }
            check(conv == 34 && pooled == (sl::pooled_rows(n) ? 12U : 0U));
        }
    }
    auto invalid = b;
    invalid.frontier = 0;
    rejects([&] { Staging s(invalid); });
    invalid = b;
    invalid.rank = 2;
    rejects([&] { Staging s(invalid); });
    invalid = b;
    invalid.version = 0;  // 1..4 are known (3 / 4: the index tail)
    rejects([&] { Staging s(invalid); });
    for (unsigned field = 0; field < 9; ++field) {
        auto x = b;
        switch (field) {
            case 0:
                x.rank = 1;
                break;
            case 1:
                x.model.bytes[0] = 1;
                break;
            case 2:
                x.prompt.bytes[0] = 1;
                break;
            case 3:
                x.frontier = 4;
                break;
            case 4:
                x.layers = 45;
                break;
            case 5:
                x.node = "r1b";
                break;
            case 6:
                x.total = 7;
                break;
            case 7:
                x.version = 2;
                break;
            default:
                x.producer = "bad";
        }
        rejects([&] { validate(x, b); });
    }
    const auto binding = hash(b.encode(true));
    agree(binding, true, vote(binding, true));
    check(true);
    rejects([&] { agree(binding, true, vote(binding, false)); });
    rejects([&] { agree(binding, false, vote(binding, true)); });
    auto other = b;
    other.transaction.bytes[0] = 1;
    rejects([&] { agree(binding, true, vote(hash(other.encode(true)), true)); });
    auto changed_rank = b;
    changed_rank.rank = 1;
    check(hash(changed_rank.encode(true)) == binding);
    const auto components = layout(b);
    auto first = frame(components[0]);
    {
        Staging s(b);
        add(s, first);
        rejects([&] { add(s, first); });
    }
    {
        Staging s(b);
        rejects([&] { add(s, frame(components[1])); });
    }
    {
        Staging s(b);
        auto bad = first;
        bad.back() ^= std::byte{1};
        rejects([&] { add(s, bad); });
    }
    {
        Staging s(b);
        auto bad = first;
        bad.pop_back();
        rejects([&] { add(s, bad); });
    }
    {
        Staging s(b);
        rejects([&] { add(s, frame(components[0], 31, 2)); });
    }
    {
        Staging s(b);
        Bytes c;
        number(c, 0);
        number(c, 0);
        Reader r{c};
        rejects([&] { s.commit(r); });
        rejects([&] { s.apply([](auto&, auto) {}); });
    }
    Staging s(b);
    for (const auto& c : components) {
        auto n = c.rows / 2;
        if (n) {
            add(s, frame(c, 0, n));
            add(s, frame(c, n, c.rows - n));
        } else
            add(s, frame(c));
    }
    Bytes manifest;
    number(manifest, s.frames.size());
    number(manifest, s.raw_bytes);
    for (const auto& f : s.frames) digest(manifest, f.sha);
    auto bad = manifest;
    bad.back() ^= std::byte{1};
    Reader rr{bad};
    rejects([&] { s.commit(rr); });
    bad = manifest;
    bad[0] ^= std::byte{1};
    Reader count{bad};
    rejects([&] { s.commit(count); });
    Reader good{manifest};
    s.commit(good);
    check(s.committed);
    Reader again{manifest};
    rejects([&] { s.commit(again); });
    rejects([&] { add(s, first); });
    std::uint64_t applied = 0;
    s.apply([&](const auto& f, auto data) {
        check(data.size() == f.bytes && data.front() == 0x5a && data.back() == 0x5a);
        applied += data.size();
    });
    check(applied == s.raw_bytes);
    // RFC 4231 HMAC-SHA256 test case 1, extended with zero key bytes (same block).
    Bytes key(32);
    std::fill_n(key.begin(), 20, std::byte{0x0b});
    std::string hi = "Hi There";
    check(hmac(key, std::as_bytes(std::span(hi))) ==
          unhex("b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"));
    auto k = hmac(key, {});
    auto k2 = k;
    k2.bytes[31] ^= 1;
    check(secure_equal(k, k) && !secure_equal(k, k2));
    std::cout << "PASS " << checks << " CPU checks\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            units();
            return 0;
        }
        if (std::string(argv[1]) == "--snapshot") {
            // Offline decode-identity snapshot check: exact receiver layout and
            // payload hash (16-byte header excluded); no producer identity or socket.
            require(argc == 6, "usage: test --snapshot FILE RANK P N");
            Begin b;
            b.rank = unsigned(std::stoul(argv[3]));
            b.layers = 46;
            b.frontier = std::stoull(argv[4]);
            b.total = std::stoull(argv[5]);
            std::uint64_t expected = 0;
            for (const auto& c : layout(b)) expected += c.rows * c.stride;
            std::ifstream in(argv[2], std::ios::binary);
            require(bool(in), "open snapshot");
            Bytes header(16);
            require(bool(in.read(reinterpret_cast<char*>(header.data()), 16)), "snapshot header");
            Reader hr{header};
            require(hr.number() == b.frontier && hr.number() == b.total, "snapshot frontier/total");
            strix::Sha256Fast sha;
            std::vector<char> buffer(1 << 20);
            std::uint64_t got = 0;
            while (in) {
                in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
                auto n = static_cast<std::size_t>(in.gcount());
                if (!n) break;
                sha.update(std::as_bytes(std::span(buffer.data(), n)));
                got += n;
            }
            require(got == expected, "snapshot payload extent differs from receiver layout");
            std::cout << "SNAPSHOT " << strix::model_content_identity_hex(sha.finalize()) << " " << got << "\n";
            return 0;
        }
        require(argc == 5 || argc == 6 || argc == 8, "usage: test ENDPOINT KEY PROMPT_SHA MODEL_SHA [RANK [P N]]");
        Begin b;
        b.rank = argc>=6?unsigned(std::stoul(argv[5])):0;
        b.layers = 46;
        b.frontier = argc==8?std::stoull(argv[6]):5;
        b.total = argc==8?std::stoull(argv[7]):6;
        b.node = "r1";
        b.prompt = unhex(argv[3]);
        b.model = unhex(argv[4]);
        if (const char* v = std::getenv("GLM_TEST_WIRE_VERSION")) b.version = unsigned(std::stoul(v));  // 2 = DSA once
        auto s = receive(argv[1], read_key(argv[2]), b);
        strix::Sha256Fast sha;
        s->apply([&](const Frame&, auto data) { sha.update(std::as_bytes(data)); });
        std::cout << "SEALED " << strix::model_content_identity_hex(sha.finalize()) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "REJECT " << e.what() << "\n";
        return 1;
    }
}
