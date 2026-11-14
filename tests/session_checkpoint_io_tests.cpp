#include "strix/session/checkpoint_io.hpp"
#include "strix/session_state_cache.hpp"
#include "strix/minimax/session_cache.hpp"
#include "strix/model_content_hash.hpp"
#include "../apps/hip_session_store.hpp"
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <thread>
#include <type_traits>
#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace io = strix::session::checkpoint_io;
namespace mm = strix::minimax::session;
using namespace strix;
namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void refuses(F f) {
    bool failed = false; try { f(); } catch (const std::exception&) { failed = true; }
    check(failed, "expected refusal");
}
PrefixCacheBuffer heap(std::uint64_t n) {
    return {std::shared_ptr<std::byte>(new std::byte[n ? static_cast<std::size_t>(n) : 1], std::default_delete<std::byte[]>()), n};
}
struct Fixture {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("checkpoint-io-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Fixture() { check(std::filesystem::create_directory(path), "fixture directory"); }
    ~Fixture() { io::faults() = {}; std::error_code ec; std::filesystem::remove_all(path, ec); }
};
mm::Bytes read_bytes(const std::filesystem::path& p) { return mm::read(p, 16U << 20); }
SessionCheckpoint qwen(unsigned i = 0) {
    SessionCheckpoint q;
    q.compatibility = {sha256_bytes({}), sha256_bytes({}), "template;raw-thinking-history", i != 0};
    q.frontier = {3,2,3,i ? 3U : 0U,7,"greedy",{1,2,3}};
    auto b = heap(3); std::memcpy(b.data.get(), "raw", 3); q.sections.push_back({"state",b});
    return q;
}
mm::Checkpoint minimax(unsigned rank = 0, unsigned world = 2) {
    mm::Checkpoint c; c.id = "history"; c.generation = "frozen";
    c.compat = mm::compat_record({{"template","thinking/raw"}}); c.rank = rank; c.world = world;
    c.history = {1,2,3}; c.frontier.consumed = c.frontier.accepted = 3;
    c.frontier.emitted = 2; c.frontier.pending = 7; c.frontier.control = rank;
    c.sections = mm::split_state(mm::Bytes(3*76800,42),3);
    c.sections.push_back({"pending.logits",false,mm::Bytes((rank ? 99968 : 100096)*4,17)});
    return c;
}
void exact_codecs(const std::filesystem::path& frozen) {
    // Recorded from the independently archived entry-source executable before edits.
    const char* qhash[] = {"046eb502bdd1305f090e5ce51a33e36e344f6ada5dfe7862426b029f9fa37a38",
        "285b81c4031142971efce6a68dbe6c99c9e9eebbab4ce40a754179e992153f55"};
    const char* mhash[] = {"8a038e251af41eb9466f2134d0ae7bcd1c6321d18e9d98c45507a82dadfb9f8b",
        "7a9251728926c94e9c114bcd6b56fc1434d15030cf88356e82f6ae5d67edc6c9"};
    for (unsigned i = 0; i != 2; ++i) {
        const auto q = qwen(i); const auto blob = encode_session_checkpoint(q,heap);
        check(model_content_identity_hex(sha256_bytes(blob.span())) == qhash[i], "Qwen old codec bytes");
        const auto back = decode_session_checkpoint(blob,q.compatibility);
        check(back.frontier.tokens == q.frontier.tokens && back.frontier.pending == 7U, "Qwen pending/raw history");
        auto c = minimax(i); auto bytes = mm::encode(c);
        check(mm::sha(bytes) == mhash[i], "MiniMax old codec bytes");
        auto d = mm::decode(bytes,c.compat,i,2,bytes.size());
        check(d.history == c.history && d.frontier.pending == 7U && d.frontier.control == i, "MiniMax history/control");
        check(&d.history == &d.frontier.tokens && &d.compat == &d.compatibility.record, "nonowning legacy references");
        auto copy = d; auto moved = std::move(copy); moved.history.push_back(99);
        check(d.history.size() == 3 && moved.frontier.tokens.size() == 4, "adapter copy/move reference ownership");
        if (!frozen.empty()) {
            const auto oldq = read_bytes(frozen/("qwen"+std::to_string(i)+".bin"));
            check(oldq.size() == blob.bytes && std::memcmp(oldq.data(),blob.data.get(),oldq.size()) == 0, "exact entry Qwen file");
            auto oldblob = heap(oldq.size()); std::memcpy(oldblob.data.get(),oldq.data(),oldq.size());
            (void)decode_session_checkpoint(oldblob,q.compatibility);
            const auto oldm = read_bytes(frozen/("minimax"+std::to_string(i)+".bin"));
            check(oldm == bytes, "exact entry MiniMax file");
            (void)mm::decode(oldm,c.compat,i,2,oldm.size());
        }
    }
    if (frozen.empty()) std::cout << "NOT RUN: external entry binary files (fixed entry hashes checked)\n";
}
void prefix_adopter(const std::filesystem::path& root, const std::filesystem::path& frozen) {
    for(unsigned mode=0;mode<3;++mode) {
        const auto dir=root/("prefix"+std::to_string(mode));
        PrefixStateCachePolicy policy; policy.host_byte_budget=1U<<20; policy.disk_directory=dir;
        policy.disk_byte_budget=1U<<20; policy.legacy=mode==0; policy.durable=mode!=2;
        {
            PrefixStateCache cache(policy,heap);
            auto segment=std::make_shared<PrefixCacheSegment>();
            segment->identity=sha256_bytes({}); segment->parent=segment->identity;
            segment->block_rows=2; segment->tokens={3,4};
            segment->digest=prefix_cache_block_digest(segment->identity,segment->tokens);
            segment->payload=heap(3); std::memcpy(segment->payload.data.get(),"raw",3);
            auto c=std::make_shared<PrefixCacheCheckpoint>(); c->identity=segment->identity; c->digest=segment->digest;
            c->block_rows=2; c->position=2; c->segments={segment}; c->fixed=heap(3);
            std::memcpy(c->fixed.data.get(),"fix",3); c->metadata={std::byte{1},std::byte{2}};
            if(mode) c->producer_certificate={17,29};
            check(cache.insert(c),"actual prefix publication admission"); cache.flush();
            check(cache.stats().disk_writes==1 && cache.stats().disk_write_failures==0,"actual prefix publication success");
        }
        for(const auto& entry:std::filesystem::directory_iterator(dir)) {
            auto bytes=read_bytes(entry.path());
            const auto expected=entry.path().extension()==".tmps" ?
                "f4665ad027bdafe2e4e3ee7a141b3a8d06fa2546aae07153255c77d2fbe7f467" : mode==0 ?
                "cc8843ef3b2b2c9ab7b43ceb3c88175502fe9b6eb142e88997efc32e0484a0f9" :
                "06475569b025415d233c7fc27ea3861b341db082e40915dcd416037bddb32c2a";
            check(mm::sha(bytes)==expected,"prefix old format exact bytes");
            if(!frozen.empty()) check(bytes==read_bytes(frozen.parent_path()/"frozen-prefix"/("prefix"+std::to_string(mode))/entry.path().filename()),"entry prefix byte comparison");
        }
        PrefixStateCache reader(policy,heap);
        std::function<bool(const PrefixCacheCheckpoint&)> eligible;
        if(mode) eligible=[](const PrefixCacheCheckpoint& c){return c.producer_certificate==std::vector<std::uint64_t>{17,29};};
        auto hit=reader.find_longest(sha256_bytes({}),std::array<std::uint32_t,2>{3,4},2,2,false,{},eligible);
        check(hit && hit->metadata==std::vector<std::byte>({std::byte{1},std::byte{2}}) &&
            hit->producer_certificate==(mode ? std::vector<std::uint64_t>{17,29}:std::vector<std::uint64_t>{}),"prefix disk adapter and certificates");
    }
}
void rejection_tests() {
    auto q = qwen(); auto blob = encode_session_checkpoint(q,heap);
    for (unsigned k = 0; k != 4; ++k) {
        auto expected = q.compatibility;
        if (k == 0) expected.model.bytes[0] ^= 1;
        if (k == 1) expected.numerical_config += "other template";
        if (k == 2) expected.format_version = 2;
        if (k == 3) expected.family = "minimax";
        refuses([&] { (void)decode_session_checkpoint(blob,expected); });
    }
    auto c = minimax(); auto bytes = mm::encode(c);
    refuses([&] { (void)mm::decode(bytes,c.compat+"other model",0,2,bytes.size()); });
    refuses([&] { (void)mm::decode(bytes,c.compat,1,2,bytes.size()); });
    refuses([&] { (void)mm::decode(bytes,c.compat,0,1,bytes.size()); });
    for (auto bad : {0U,1U,2U,3U}) {
        auto d = c;
        if (bad == 0) d.sections.pop_back();
        if (bad == 1) d.sections[1].name = d.sections[0].name;
        if (bad == 2) d.sections[0].rank = 1;
        if (bad == 3) d.frontier.pending.reset();
        refuses([&] { (void)mm::encode(d); });
    }
    for (unsigned bad = 0; bad != 4; ++bad) {
        auto b = bytes;
        if (bad == 0) b.resize(71);
        if (bad == 1) b[0] = 255; // bounded invalid frame
        if (bad == 2) b[b.size()/2] ^= 1;
        if (bad == 3) b.push_back(0);
        refuses([&] { (void)mm::decode(b,c.compat,0,2,b.size()); });
    }
    // Generic rank-aware representation is broader than either legacy codec.
    SessionCheckpoint generic = c; generic.compatibility.family = "future"; generic.world = 4; generic.rank = 3;
    generic.frontier.selection = 12; generic.frontier.predictor = 2;
    for (auto& section : generic.sections) section.rank = 3;
    validate_session_envelope(generic);
    generic.sections[0].rank = 2; refuses([&] { validate_session_envelope(generic); });
    std::size_t cursor = 0; std::array<std::byte,3> frame{};
    refuses([&] { (void)io::take(frame,cursor,UINT64_MAX); }); check(cursor == 0, "bounds failure cursor unchanged");
    cursor = SIZE_MAX; refuses([&] { (void)io::take(frame,cursor,1); });
}
void io_tests(const std::filesystem::path& root) {
    const auto target = root/"parents"/"nested"/"payload";
    const std::array<std::byte,3> a{std::byte{1},std::byte{2},std::byte{3}}, b{std::byte{4},std::byte{5},std::byte{6}};
    refuses([&] { io::Publication bad(target,{true,io::DirectIo::buffered,0600,"../outside"}); });
    io::atomic_write(target,a);
    io::faults().max_write = 1; io::atomic_write(target,b); io::faults() = {};
    check(read_bytes(target) == mm::Bytes({4,5,6}), "short writes and replacement of existing target");
    for (const auto operation : {io::Operation::create,io::Operation::write,io::Operation::sync,io::Operation::replace}) {
        io::faults().fail = operation; refuses([&] { io::atomic_write(target,a); }); io::faults() = {};
        check(read_bytes(target) == mm::Bytes({4,5,6}), "failed publication preserves target");
        for (const auto& e : std::filesystem::directory_iterator(target.parent_path()))
            check(e.path() == target, "failed publication cleans exclusive temporary");
    }
    io::faults().fail_write_at = 1; refuses([&] { io::atomic_write(target,a); }); io::faults() = {};
    { io::Publication abandoned(target); abandoned.write(a); }
    check(read_bytes(target) == mm::Bytes({4,5,6}), "interrupted publication leaves old target");
    bool allocated = false;
    refuses([&] { io::read(target,0,2,[&](auto) { allocated = true; return std::span<std::byte>{}; }); });
    check(!allocated, "length refusal precedes allocation");
    for (const auto operation : {io::Operation::allocation,io::Operation::read}) {
        io::faults().fail = operation; refuses([&] { (void)read_bytes(target); }); io::faults() = {};
    }
    io::faults().fail = io::Operation::sync;
    io::atomic_write(target,a,{false,io::DirectIo::buffered,0600}); io::faults() = {};
    io::faults().fail = io::Operation::directory_sync;
    refuses([&] { io::atomic_write(target,b); }); io::faults() = {};
    // Post-replacement sync failure is uncertain durability, never a partial file.
    check(read_bytes(target) == mm::Bytes({4,5,6}), "directory sync failure preserves complete bytes");
    std::atomic<bool> failed{false};
    auto publish = [&](std::byte value) {
        try { std::vector<std::byte> bytes(4096,value); for (unsigned i=0;i<10;++i) io::atomic_write(target,bytes); }
        catch (...) { failed = true; }
    };
    std::thread first(publish,std::byte{11}), second(publish,std::byte{22}); first.join(); second.join();
    auto bytes = read_bytes(target);
    check(!failed && bytes.size()==4096 && std::all_of(bytes.begin(),bytes.end(),[&](auto v){return v==bytes[0];}), "concurrent full publication");
    const auto locks = root/"locks";
    {
        io::DirectoryLock held(locks,io::LockMode::blocking);
        refuses([&] { io::DirectoryLock contender(locks,io::LockMode::nonblocking); });
        // Both real family wrappers use this same persistent lock.
        auto c = minimax(0,1); mm::Cache cache(c.compat,0,1,32U<<20,locks,[](auto s){return mm::Bytes(s.begin(),s.end());});
        refuses([&] { cache.checkpoint(c); });
    }
    check(std::filesystem::exists(locks/".session.lock"), "persistent lock file");
#ifndef _WIN32
    struct stat before{}, after{}; check(::stat((locks/".session.lock").c_str(),&before)==0,"stat lock");
    { io::DirectoryLock held(locks,io::LockMode::nonblocking); }
    check(::stat((locks/".session.lock").c_str(),&after)==0 && before.st_ino==after.st_ino,"persistent POSIX lock inode");
    std::cout << "RUN: POSIX directory fsync and flock\n";
#else
    std::cout << "NOT RUN: POSIX flock/directory fsync; RUN: Windows lock and atomic replacement\n";
#endif
    auto held = std::make_unique<strix::decode_cli::SessionDirectoryLock>(locks);
    std::promise<void> entered; auto ready = entered.get_future();
    std::atomic<bool> acquired{false};
    std::thread waiting([&] { entered.set_value(); strix::decode_cli::SessionDirectoryLock lock(locks); acquired = true; });
    ready.wait(); std::this_thread::sleep_for(std::chrono::milliseconds(20));
    check(!acquired, "Qwen blocking lock contention"); held.reset(); waiting.join(); check(acquired,"Qwen lock release");
}
struct Pair {
    std::barrier<> sync{2}; std::array<mm::Bytes,2> messages;
    mm::Exchange at(unsigned rank) { return [this,rank](std::span<const std::uint8_t> b) {
        messages[rank].assign(b.begin(),b.end()); sync.arrive_and_wait(); auto peer = messages[1-rank]; sync.arrive_and_wait(); return peer;
    }; }
};
template<class F> void pair_run(F f) {
    std::array<std::exception_ptr,2> errors;
    std::thread a([&]{try { f(0); } catch (...) { errors[0]=std::current_exception(); }});
    std::thread b([&]{try { f(1); } catch (...) { errors[1]=std::current_exception(); }});
    a.join(); b.join(); for (auto e : errors) if(e) std::rethrow_exception(e);
}
void real_adopters(const std::filesystem::path& root) {
    const auto q = qwen(); const auto blob = encode_session_checkpoint(q,heap);
    strix::decode_cli::HttpSessionStore store(root/"qwen",8U<<20,8,heap,q.compatibility,16U<<20);
    store.park("history",blob); (void)store.command("checkpoint","history","disk");
    strix::decode_cli::HttpSessionStore restart(root/"qwen",8U<<20,8,heap,q.compatibility,16U<<20);
    auto hit = restart.matching("history",std::array<std::uint32_t,4>{1,2,3,7});
    check(hit && hit->frontier.pending==7U,"actual Qwen named-session durable restore");
    check(!restart.matching("history",std::array<std::uint32_t,4>{1,2,99,7}),"thinking/raw history miss");
    SessionCheckpoint untouched = q; untouched.frontier.pending=333;
    for (auto op : {io::Operation::allocation,io::Operation::read}) {
        io::faults().fail=op;
        refuses([&]{(void)read_session_checkpoint(root/"qwen"/"history",q.compatibility,heap,blob.bytes,nullptr,&untouched);});
        io::faults()={}; check(untouched.frontier.pending==333U,"rejected read never publishes decoded state");
    }
    refuses([&]{(void)read_session_checkpoint(root/"qwen"/"history",q.compatibility,
        [](auto)->PrefixCacheBuffer {throw std::bad_alloc();},blob.bytes,nullptr,&untouched);});
    check(untouched.frontier.pending==333U,"allocation failure never publishes decoded state");
    Pair pair; std::array<std::unique_ptr<mm::Cache>,2> cache;
    for (unsigned rank=0;rank<2;++rank) cache[rank]=std::make_unique<mm::Cache>(minimax().compat,rank,2,32U<<20,root/("mm"+std::to_string(rank)),pair.at(rank));
    // Control/replicated state must agree across ranks; local logits shards differ.
    pair_run([&](unsigned rank){auto c=minimax(rank); c.frontier.control=1; cache[rank]->checkpoint(c);});
    std::array<mm::Bytes,2> last;
    for(unsigned rank=0;rank<2;++rank) last[rank]=read_bytes(root/("mm"+std::to_string(rank))/"history.group");
    for (unsigned failure=0;failure<2;++failure) {
        pair_run([&](unsigned rank){
            if(rank==1) { io::faults().fail= failure ? io::Operation::replace : io::Operation::write; io::faults().skip=failure ? 1U:0U; }
            auto c=minimax(rank); c.frontier.control=1; c.frontier.pending=8;
            refuses([&]{cache[rank]->checkpoint(c);}); io::faults()={};
        });
        for(unsigned rank=0;rank<2;++rank) check(read_bytes(root/("mm"+std::to_string(rank))/"history.group")==last[rank],"failed rank preserves last good group");
        pair_run([&](unsigned rank){ auto c=cache[rank]->resume("history",true); check(c.frontier.pending==7U,"old generation still resumes"); });
    }
    pair_run([&](unsigned rank){cache[rank]->erase("history");});
    for(unsigned rank=0;rank<2;++rank) check(!cache[rank]->on_disk("history"),"actual MiniMax erase");
}
}
int main(int argc,char** argv) {
    static_assert(std::is_same_v<mm::Frontier,SessionFrontier>);
    static_assert(std::is_same_v<mm::Section,SessionSection>);
    static_assert(std::is_base_of_v<SessionCheckpoint,mm::Checkpoint>);
    try {
        Fixture fixture; exact_codecs(argc>1 ? std::filesystem::path(argv[1]) : std::filesystem::path{});
        prefix_adopter(fixture.path,argc>1 ? std::filesystem::path(argv[1]) : std::filesystem::path{});
        rejection_tests(); io_tests(fixture.path); real_adopters(fixture.path);
        std::cout << "session_checkpoint_io_tests PASS\n"; return 0;
    } catch (const std::exception& e) { std::cerr << "session_checkpoint_io_tests FAIL: " << e.what() << '\n'; return 1; }
}
