#include "strix/deepseek41_prefix_cache.hpp"
#include "strix/deepseek41_attention_host.hpp"
#include "strix/ds41_attention.hpp"
#include <cstring>
#include <iostream>
#include <string>
#include "strix/session/checkpoint_io.hpp"
#include "strix/unbuffered_file.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <thread>

namespace ds = strix::deepseek41;
namespace fs = std::filesystem;
namespace attn = strix::ds41::attn;
namespace io = strix::session::checkpoint_io;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
std::vector<std::uint32_t> tokens(bool branch = false) {
    std::vector<std::uint32_t> t(33);
    for (unsigned i=0;i<t.size();++i) t[i]=13+i*17+(branch && i>=16 ? 101U:0U);
    return t;
}
ds::AttentionState sample(const std::vector<std::uint32_t>& t, unsigned p, bool unknown=false) {
    ds::AttentionState s; s.position=p; s.tokens.assign(t.begin(),t.begin()+p);
    s.next_token=unknown ? ds::kPrefixUnknownNextToken : t[p-1]+19;
    for(unsigned l=0;l<4;++l) {
        ds::AttentionLayerState v; v.owns_kv=l<3; v.owns_index=l<2; v.ratio=l==1?2U:l==2?8U:1U;
        v.ring.resize(ds::AttentionHost::ring_capacity*attn::kWindowEntryBytes);
        for(std::size_t j=0;j<v.ring.size();++j) v.ring[j]=static_cast<std::uint8_t>(j*7+l*11+t[p-1]);
        if(v.owns_kv) {
            v.kv.resize((p/v.ratio)*attn::kKvEntryBytes); v.idx_keys.resize((p/v.ratio)*attn::kIdxEntryBytes);
            for(unsigned row=0;row<p/v.ratio;++row) {
                for(unsigned b=0;b<attn::kKvEntryBytes;++b) v.kv[row*attn::kKvEntryBytes+b]=static_cast<std::uint8_t>(t[(row+1)*v.ratio-1]+b+l);
                for(unsigned b=0;b<attn::kIdxEntryBytes;++b) v.idx_keys[row*attn::kIdxEntryBytes+b]=static_cast<std::uint8_t>(t[(row+1)*v.ratio-1]+3*b+l);
            }
            if(v.ratio==2) { v.carry.resize(2*attn::kHeadDim); for(unsigned j=0;j<v.carry.size();++j) v.carry[j]=static_cast<float>(j+3)/8; }
        }
        v.topk_known=false; s.layers.push_back(std::move(v));
    }
    return s;
}
void exact(const ds::AttentionState& a,const ds::AttentionState& b) {
    check(a.position==b.position && a.next_token==b.next_token && a.tokens==b.tokens,"exact frontier/history/next");
    check(a.layers.size()==b.layers.size(),"exact layers");
    for(std::size_t l=0;l<a.layers.size();++l) {
        const auto& x=a.layers[l]; const auto& y=b.layers[l];
        check(x.owns_kv==y.owns_kv && x.owns_index==y.owns_index && x.ratio==y.ratio,"exact layout");
        check(x.kv==y.kv && x.idx_keys==y.idx_keys && x.ring==y.ring,"exact all KV/index/ring bytes");
        check(x.carry_valid==y.carry_valid && x.carry.size()==y.carry.size() &&
            (x.carry.empty() || std::memcmp(x.carry.data(),y.carry.data(),x.carry.size()*sizeof(float))==0),"exact all carry bytes");
        check(!x.topk_known && x.topk.empty(),"topk omitted as contracted");
    }
}
ds::PrefixCacheOptions options(const fs::path& p,bool drop) {
    ds::PrefixCacheOptions o; o.dir=p; o.block=8; o.drop_page_cache=drop; o.model_id[0]=73; o.io_threads=1; return o;
}
void capture(const fs::path& root) {
    check(!fs::exists(root),"capture destination must be new");
    for(bool drop:{false,true}) {
        ds::PrefixCache c(options(root/(drop?"drop":"keep"),drop));
        check(c.insert(sample(tokens(),8),101),"entry insert 8");
        check(c.insert(sample(tokens(),16,true),0),"entry insert unknown16");
        check(c.insert(sample(tokens(),24),103),"entry insert 24");
        check(c.insert(sample(tokens(true),24),104),"entry insert branch24");
        c.flush(); check(c.stats().write_errors==0 && c.stats().written_files==8,"entry 4 blocks/4 snapshots");
    }
}
void verify(const fs::path& root) {
    for(bool drop:{false,true}) {
        ds::PrefixCache c(options(root/(drop?"drop":"keep"),drop));
        check(c.stats().scanned_files==8 && c.stats().ignored_files==0 && c.inventory().size()==8,"restart inventory 8");
        for(bool branch:{false,true}) {
            const auto t=tokens(branch); const auto h=c.lookup(t);
            check(h.resume==24 && h.shared==24 && h.disk && h.next_token==sample(t,24).next_token && h.logits_fnv==(branch?104U:103U),"restart deepest target");
            const auto inventory=c.inventory(); const auto loaded=c.load(t,h.resume,h.key);
            exact(loaded,sample(t,24)); check(c.inventory()==inventory,"load leaves inventory unchanged");
            c.use(loaded,h.key); check(!c.lookup(t).disk,"use promotes"); c.clear_ram();
        }
        auto short_t=tokens(); short_t.resize(16);
        auto h=c.lookup(short_t); check(h.resume==8 && h.shared==16,"unknown whole boundary excluded");
        short_t.push_back(999); h=c.lookup(short_t); check(h.resume==16,"unknown partial boundary allowed");
        exact(c.load(short_t,16,h.key),sample(tokens(),16,true));
        auto whole=tokens(); whole.resize(24); check(c.lookup(whole).resume==24,"known whole boundary allowed");
        auto peer=c.inventory(); peer.erase(std::remove_if(peer.begin(),peer.end(),[&](const ds::PrefixDiskItem& i){return i.key==ds::prefix_key(tokens(true),8,3);}),peer.end());
        check(c.retain(peer)==1,"retain counts the branch entry once"); c.flush();
        check(c.lookup(tokens(true)).resume==16,"retained shared unknown snapshot");
        c.forget(tokens(),16); c.flush(); c.clear_ram(); check(c.inventory().empty() && c.lookup(tokens()).resume==0,"forget removes dependent chain");
    }
}
std::vector<std::uint8_t> raw(const fs::path& p) {
    std::ifstream f(p,std::ios::binary);
    check(bool(f),"fixture read open");
    return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
void clean(const fs::path& p) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
    for(;;) {
        std::vector<fs::path> candidates;
        // Close enumeration before observing existence. The host can transiently
        // expose deleted/renamed aliases even with the archived entry IO code.
        for(const auto& e:fs::directory_iterator(p))
            if(e.path().extension()!=".blk" && e.path().extension()!=".snp") candidates.push_back(e.path());
        fs::path remaining;
        for(const auto& file:candidates) if(fs::exists(file)) {remaining=file;break;}
        if(remaining.empty()) return;
        if(std::chrono::steady_clock::now()>=deadline)
            throw std::runtime_error("no leaked temporary: "+remaining.string());
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
struct Worker {
    io::Faults inject;
    bool inject_block=false;
    std::thread::id caller=std::this_thread::get_id();
    std::uint64_t activity=0, at=0, writes=0, deletes=0;
    bool isolated=true, hold=false, started=false, released=false;
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<std::pair<fs::path,bool>> order;
    static void hook(void* context,bool before,const fs::path& path,bool write) noexcept {
        auto& w=*static_cast<Worker*>(context);
        if(before) {
            w.isolated=w.isolated && w.caller!=std::this_thread::get_id();
            io::faults()=write && path.extension()==(w.inject_block?".blk":".snp") ? w.inject : io::Faults{};
            w.at=io::activity();
            if(w.hold) {
                std::unique_lock lock(w.mutex); w.started=true; w.ready.notify_all();
                w.ready.wait(lock,[&]{return w.released;}); w.hold=false;
            }
        } else {
            w.activity+=io::activity()-w.at; write?++w.writes:++w.deletes;
            w.order.emplace_back(path,write); io::faults()={};
        }
    }
    Worker() { ds::prefix_io_test::set_writer_hook(hook,this); }
    ~Worker() { ds::prefix_io_test::set_writer_hook(nullptr,nullptr); }
};
void laziness(const fs::path& root) {
    const auto before=io::activity();
    ds::PrefixCacheOptions o; o.block=8; o.budget_bytes=1;
    ds::PrefixCache c(o); check(!c.insert(sample(tokens(),8),1),"refused tiny RAM capture");
    c.flush(); check(c.inventory().empty() && io::activity()==before,"no disk/refused capture uses no common IO");
    check(!fs::exists(root),"no disk created no directory");
    auto disk=options(root,false); disk.queue_budget_bytes=1;
    Worker w; ds::PrefixCache d(disk);
    const auto opened=io::activity();
    check(d.insert(sample(tokens(),8),1),"queue refusal still retains RAM"); d.flush();
    check(io::activity()==opened && w.writes==0 && w.activity==0,"refused persistence performs no common IO");
    const auto s=d.stats();
    check(s.disk_resource_refused==1 && s.queued_bytes==0 && s.queue_peak_bytes==0 && s.disk_bytes==0 && d.inventory().empty(),"queue refusal drains reservations and index");
    check(fs::is_empty(root),"queue refusal creates no files");
}
void failures(const fs::path& root) {
    const auto donor=root/"donor";
    { ds::PrefixCache c(options(donor,false)); check(c.insert(sample(tokens(),8),7),"donor insert"); c.flush(); }
    fs::path snap; std::uint64_t block_bytes=0;
    for(const auto& e:fs::directory_iterator(donor)) if(e.path().extension()==".snp") snap=e.path().filename(); else block_bytes=e.file_size();
    const auto old=raw(donor/snap);
    unsigned arm=0;
    for(bool drop:{false,true}) for(unsigned failure=0;failure<8;++failure) {
        const auto dir=root/("fault-"+std::to_string(++arm));
        Worker w;
        if(failure<7) w.inject.fail=std::array{io::Operation::allocation,io::Operation::create,io::Operation::write,
            io::Operation::sync,io::Operation::replace,io::Operation::directory_sync,io::Operation::none}[failure];
        if(failure==6) {w.inject.max_write=3;w.inject.fail_write_at=17;}
        if(failure==7) w.inject.max_write=13;
        {
            ds::PrefixCache c(options(dir,drop));
            // Existing complete bytes, installed AFTER open so insert really replaces
            // a target. These bytes were made by the actual donor PrefixCache writer.
            fs::copy(donor,dir,fs::copy_options::overwrite_existing|fs::copy_options::recursive);
            auto newer=sample(tokens(),8); newer.next_token+=1;
            check(c.insert(newer,9),"fault insertion retains existing index contract"); c.flush();
            const auto s=c.stats(); const bool success=failure==7;
            check(s.write_errors==(success?0U:1U) && s.written_files==(success?2U:1U),"worker error/success accounting");
            check(s.written_bytes==(success ? s.disk_bytes:block_bytes) && s.queued_bytes==0 && s.queue_peak_bytes==s.disk_bytes,"payload bytes/reservation accounting");
            check(w.isolated && w.activity>0 && w.writes==2,"actual shared worker IO activity/thread isolation");
            check(io::faults().fail==io::Operation::none && io::faults().fail_write_at==UINT64_MAX,"worker TLS does not escape to caller");
            const bool visible=success || failure==5;
            check((raw(dir/snap)!=old)==visible,"last-good bytes before replacement; uncertain visible sync failure"); clean(dir);
            {
                auto o=options(dir,drop); ds::PrefixCache restart(o);
                const auto h=restart.lookup(tokens()); check(h.resume==8 && h.logits_fnv==(visible?9U:7U),"last-good/new restart selection");
                exact(restart.load(tokens(),8,h.key),visible?newer:sample(tokens(),8));
            }
            // Existing recovery protocol explicitly forgets the failed path before retry.
            w.inject={}; c.forget(tokens(),8); c.flush();
            check(c.insert(newer,9),"bounded actual retry after forget"); c.flush(); c.clear_ram();
            const auto h=c.lookup(tokens()); exact(c.load(tokens(),8,h.key),newer);
            check(c.stats().queued_bytes==0 && c.stats().write_errors==(success?0U:1U),"retry reservations drained/no new errors"); clean(dir);
        }
    }
    // A caller-thread failure must NOT inject into the asynchronous writer.
    Worker w; ds::PrefixCache c(options(root/"caller-isolation",true));
    io::faults().fail=io::Operation::write;
    check(c.insert(sample(tokens(),8),3),"caller isolation insert"); c.flush(); io::faults()={};
    check(c.stats().write_errors==0 && w.isolated && w.activity>0,"caller TLS fault did not affect real worker");
}
void order_and_active(const fs::path& root) {
    Worker w; w.hold=true; ds::PrefixCache c(options(root,true));
    check(c.insert(sample(tokens(),24),1),"ordered insert");
    bool started;
    {std::unique_lock lock(w.mutex); started=w.ready.wait_for(lock,std::chrono::seconds(5),[&]{return w.started;});}
    const auto s=c.stats(); c.forget(tokens(),24);
    {std::lock_guard lock(w.mutex);w.released=true;} w.ready.notify_all();
    c.flush();
    check(started && s.queued_bytes==s.disk_bytes && s.queued_bytes>0,"active payload stays charged");
    check(c.stats().queued_bytes==0 && c.inventory().empty() && fs::is_empty(root),"delete after pending writes drains to empty");
    check(w.writes==4 && w.deletes==4 && w.activity>0,"actual three block/snapshot writes and deletes");
    for(std::size_t i=0;i<w.order.size();++i) if(!w.order[i].second) {
        bool earlier=false; for(std::size_t j=0;j<i;++j) earlier=earlier || (w.order[j].first==w.order[i].first && w.order[j].second);
        check(earlier,"each actual deletion follows its write");
    }
}
void block_failures(const fs::path& root) {
    const auto donor=root/"donor";
    {ds::PrefixCache c(options(donor,false));check(c.insert(sample(tokens(),8),7),"block donor insert");c.flush();}
    fs::path block;std::uint64_t snap_bytes=0;
    for(const auto& e:fs::directory_iterator(donor)) if(e.path().extension()==".blk") block=e.path().filename();else snap_bytes=e.file_size();
    const auto old=raw(donor/block);
    unsigned arm=0;
    for(bool drop:{false,true}) for(unsigned failure=0;failure<7;++failure) {
        Worker w;w.inject_block=true;const auto dir=root/std::to_string(++arm);
        bool persist=false;auto o=options(dir,drop);o.disk_admission=[&](bool){return persist;};
        ds::PrefixCache c(o);
        auto other=tokens();other[0]+=1;
        // Prime the clock using a real RAM capture and explicit peer refusal;
        // no unrelated files need deletion before installing the donor fixture.
        check(c.insert(sample(other,8),1),"prime logical clock through real insert");c.flush();c.clear_ram();persist=true;
        const auto baseline=c.stats();fs::copy(donor,dir,fs::copy_options::overwrite_existing|fs::copy_options::recursive);
        if(failure<6) w.inject.fail=std::array{io::Operation::allocation,io::Operation::create,io::Operation::write,
            io::Operation::sync,io::Operation::replace,io::Operation::directory_sync}[failure];
        else {w.inject.max_write=3;w.inject.fail_write_at=17;}
        auto newer=sample(tokens(),8);newer.next_token+=1;
        check(c.insert(newer,9),"block fault insert");c.flush();
        const auto s=c.stats();
        check(s.write_errors==1 && s.written_files==baseline.written_files+1 && s.written_bytes==baseline.written_bytes+snap_bytes,
            "block failure is not reported as committed bytes");
        check((raw(dir/block)!=old)==(failure==5),"block last-good/prepublication and uncertain visible replacement");
        check(s.queued_bytes==0 && w.isolated && w.activity>0,"block fault worker survives and drains reservation");clean(dir);
        // Blocks are immutable by token prefix: only their insertion sequence changed.
        // Either complete block is the same payload and decodes with the new snapshot.
        c.clear_ram();exact(c.load(tokens(),8,c.lookup(tokens()).key),newer);
        w.inject={};c.forget(tokens(),8);c.flush();check(c.insert(newer,9),"block failure retry");c.flush();
        check(c.stats().write_errors==1 && c.stats().queued_bytes==0,"block retry succeeds without leaked reservation");clean(dir);
    }
}
void safe_misses(const fs::path& root) {
    unsigned arm=0;
    for(bool drop:{false,true}) for(unsigned failure=0;failure<5;++failure) {
        const auto dir=root/std::to_string(++arm); ds::PrefixCache c(options(dir,drop));
        check(c.insert(sample(tokens(),24),1),"read fault insert"); c.flush(); c.clear_ram();
        const auto h=c.lookup(tokens()); const auto inv=c.inventory();
        fs::path block; for(const auto& e:fs::directory_iterator(dir)) if(e.path().extension()==".blk") {block=e.path();break;}
        if(failure==0) {std::fstream f(block,std::ios::in|std::ios::out|std::ios::binary); f.seekp(127); f.put('x');}
        if(failure==1) fs::resize_file(block,129);
        if(failure==2) fs::remove(block);
        if(failure>=3) io::faults().fail=failure==3?io::Operation::read:io::Operation::allocation;
        const auto activity=io::activity(); bool missed=false;
        try {(void)c.load(tokens(),24,h.key);} catch(const ds::PrefixCacheReadError&) {missed=true;}
        io::faults()={}; check(missed && io::activity()>activity,"actual shared read/allocation or corrupt/truncated/missing safe miss");
        check(c.inventory()==inv && c.stats().read_errors==1,"failed load never changes inventory");
        c.forget(tokens(),24); c.flush(); check(c.inventory().empty(),"failed read path forgotten");
        check(c.insert(sample(tokens(),24),1),"read fault retry"); c.flush(); c.clear_ram();
        exact(c.load(tokens(),24,c.lookup(tokens()).key),sample(tokens(),24));
    }
}
void policies(const fs::path& root) {
    const auto dir=root/"limits";
    ds::PrefixCacheOptions o=options(dir,false); o.disk_budget_bytes=1;
    ds::PrefixCache c(o); const auto activity=io::activity(); check(c.insert(sample(tokens(),8),1),"disk byte limit retains RAM"); c.flush();
    check(c.stats().disk_refused==1 && c.stats().disk_bytes==0 && io::activity()==activity && fs::is_empty(dir),"disk byte limit no IO or growth");
    auto peer=options(root/"peer",true);unsigned calls=0; peer.disk_admission=[&](bool local){check(local,"local admission");++calls;return false;};
    ds::PrefixCache denied(peer); check(denied.insert(sample(tokens(),8),1),"peer refusal retains RAM");denied.flush();
    check(calls==1 && denied.inventory().empty() && denied.stats().queued_bytes==0 && fs::is_empty(peer.dir),"both-rank refusal releases reservation");
#ifdef _WIN32
    auto unsupported=options(root/"unsupported",false);unsupported.quota_dir=root/"quota";
    bool refused=false;try{ds::PrefixCache unsupported_cache(unsupported);}catch(const std::runtime_error&){refused=true;}
    check(refused && !fs::exists(unsupported.quota_dir) && fs::is_empty(unsupported.dir),"Windows quota/owner still unsupported");
    std::cout<<"NOT RUN: Linux direct/fallback, page residency, 0644, quota/owner/cross-process locks\n";
#else
    for(bool fallback:{false,true}) {
        strix::state_file_io::fault_injection().refuse_direct=fallback;
        ds::PrefixCache direct(options(root/(fallback?"fallback":"automatic"),true));
        check(direct.insert(sample(tokens(),24),1),"direct policy insert"); direct.flush(); direct.clear_ram();
        exact(direct.load(tokens(),24,direct.lookup(tokens()).key),sample(tokens(),24));
    }
    strix::state_file_io::fault_injection().refuse_direct=false;
#endif
}
void tests(const fs::path& root) {
    check(!fs::exists(root),"test root must be new"); fs::create_directories(root);
    const auto cleanup=root/"cleanup-check";fs::create_directory(cleanup);
    {std::ofstream leak(cleanup/"retained.tmp",std::ios::binary);leak<<"must be detected";}
    bool detected=false;try{clean(cleanup);}catch(const std::runtime_error&){detected=true;}
    check(detected,"cleanup assertion detects a real remaining temporary");
    fs::remove(cleanup/"retained.tmp");clean(cleanup);
    laziness(root/"lazy"); failures(root/"failures"); block_failures(root/"blocks"); order_and_active(root/"ordering");
    safe_misses(root/"reads"); policies(root/"policies");
    capture(root/"codec"); verify(root/"codec");
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--fail-sentinel") check(false,"deliberate active-check sentinel");
        check(argc==3,"usage: write|read DIRECTORY");
        if(std::string(argv[1])=="write") capture(argv[2]);
        else if(std::string(argv[1])=="test") {
            // Only remove this test's explicitly supplied output, never entry fixtures.
            fs::remove_all(argv[2]); tests(argv[2]);
        } else { check(std::string(argv[1])=="read","unknown mode"); verify(argv[2]); }
        std::cout<<"DS41 prefix entry codec PASS\n"; return 0;
    } catch(const std::exception& e) { std::cerr<<"DS41 prefix IO FAIL: "<<e.what()<<'\n'; return 1; }
}
