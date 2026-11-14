#include "../apps/hip_session_store.hpp"
#include "strix/model_content_hash.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#ifndef _WIN32
#include <sys/wait.h>
#endif

namespace {
void check(bool value,const char* why){if(!value)throw std::runtime_error(why);}
template<class F>void refused(F f){bool threw=false;try{f();}catch(const std::exception&){threw=true;}check(threw,"expected refusal");}
strix::PrefixCacheBuffer heap(std::uint64_t n){return {std::shared_ptr<std::byte>(new std::byte[static_cast<std::size_t>(n?n:1)],std::default_delete<std::byte[]>()),n};}
strix::ModelContentIdentity hash(std::string_view s){return strix::sha256_bytes(std::as_bytes(std::span(s.data(),s.size())));}
}
int main(){
    const auto dir=std::filesystem::temp_directory_path()/("strix-http-session-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try{
        using namespace strix;using namespace strix::decode_cli;
        check(session_capture_peak_bound(10,2,20,400,400)==200,"four-copy capture bound");
        for(const auto available:{0ULL,199ULL}) {
            try { (void)session_capture_peak_bound(10,2,20,400,available); check(false,"fresh UMA refusal missing"); }
            catch(const SessionApiError& e) {check(e.status==507 && e.code=="session_budget","typed local UMA refusal");}
        }
        refused([&]{(void)session_capture_peak_bound(10,2,UINT64_MAX,UINT64_MAX,UINT64_MAX);});
        SessionCompatibility compat{hash("model"),hash("exe"),"fixed serial controller v1",true};
        SessionCheckpoint checkpoint{compat,{}, {}};
        checkpoint.frontier.consumed=3;checkpoint.frontier.accepted=3;checkpoint.frontier.predictor=2;
        checkpoint.frontier.tokens={1,2,3};checkpoint.frontier.pending=4;checkpoint.frontier.emitted=1;
        auto payload=heap(8192);std::memset(payload.data.get(),7,8192);
        checkpoint.sections.push_back({"mtp.kv",payload});checkpoint.sections.push_back({"mtp.held",heap(32)});
        auto blob=encode_session_checkpoint(checkpoint,heap);
        const std::vector<std::uint32_t> prompt{1,2,3,4,5};
        check(session_prompt_extends(checkpoint.frontier,prompt),"extension rejected");
        check(!session_prompt_extends(checkpoint.frontier,std::vector<std::uint32_t>{1,2,3,9,5}),"pending mismatch accepted");
        check(!session_prompt_extends(checkpoint.frontier,std::vector<std::uint32_t>{1,9,3,4,5}),"token mismatch accepted");
        check(!session_prompt_extends(checkpoint.frontier,std::vector<std::uint32_t>{1,2,3}),"missing pending accepted");
        {
            const auto tier=dir/"prefix-startup";
            std::filesystem::create_directories(tier);
            { std::ofstream seed(tier/"occupancy"); seed << "payload"; }
            PrefixStateCache prefix({1234,tier,5678},heap);
            StateCacheStatsMirror startup; startup.publish(prefix_snapshot(prefix));
            const auto stats=*serve::parse_json(startup.read()).find("engine")->find("state_cache");
            check(stats.find("host")->find("budget_bytes")->as_int()==1234,"prefix startup host budget");
            check(stats.find("disk")->find("budget_bytes")->as_int()==5678,"prefix startup disk budget");
            check(stats.find("disk")->find("payload_bytes")->as_int()==7,"prefix startup disk occupancy");
        }
        {
            PrefixStateCache prefix({1234,{},0},heap);
            prefix.record_snapshot_preparation(); prefix.record_snapshot_preparation();
            prefix.record_staging(512,128,true);
            set_prefix_batch_certification_enabled(true);
            StateCacheStatsMirror stats; stats.publish(prefix_snapshot(prefix));
            auto p=*serve::parse_json(stats.read()).find("engine")->find("state_cache")->find("prefix");
            check(p.find("snapshot_preparations")->as_int()==2 && p.find("staging_bytes")->as_int()==128,"HTTP stats expose preparations and live staging");
            prefix.record_staging(512,128,false); stats.publish(prefix_snapshot(prefix));
            p=*serve::parse_json(stats.read()).find("engine")->find("state_cache")->find("prefix");
            check(p.find("snapshot_preparations")->as_int()==2 && p.find("staging_bytes")->as_int()==0,"settled HTTP stats retain monotonic count and zero staging");
            set_prefix_batch_certification_enabled(false);
            check(!state_cache_json(prefix_snapshot(prefix)).find("prefix")->find("snapshot_preparations"),"default-off stats omit preparation counter");
        }
        HttpSessionStore store(dir,1U<<20,2,heap,compat);
        StateCacheStatsMirror mirror;
        auto initial=serve::parse_json(store.command("stats","",""));
        mirror.publish(store.snapshot(),initial);
        auto published=serve::parse_json(mirror.read());
        for(const auto* key:{"tombstone_bytes","tombstones","disk_fallback_disabled","max_parked","evicted"})
            check(published.find("engine")->find(key)->dump()==initial.find("engine")->find(key)->dump(),"startup legacy engine field lost");
        for(const auto* key:{"ram_bytes","ram_budget","parked","disk_bytes","hits","misses","refusals"})
            check(published.find(key)->dump()==initial.find(key)->dump(),"startup legacy top field lost");
        auto sample=*published.find("engine")->find("state_cache");
        check(sample.find("disk")->find("budget_bytes")->dump()==serve::Json(store.snapshot().disk.budget_bytes).dump(),"startup disk budget lost");
        check(sample.find("published")->as_bool() && sample.find("sequence")->as_int()==1,"initial publication metadata");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        auto aged=*serve::parse_json(mirror.read()).find("engine")->find("state_cache");
        check(aged.find("sequence")->as_int()==1 && aged.find("age_ms")->as_double()>sample.find("age_ms")->as_double(),"read-time freshness without publication");
        mirror.publish(store.snapshot(),initial);
        check(serve::parse_json(mirror.read()).find("engine")->find("state_cache")->find("sequence")->as_int()==2,"publication sequence advances");
        store.park("A",blob);store.park("B",blob);
        check(store.matching("A",prompt).has_value(),"A-B-A parking failed");
        check(!store.matching("A",std::vector<std::uint32_t>{1,2,3,9,5}),"mismatch was not a miss");
        store.park("C",blob);
        check(!store.matching("B",prompt),"count LRU did not evict B");
        (void)store.command("checkpoint","A","disk");
        {
            HttpSessionStore restarted(dir,1U<<20,2,heap,compat);
            auto restored=restarted.matching("A",prompt,false);
            check(restarted.snapshot().named.parked==0,"no-store disk hit promoted named state");
            auto before=read_session_checkpoint(dir/"A",compat,heap,1U<<20);
            check(restored && restored->sections[0].payload.data.get()!=payload.data.get(),"named restore must use a copy");
            restored->sections[0].payload.data.get()[0]=std::byte{99};
            auto after=read_session_checkpoint(dir/"A",compat,heap,1U<<20);
            check(before.bytes==after.bytes && !std::memcmp(before.data.get(),after.data.get(),before.bytes),"no-store continuation changed published envelope");
            restored=restarted.matching("A",prompt);

            check(restored && restored->frontier.predictor==2 && restored->frontier.pending==4U,"restart frontier changed");
            check(restored->sections[0].payload.bytes==8192 && !std::memcmp(restored->sections[0].payload.data.get(),payload.data.get(),8192),"restart bytes changed");
            check(serve::parse_json(restarted.command("list","","")).find("sessions")->size()==1,"disk list missing session");
            check(!restarted.command("stats","","").empty(),"missing stats");
        }
        {
            const auto tier=dir/"bounded";
            HttpSessionStore bounded(tier,1U<<20,2,heap,compat,blob.bytes);
            bounded.park("first",blob); bounded.park("second",blob);
            check(!std::filesystem::exists(tier/"first"/"session.bin") &&
                std::filesystem::exists(tier/"second"/"session.bin"),"disk byte cap did not evict oldest checkpoint");
            HttpSessionStore restart(tier,1U<<20,2,heap,compat,blob.bytes);
            check(restart.matching("second",prompt).has_value(),"default disk tier restart miss");
        }
        // Atomic replacement must budget the old destination plus the temporary file.
        {
            HttpSessionStore peak(dir/"peak",1U<<20,2,heap,compat,blob.bytes);
            peak.park("A",blob);
            try { (void)peak.command("checkpoint","A","disk"); check(false,"replacement escaped peak quota"); }
            catch (const SessionApiError& e) { check(e.status==507 && e.code=="session_budget","wrong budget error"); }
            auto stats=serve::parse_json(peak.command("stats","",""));
            check(stats.find("disk_bytes")->as_int()==static_cast<std::int64_t>(blob.bytes) &&
                stats.find("refusals")->as_int()==1,"disk stats or refusals missing");
            HttpSessionStore replacement(dir/"replacement",1U<<20,2,heap,compat,blob.bytes*2);
            replacement.park("A",blob);
            (void)replacement.command("checkpoint","A","disk");
            check(serve::parse_json(replacement.command("stats","","")).find("disk_bytes")->as_int()==static_cast<std::int64_t>(blob.bytes),
                "replacement peak admitted but final usage incorrect");
            auto rows=serve::parse_json(peak.command("list","",""));
            const auto& row=rows.find("sessions")->as_array().at(0);
            for (const auto* key : {"id","position","consumed","pending","bytes","state","durability"})
                check(row.find(key)!=nullptr,"common list field missing");
        }
#ifndef _WIN32
        // Independent stores/processes start publication together into a one-envelope quota.
        {
            const auto shared=dir/"processes";
            int barrier[2]; check(::pipe(barrier)==0,"pipe failed");
            std::vector<pid_t> children;
            for (unsigned i=0;i<4;++i) {
                const auto pid=::fork(); check(pid>=0,"fork failed");
                if (!pid) {
                    ::close(barrier[1]); char signal=0;
                    if (::read(barrier[0],&signal,1)!=1) ::_exit(2);
                    try {
                        HttpSessionStore writer(shared,1U<<20,2,heap,compat,blob.bytes);
                        writer.park("worker_"+std::to_string(i),blob);
                        ::_exit(0);
                    } catch (...) { ::_exit(3); }
                }
                children.push_back(pid);
            }
            ::close(barrier[0]); check(::write(barrier[1],"xxxx",4)==4,"barrier failed"); ::close(barrier[1]);
            for (const auto pid : children) { int status=0; check(::waitpid(pid,&status,0)==pid && WIFEXITED(status) && WEXITSTATUS(status)==0,"writer failed"); }
            HttpSessionStore observer(shared,1U<<20,2,heap,compat,blob.bytes);
            check(serve::parse_json(observer.command("stats","","")).find("disk_bytes")->as_int()==static_cast<std::int64_t>(blob.bytes),"interprocess quota exceeded");
        }
#endif
        {
            HttpSessionStore api(dir/"errors",1U<<20,2,heap,compat);
            try { (void)api.command("resume","unknown",""); check(false,"missing session accepted"); }
            catch (const SessionApiError& e) {
                auto json=serve::session_error(e.code,e.what(),e.status);
                check(e.status==404 && json.find("error")->find("code")->as_string()=="session_not_found","404 shape");
            }
            publish_session_checkpoint(dir/"errors"/"bad",blob);
            { std::ofstream corrupt(dir/"errors"/"bad"/"session.bin",std::ios::binary|std::ios::trunc); corrupt<<"corrupt"; }
            try { (void)api.command("resume","bad",""); check(false,"corrupt session accepted"); }
            catch (const SessionApiError& e) { check(e.status==422 && e.code=="session_corrupt","corrupt status"); }
        }
        auto wrong=compat;wrong.numerical_config+="wrong";
        HttpSessionStore incompatible(dir,1U<<20,2,heap,wrong);
        check(!incompatible.matching("A",prompt),"incompatible opportunistic restore was not miss");
        try { (void)incompatible.command("resume","A",""); check(false,"incompatible accepted"); }
        catch (const SessionApiError& e) { check(e.status==409 && e.code=="session_incompatible","incompatible status"); }
        refused([&]{(void)store.command("checkpoint","A","invalid");});
        refused([&]{(void)store.command("delete","../A","");});
        refused([&]{store.admit((1U<<20)+1U);});
        // Outstanding resume handles remain charged even after deletion, including temporary reservation.
        PrefixStateCache cache({blob.bytes*2U+4096U,{},0},heap);
        check(cache.park_session("held",heap(blob.bytes),2),"park failed");
        auto held=cache.resume_session("held");check(cache.delete_session("held"),"delete failed");
        check(!cache.admit_session_working_bytes(cache.policy().host_byte_budget),"retained handle escaped budget");
        held={};
        // checkpoint still owns section bytes, but the encoded envelope is no longer retained externally.
        check(cache.admit_session_working_bytes(cache.policy().host_byte_budget),"released envelope still charged");
        store.park("A",encode_session_checkpoint(checkpoint,heap));
        check(std::filesystem::exists(dir/"A"/"session.bin"),"explicit durable checkpoint was removed by RAM turn");
        store.park("D",encode_session_checkpoint(checkpoint,heap));
        store.park("E",encode_session_checkpoint(checkpoint,heap));
        check(!store.matching("A",prompt),"evicted newer RAM frontier fell back to stale disk");
        store.park("A",encode_session_checkpoint(checkpoint,heap));store.invalidate_turn("A");
        check(!store.matching("A",prompt),"cancelled turn restored stale disk");
        (void)store.command("delete","A","");check(!store.matching("A",prompt),"deleted session resumed");
        // Unique RAM-only names allocate no tombstones, even with disk enabled.
        HttpSessionStore churn(dir,1U<<20,2,heap,compat);
        for (unsigned i=0;i<1000;++i) churn.park("ram_"+std::to_string(i),blob);
        auto churn_stats=serve::parse_json(churn.command("stats","",""));
        check(churn_stats.find("engine")->find("tombstones")->as_int()==0,"RAM-only names retained tombstones");
        check(churn_stats.find("engine")->find("tombstone_bytes")->as_int()==258,"metadata reservation changed");
        check(churn_stats.find("ram_bytes")->as_int()<=1U<<20,"metadata escaped RAM budget");
        // More dirty durable names than the reserved slots fail closed globally.
        for (const auto* name : {"disk_1","disk_2","disk_3"}) {
            churn.park(name,blob);(void)churn.command("checkpoint",name,"disk");
            churn.invalidate_turn(name);
        }
        auto saturated=serve::parse_json(churn.command("stats","",""));
        check(saturated.find("engine")->find("disk_fallback_disabled")->as_bool(),"tombstone overflow did not fail closed");
        check(saturated.find("engine")->find("tombstones")->as_int()<=2,"tombstone limit exceeded");
        for (const auto* name : {"disk_1","disk_2","disk_3"})
            check(!churn.matching(name,prompt),"overflow resurrected stale disk frontier");
        churn.park("current",blob);check(churn.matching("current",prompt).has_value(),"overflow disabled RAM lookup");
        HttpSessionStore disk_restart(dir,1U<<20,2,heap,compat);
        check(disk_restart.matching("disk_1",prompt).has_value(),"restart did not restore explicit durable checkpoint");
        HttpSessionStore ram_only({},1U<<20,2,heap,compat);
        ram_only.park("ram",blob);ram_only.invalidate_turn("ram");
        check(serve::parse_json(ram_only.command("stats","","")).find("engine")->find("tombstone_bytes")->as_int()==0,"RAM-only store reserved tombstones");
        std::filesystem::remove_all(dir);std::cout<<"http_session_store_tests PASS\n";return 0;
    }catch(const std::exception& e){std::cerr<<"http_session_store_tests FAIL: "<<e.what()<<'\n';std::error_code ec;std::filesystem::remove_all(dir,ec);return 1;}
}
