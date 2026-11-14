#include "../apps/cuda_ds41_session_draft.hpp"
#include "../apps/cuda_ds41_replica_serve.hpp"
#include <iostream>
#include <sstream>
#include <functional>
#include <chrono>
#include <source_location>
using namespace strix::ds41::session;
#ifdef NDEBUG
#error CUDA DS41 host tests require active assertions
#endif
static int checks=0;
void check(bool ok,std::source_location at=std::source_location::current()) {++checks;if(!ok)throw std::runtime_error("test check failed at line "+std::to_string(at.line()));}
void refuses(const std::function<void()>& f) {bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}check(rejected);}
// Optional independent codec oracle. Fixed lineage and turn counters are deliberate.
static Snapshot oracle_snapshot(bool draft,bool advanced=false) {
    Snapshot s;s.compat=sha(bytes("r12-fixed-compat"));s.lineage="0123456789abcdef0123456789abcdef";
    s.committed_generation=advanced?8:7;
    strix::deepseek41::AttentionState a;a.position=3;a.next_token=advanced?29:23;
    a.tokens={1,7,11};a.layers.resize(40);
    for(auto& l:a.layers)l.ring.resize(136*528,17);
    s.attention=strix::deepseek41::serialize_attention_state(a);
    s.frontier.consumed=s.frontier.accepted=3;s.frontier.pending=a.next_token;
    s.frontier.emitted=advanced?19:17;s.frontier.turns=advanced?5:4;
    if(draft) {
        DraftState d;d.start=0;d.next=3;d.core=128;d.extra={128,900};
        for(unsigned i=0;i<3;++i)d.rings[i].resize(128*528,static_cast<std::uint8_t>(i+31));
        s.drafter=d.encode();s.frontier.dspark=true;s.frontier.fixed=3;s.frontier.draft=3;
    }
    return s;
}
static void oracle_equal(const Snapshot& a,const Snapshot& b) {
    check(a.attention==b.attention&&a.drafter==b.drafter&&a.frontier.encode()==b.frontier.encode());
    check(a.compat==b.compat&&a.lineage==b.lineage&&a.committed_generation==b.committed_generation);
}
static int oracle(const std::string& mode,const fs::path& root) {
    const bool emit=mode=="--capture";
    if(emit) {check(!fs::exists(root));fs::create_directories(root);}
    for(bool draft:{false,true}) {
        const auto folder=root/(draft?"draft":"plain");auto old=oracle_snapshot(draft),next=oracle_snapshot(draft,true);
        Config c;c.dir=folder/"managed";c.disk=true;c.disk_explicit=true;c.disk_bytes=32*1024*1024;
        if(emit) {
            publish(folder/"immutable",old);
            Disk disk;disk.open(c,1024);check(disk.enabled());disk.put("A",old);
            oracle_equal(restore(disk.path("A"),old.compat,16*1024*1024),old);
            disk.put("A",next);refuses([&]{disk.put("A",old);});
        }
        oracle_equal(restore(folder/"immutable",old.compat,16*1024*1024),old);
        Disk restart;restart.open(c,1024);check(restart.enabled());
        oracle_equal(restore(restart.path("A"),old.compat,16*1024*1024),next);
        check(read_pointer(restart.path("A")).directory=="A.g2");
    }
    std::cout<<"independent oracle "<<mode<<" PASS\n";return 0;
}
#ifdef _WIN32
struct Child {
    PROCESS_INFORMATION process{};
    Child(const std::wstring& mode,const fs::path& path) {
        wchar_t exe[32768];check(GetModuleFileNameW(nullptr,exe,32768)>0);
        auto command=L"\""+std::wstring(exe)+L"\" "+mode+L" \""+fs::absolute(path).wstring()+L"\"";
        STARTUPINFOW startup{};startup.cb=sizeof(startup);
        check(CreateProcessW(nullptr,command.data(),nullptr,nullptr,FALSE,CREATE_NO_WINDOW,nullptr,nullptr,&startup,&process)!=0);
    }
    ~Child(){if(process.hProcess){if(WaitForSingleObject(process.hProcess,0)==WAIT_TIMEOUT){TerminateProcess(process.hProcess,99);WaitForSingleObject(process.hProcess,10000);}
        CloseHandle(process.hThread);CloseHandle(process.hProcess);}}
    DWORD finish(){check(WaitForSingleObject(process.hProcess,10000)==WAIT_OBJECT_0);DWORD code=99;check(GetExitCodeProcess(process.hProcess,&code)!=0);return code;}
};
static std::array<DWORD,3> inode(const fs::path& path) {
    const auto h=CreateFileW(path.c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    check(h!=INVALID_HANDLE_VALUE);BY_HANDLE_FILE_INFORMATION i{};check(GetFileInformationByHandle(h,&i)!=0);CloseHandle(h);
    return {i.dwVolumeSerialNumber,i.nFileIndexHigh,i.nFileIndexLow};
}
static void process_publication_boundaries(const fs::path& root) {
    for(unsigned stop=0;stop<5;++stop) {
        const auto tier=root/std::to_string(stop);
        {Child crash(L"--crash-publish-"+std::to_wstring(stop),tier);check(crash.finish()==0);}
        {Child restart(L"--verify-crash-"+std::to_wstring(stop),tier);check(restart.finish()==0);}
    }
}
static void legacy_process_locks(const fs::path& root) {
    fs::create_directories(root);
    for(const auto* name:{L".owner.lock",L".session.lock"}) {
        const auto file=root/name;
        const auto h=CreateFileW(file.c_str(),GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
        check(h!=INVALID_HANDLE_VALUE);const auto identity=inode(file);
        for(DWORD offset:{0U,1U}) {
            OVERLAPPED ov{};ov.Offset=offset;check(LockFileEx(h,LOCKFILE_EXCLUSIVE_LOCK|LOCKFILE_FAIL_IMMEDIATELY,0,1,0,&ov)!=0);
            {
                Child child(name==std::wstring(L".owner.lock")?L"--owner-only-child":L"--transaction-child",root);
                if(offset==0&&name==std::wstring(L".session.lock")) {
                    check(WaitForSingleObject(child.process.hProcess,100)==WAIT_TIMEOUT);
                    check(!fs::exists(root/"transaction-acquired"));
                    check(UnlockFileEx(h,0,1,0,&ov)!=0);check(child.finish()==0);
                } else {
                    check(child.finish()==(offset==0?9U:0U));check(UnlockFileEx(h,0,1,0,&ov)!=0);
                }
            }
            check(inode(file)==identity);fs::remove(root/"transaction-acquired");
        }
        CloseHandle(h);
    }
    // Child owns a real Disk until process exit; the next process reopens it.
    {Child first(L"--exit-owner-child",root);check(first.finish()==0);}
    {Child transaction(L"--exit-transaction-child",root);check(transaction.finish()==0);}
    {Child released(L"--transaction-child",root);check(released.finish()==0);}
    {Child restart(L"--lock-child",root);check(restart.finish()==0);}
    check(fs::exists(root/".owner.lock")&&fs::exists(root/".session.lock"));
}
#endif
int main(int argc,char** argv) {
 try {
    if(argc==3&&(std::string(argv[1])=="--capture"||std::string(argv[1])=="--verify"))return oracle(argv[1],argv[2]);
#ifdef _WIN32
    if(argc==3&&(std::string(argv[1]).starts_with("--crash-publish-")||std::string(argv[1]).starts_with("--verify-crash-"))) {
        const std::string mode=argv[1];const auto stop=unsigned_value(mode.substr(mode.rfind('-')+1));check(stop<5);
        Config c;c.dir=argv[2];c.disk=true;c.disk_explicit=true;c.disk_bytes=32*1024*1024;
        Disk d;d.open(c,1024);check(d.enabled());
        const auto old=oracle_snapshot(true),next=oracle_snapshot(true,true);
        if(mode.starts_with("--crash")) {
            d.put("A",old);d.put("A",next,[&](auto p){if(static_cast<unsigned>(p)==stop)ExitProcess(0);});
            return 1;
        }
        oracle_equal(restore(d.path("A"),old.compat,16*1024*1024),stop>=3?next:old);
        const auto selected=read_pointer(d.path("A"));unsigned generations=0;
        for(const auto& e:fs::directory_iterator(c.dir)) {
            check(e.path().filename().string().find(".tmp")==std::string::npos);
            if(e.is_directory()){++generations;check(e.path().filename()==selected.directory);}
        }
        check(generations==1);return 0;
    }
    if(argc==3&&(std::string(argv[1])=="--transaction-child"||std::string(argv[1])=="--exit-transaction-child")) {
        DirectoryLock lock(argv[2]);write_file(fs::path(argv[2])/"transaction-acquired",bytes("ok"));
        if(std::string(argv[1])=="--exit-transaction-child")ExitProcess(0);return 0;
    }
    if(argc==3&&std::string(argv[1])=="--owner-only-child") {
        try{DirectoryLock lock(argv[2],true);return 0;}
        catch(const Error& e){check(e.code=="session_tier_busy"&&e.status==409);return 9;}
    }
    if(argc==3&&(std::string(argv[1])=="--lock-child"||std::string(argv[1])=="--exit-owner-child")) {
        Config child_config;child_config.enabled=true;child_config.disk=true;child_config.dir=argv[2];
        child_config.disk_explicit=true;child_config.disk_bytes=128*1024;
        Disk child_disk;
        try{child_disk.open(child_config,1024);}catch(const Error& e){
            require(e.code=="session_tier_busy"&&e.status==409,"child tier busy");return 9;
        }
        require(child_disk.enabled(),"child disk open");
        Snapshot snapshot;snapshot.compat=sha(bytes("child"));snapshot.attention=bytes("child-state");
        child_disk.erase("child");child_disk.put("child",snapshot);
        require(child_disk.used()<=child_disk.budget(),"child quota");
        child_disk.touch("child");require(child_disk.erase("child"),"child eviction");
        child_disk.put("child",snapshot);
        write_file(fs::path(argv[2])/"child-acquired",bytes("ok"));
        if(std::string(argv[1])=="--exit-owner-child")ExitProcess(0);return 0;
    }
#endif
    const auto dir=fs::path(argc>1?argv[1]:"session-cache-test-output") /
        ("run-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    require(!fs::exists(dir),"test output must be a fresh directory");fs::create_directories(dir);
    {
        const Bytes large(8U*1024U*1024U+3U,19);write_file(dir/"read-progress",large);
        std::vector<std::uint64_t> chunks;
        check(read_file(dir/"read-progress",large.size(),[&](const auto& stage,auto done,auto total){
            check(stage=="restore-read:read-progress"&&total==large.size());chunks.push_back(done);
        })==large);
        check(chunks==std::vector<std::uint64_t>({8U*1024U*1024U,large.size()}));
        try{(void)read_file(dir/"read-progress",1);check(false);}
        catch(const Error& e){check(e.code=="session_budget"&&e.status==507);}
    }
    check(sha(bytes("abc"))=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    // CPU injection of CUDA-resolved identity: selector spellings have no role.
    const auto compat=[&](const std::wstring& selector,const std::wstring& cache,
                          const std::string& output,const std::string& numeric) {
        Bytes record;std::array<std::uint8_t,16> uuid{1,2,3,4};
        compat_hardware(record,uuid,12,0,13000,13000);
        for(const auto& v:{selector,cache,std::wstring(L"STRIX_NUMERIC=1")})
            if(compat_environment(v))blob(record,std::span(reinterpret_cast<const std::uint8_t*>(v.data()),v.size()*sizeof(wchar_t)));
        compat_arguments(record,{"--controller-save",output,"--json",output,"--draft-device","0","--numeric",numeric},
                         [&](const auto& path){return sha(read_file(path,1024));});
        return sha(record);
    };
    const auto identity=compat(L"CUDA_VISIBLE_DEVICES=0",L"STRIX_HASH_CACHE=old","old-output","1");
    check(identity==compat(L"CUDA_VISIBLE_DEVICES=GPU-same-card",L"STRIX_HASH_CACHE=new","new-output","1"));
    check(identity!=compat(L"CUDA_VISIBLE_DEVICES=0",L"STRIX_HASH_CACHE=old","old-output","2"));
    check(!compat_environment(L"cuda_visible_devices=0")&&!compat_environment(L"strix_hash_cache=new"));
    check(compat_environment(L"STRIX_NUMERIC=2")&&compat_environment(L"CUDA_DEVICE_MAX_CONNECTIONS=1"));
#ifdef _WIN32
    // Rebuild the launch environment after an ordinary request's allowed --env.
    // The first session must hash exactly what a fresh worker would hash.
    namespace rs=strix::ds41::replica_serve;
    std::map<std::string,std::string> start_env;
    for(const char* key:{"STRIX_DS41_CUDA_ENGRAM_CHECK","STRIX_DS41_REPLICA_WINDOW_ENGRAM_PREFETCH"}) {
        check(rs::is_request_env(key));const auto value=std::getenv(key);start_env[key]=value?value:"";
    }
    const std::vector<std::string> launch_env{"STRIX_DS41_CUDA_ENGRAM_CHECK=1"};
    const auto apply_env=[&](const std::vector<std::string>& overrides) {
        for(const auto& [key,value]:start_env)check(_putenv_s(key.c_str(),value.c_str())==0);
        for(const auto& e:overrides) {
            const auto at=e.find('=');check(rs::is_request_env(e.substr(0,at)));
            check(_putenv_s(e.substr(0,at).c_str(),e.substr(at+1).c_str())==0);
        }
    };
    apply_env(launch_env);const auto launch_key_env=compat_environment_record();
    rs::check_request_args({"--env","STRIX_DS41_CUDA_ENGRAM_CHECK=0","--env","STRIX_DS41_REPLICA_WINDOW_ENGRAM_PREFETCH=0"});
    auto request_env=launch_env;request_env.push_back("STRIX_DS41_CUDA_ENGRAM_CHECK=0");
    request_env.push_back("STRIX_DS41_REPLICA_WINDOW_ENGRAM_PREFETCH=0");apply_env(request_env);
    check(compat_environment_record()!=launch_key_env);
    apply_env(launch_env);check(compat_environment_record()==launch_key_env);
    for(const auto& [key,value]:start_env)check(_putenv_s(key.c_str(),value.c_str())==0);
#endif
    Bytes hw1,hw2;std::array<std::uint8_t,16> uuid1{1},uuid2{2};
    compat_hardware(hw1,uuid1,12,0,13000,13000);compat_hardware(hw2,uuid2,12,0,13000,13000);
    check(hw1!=hw2);hw2.clear();compat_hardware(hw2,uuid1,12,0,13001,13000);check(hw1!=hw2);
    Snapshot s;s.compat=sha(bytes("config-a"));s.attention={0,1,2,0,255};
    s.frontier.consumed=s.frontier.accepted=8192;s.frontier.pending=23;s.frontier.emitted=17;s.frontier.turns=1;
    strix::deepseek41::AttentionState attention;attention.position=2;attention.next_token=23;attention.tokens={1,2};attention.layers.resize(40);
    for(auto& layer:attention.layers)layer.ring.resize(136*528);
    // An ordinary alternate import precedes explicit create and history fallback.
    // Compare complete serialized launch state with an untouched fresh worker.
    {
        using State=strix::deepseek41::AttentionState;
        const auto launch_bytes=strix::deepseek41::serialize_attention_state(attention);
        const auto launch_path=dir/"launch-import.bin";
        strix::deepseek41::save_attention_state(launch_path,attention);
        const auto alternate_path=dir/"alternate-import.bin";
        auto alternate=attention;alternate.position=3;alternate.next_token=42;alternate.tokens.push_back(9);
        alternate.layers[0].ring[0]=255;
        strix::deepseek41::save_attention_state(alternate_path,alternate);
        for(bool admit_before_replace:{false,true}) {
            auto imported=attention;
            const auto loader=[](const fs::path& p){return strix::deepseek41::load_attention_state(p);};
            LaunchSeed<State> seed(true,launch_path,loader),fresh(true,launch_path,loader);
            for(unsigned request=0;request<3;++request) {
                seed.before_replace(imported,true,launch_path);
                imported=strix::deepseek41::load_attention_state(launch_path);seed.after_replace(launch_path);
                check(!seed.materialized());
            }
            if(admit_before_replace)(void)seed.materialize(imported);
            seed.before_replace(imported,true,alternate_path);
            imported=strix::deepseek41::load_attention_state(alternate_path);seed.after_replace(alternate_path);
            const auto& created=seed.materialize(imported);
            check(strix::deepseek41::serialize_attention_state(created)==launch_bytes);
            check(created.digest()==fresh.materialize(attention).digest());
            // Nested ring/token storage is independent even under in-place edits.
            imported.layers[0].ring[0]=17;imported.tokens[0]=99;
            auto saved=created;saved.tokens.push_back(7);++saved.position;
            check(!history_prefix(saved.tokens,attention.tokens));
            const auto& fallback=seed.materialize(imported);
            check(strix::deepseek41::serialize_attention_state(fallback)==
                  strix::deepseek41::serialize_attention_state(fresh.materialize(attention)));
            seed.before_replace(imported,true,alternate_path);imported=alternate;seed.after_replace(alternate_path);
            check(strix::deepseek41::serialize_attention_state(seed.materialize(imported))==launch_bytes);
        }
    }
    auto checked=s;checked.frontier.consumed=checked.frontier.accepted=2;
    checked.attention=strix::deepseek41::serialize_attention_state(attention);
    check(validate_attention(checked,attention,100).digest()==attention.digest());
    auto bad=checked;bad.frontier.pending=42;refuses([&]{validate_attention(bad,attention,100);});
    refuses([&]{validate_attention(checked,attention,2);});
    auto wrong=attention;wrong.layers[0].ring.resize(528);bad.attention=strix::deepseek41::serialize_attention_state(wrong);bad.frontier=checked.frontier;
    refuses([&]{validate_attention(bad,attention,100);});
    wrong=attention;wrong.layers[0].carry_valid=true;bad.attention=strix::deepseek41::serialize_attention_state(wrong);
    refuses([&]{validate_attention(bad,attention,100);});
    const auto n=s.size();Lru l(n*2);l.put("A",s);l.put("B",s);(void)l.get("A");l.put("C",s);
    check(l.contains("A")&&!l.contains("B")&&l.contains("C"));check(l.used()==n*2&&l.evicted()==1);
    auto big=s;big.attention.resize(n*3);refuses([&]{l.put("A",big);});check(l.contains("A")&&l.used()==n*2);
    refuses([&]{l.get("B");});check(l.erase("C"));check(!l.erase("C"));check(l.used()==n);
    auto original=s.attention;l.put("B",s);s.attention[0]=99;check(l.get("B").attention==original);s.attention=original;
    auto compat_snapshot=s;compat_snapshot.compat=identity;publish(dir/"compat",compat_snapshot);
    check(restore(dir/"compat",compat(L"CUDA_VISIBLE_DEVICES=GPU-same-card",L"STRIX_HASH_CACHE=moved","different-output","1"),1024*1024).attention==s.attention);
    refuses([&]{restore(dir/"compat",compat(L"CUDA_VISIBLE_DEVICES=0",L"STRIX_HASH_CACHE=old","old-output","2"),1024*1024);});
    publish(dir/"valid",s);const auto restored=restore(dir/"valid",s.compat,1024*1024);
    check(restored.attention==s.attention&&restored.frontier.encode()==s.frontier.encode());
    refuses([&]{publish(dir/"valid",s);});
    refuses([&]{restore(dir/"valid",sha(bytes("flag-changed")),1024*1024);});
    refuses([&]{restore(dir/"valid",sha(bytes("exe-changed")),1024*1024);});
    refuses([&]{restore(dir/"valid",s.compat,1);});
    auto copy=[&](const char* name){fs::copy(dir/"valid",dir/name,fs::copy_options::recursive);};
    copy("corrupt");write_file(dir/"corrupt"/"attention.bin",bytes("xxxxx"));
    refuses([&]{restore(dir/"corrupt",s.compat,1024*1024);});
    copy("truncated");fs::resize_file(dir/"truncated"/"attention.bin",1);
    refuses([&]{restore(dir/"truncated",s.compat,1024*1024);});
    copy("manifest");auto m=read_file(dir/"manifest"/"manifest.bin",65536);m.back()^=1;write_file(dir/"manifest"/"manifest.bin",m);
    refuses([&]{restore(dir/"manifest",s.compat,1024*1024);});
    fs::create_directory(dir/"interrupted.tmp");refuses([&]{restore(dir/"interrupted",s.compat,1024*1024);});
    DraftState d;d.start=17;d.next=8209;d.core=128;d.extra={128,900};
    for(unsigned i=0;i<3;++i){d.rings[i].resize(128*528);for(std::size_t j=0;j<d.rings[i].size();++j)d.rings[i][j]=static_cast<std::uint8_t>(j+i);}
    auto b=d.encode();check(DraftState::decode(b).encode()==b);
    auto drafted=s;drafted.drafter=b;drafted.frontier.dspark=true;drafted.frontier.fixed=3;
    drafted.frontier.draft=drafted.frontier.consumed;publish(dir/"drafted",drafted);
    check(restore(dir/"drafted",drafted.compat,1024*1024).drafter==b);
    auto short_b=b;short_b.pop_back();refuses([&]{DraftState::decode(short_b);});
    d.extra={128,128};refuses([&]{d.encode();});d.extra={129280};refuses([&]{d.encode();});
    auto f=s.frontier;f.dspark=true;f.fixed=3;f.draft=f.consumed;check(Frontier::decode(f.encode()).encode()==f.encode());
    f.draft--;refuses([&]{Frontier::decode(f.encode());});
    for(auto id:{"","../escape","a/b","a.b","a b"})refuses([&]{valid_id(id);});
    for(auto args:std::vector<std::vector<std::string>>{
        {"--session","A","--session-dspark","on"}, {"--session","A","--session-fixed","6"},
        {"--session","A","--session-steps","-1"}, {"--session","A","--session-steps","4294967296"},
        {"--session","A","--session-history","x","--session-tokens","y"},
        {"--session",std::string(129,'a')}, {"--session","A","--env","X=1"},
        {"--session","A","--session","B"}, {"--session","A","--session-op","park","--session-steps","1"}})
        refuses([&]{parse(args);});
    std::vector<std::string> a{"--session","A","--session-op","create","--session-dspark","on","--session-fixed","3","--session-steps","64"};
    auto c=parse(a);check(a.empty()&&c.id=="A"&&c.op=="create"&&c.steps==64&&c.fixed==3&&c.dspark);
    a={"--phases","exact0"};c=parse(a);check(!c.used&&a.size()==2);
    for(const auto* text:{"1 18446744073709551616","1 4294967296","1 -1","1 +1","1 12x","129280"}) {
        std::istringstream input(text);refuses([&]{read_tokens(input,100);});
    }
    {std::istringstream input("1 2 129279");check(read_tokens(input,3)==std::vector<std::uint32_t>({1,2,129279}));}
    {std::istringstream input("1 2");refuses([&]{read_tokens(input,1);});}
    {std::istringstream input("");check(read_tokens(input,0).empty());}
#ifdef _WIN32
    _putenv_s("STRIX_HASH_CACHE",(dir/"hash-cache").string().c_str());
#else
    setenv("STRIX_HASH_CACHE",(dir/"hash-cache").string().c_str(),1);
#endif
    strix::FileContentHashStats stats;
    write_file(dir/"persistent-model",bytes("abcdef"));
    auto digest=strix::sha256_file_content(dir/"persistent-model",&stats);check(!stats.cache_hit);
    check(strix::sha256_file_content(dir/"persistent-model",&stats)==digest&&stats.cache_hit);
#ifdef _WIN32
    // Metadata-only NTFS events (another hard link removed) keep the record; POSIX identities include
    // the change time, so the same event there conservatively re-hashes. Rewritten content never hits.
    fs::create_hard_link(dir/"persistent-model",dir/"persistent-model-link");fs::remove(dir/"persistent-model-link");
    check(strix::sha256_file_content(dir/"persistent-model",&stats)==digest&&stats.cache_hit);
#endif
    write_file(dir/"persistent-model",bytes("abXdefg"));
    check(strix::sha256_file_content(dir/"persistent-model",&stats)!=digest&&!stats.cache_hit);
    // Streaming liveness preserves the full digest and the verified cache hit.
    // Two full read blocks of the hash (8 MiB each) and a tail: progress at 0, after each block, at the end.
    const Bytes hash_payload(2U*8U*1024U*1024U+7U,42);
    write_file(dir/"progress-model",hash_payload);
    std::vector<std::uint64_t> hashed_bytes;
    const auto on_hash=[&](auto done,auto total) {
        require(done<=total&&total==hash_payload.size(),"invalid hash progress");hashed_bytes.push_back(done);
    };
    const auto progress_digest=strix::sha256_file_content(dir/"progress-model",&stats,on_hash);
    check(!stats.cache_hit && hashed_bytes.size()>=4 && hashed_bytes.front()==0 && hashed_bytes.back()==hash_payload.size());
    check(strix::model_content_identity_hex(progress_digest)==sha(hash_payload));
    hashed_bytes.clear();
    check(strix::sha256_file_content(dir/"progress-model",&stats,on_hash)==progress_digest&&stats.cache_hit);
    check(hashed_bytes.size()==1&&hashed_bytes.back()==hash_payload.size());
    std::vector<std::string> hash_stages;
    const Progress hash_progress=[&](const auto& stage,auto,auto){hash_stages.push_back(stage);};
    FileHashes hashes;write_file(dir/"model",bytes("abcdef"));auto h1=hashes.full(dir/"model",hash_progress);
    check(hash_stages.back().find("cache=miss ms=")!=std::string::npos);
    hash_stages.clear();check(hashes.full(dir/"model",hash_progress)==h1);
    check(hash_stages.back().find("cache=hit ms=")!=std::string::npos);
    write_file(dir/"model",bytes("abXdefg"));check(hashes.full(dir/"model")!=h1);
    a={"--phases","chats0","--session","A","--steps","3"};ignore_name(a);
    check(a==std::vector<std::string>({"--phases","chats0","--steps","3"}));
    auto ordinary=a;ignore_name(a);check(a==ordinary);
    a={"--session"};refuses([&]{ignore_name(a);});
    check(default_disk_budget(dir)>0);
    // Default budgets, bounded entry count and automatic disk publication/eviction.
    check(default_directory().filename()=="sessions");
    Config cfg;cfg.enabled=true;cfg.disk=true;cfg.dir=dir/"automatic";cfg.disk_explicit=true;cfg.disk_bytes=130*1024;
    Disk disk;disk.open(cfg,1024);check(disk.enabled());
    disk.put("A",s);check(disk.contains("A")&&disk.used()<=disk.budget());
    check(restore(disk.path("A"),s.compat,1024*1024).attention==s.attention);
    refuses([&]{disk.put("A",s);});auto advanced=s;++advanced.committed_generation;
    disk.put("A",advanced);disk.touch("A");check(disk.list()==std::vector<std::string>{"A"});
    check(disk.erase("A")&&!disk.erase("A"));
    cfg.disk_bytes=64*1024+s.attention.size()+s.frontier.encode().size();
    cfg.dir=dir/"tiny";Disk tiny_disk;tiny_disk.open(cfg,1024);tiny_disk.put("A",s);tiny_disk.put("B",s);
    check(!tiny_disk.contains("A")&&tiny_disk.contains("B"));
    Lru bounded(n*10,1);bounded.put("A",s);bounded.put("B",s);check(!bounded.contains("A")&&bounded.contains("B"));
    valid_id(std::string(128,'a'));check(unsigned_value("8589934592")==8*GiB);
    refuses([&]{unsigned_value("-1");});refuses([&]{unsigned_value("18446744073709551616");});
    a={"--session","A","--session-op","checkpoint","--session-durability","disk"};check(parse(a).path.empty());
    a={"--session","A","--session-op","restore","--session-durability","disk"};check(parse(a).op=="restore");
    a={"--session","A","--session-op","resume","--session-durability","disk"};check(parse(a).op=="restore");
    for(auto invalid:std::vector<std::vector<std::string>>{
            {"--session","A","--session-op","restore"},
            {"--session","A","--session-op","turn","--session-path","checkpoint"}})
        refuses([&]{parse(invalid);});
    // The qualified production route is automatic; unknown routes still refuse.
    Config automatic;automatic.enabled=true;qualify(automatic,"rtx-tp1-decode-rows");
    check(automatic.enabled&&automatic.reason.empty());
    Config unknown;unknown.enabled=true;qualify(unknown,"unknown-route");
    check(!unknown.enabled&&unknown.reason=="route not yet GPU-qualified: unknown-route");
    Config baseline_off;baseline_off.off("baseline");qualify(baseline_off,"rtx-tp1-decode-rows");
    check(!baseline_off.enabled&&baseline_off.reason=="baseline");
    Config explicit_gate;explicit_gate.enabled=true;explicit_gate.explicit_opt_in=true;
    qualify(explicit_gate,"rtx-tp1-decode-rows");check(explicit_gate.enabled);
    check(gpu_qualified_routes.size()==1);
    check(!explicit_request({"--session","A","--phases","chats0"}));
    check(explicit_request({"--session","A","--session-op","checkpoint"}));
    check(explicit_request({"--session","A","--session-path","checkpoint"}));
    check(explicit_request({"--session","A","--session-tokens","suffix"}));
    a={"--session","A","--phases","chats0","--seqs-file","prompt.tsv","--chat-prompt","12","--chat-steps","64"};
    check(automatic_name(a)=="A"&&a.front()=="--phases"&&!explicit_request(a));
    for(auto invalid_named:std::vector<std::vector<std::string>>{{"--session","A","--session","B"},
            {"--session","A","--env","STRIX_NUMERIC=2"},{"--session","A","--session-tokens","suffix"}})
        refuses([&]{automatic_name(invalid_named);});
    check(history_prefix(std::vector<std::uint32_t>{1,2},std::vector<std::uint32_t>{1,2,3}));
    check(history_prefix(std::vector<std::uint32_t>{1,2},std::vector<std::uint32_t>{1,2}));
    check(!history_prefix(std::vector<std::uint32_t>{1,2,3},std::vector<std::uint32_t>{1,2}));
    check(!history_prefix(std::vector<std::uint32_t>{1,2},std::vector<std::uint32_t>{1,3,2}));
    a={"--session","A","--session-op","resume","--session-path",(dir/"valid").string()};
    check(parse(a).op=="restore");
    for(auto op:{"checkpoint","park","resume","list","stats","delete"}) {
        a={"--session","A","--session-op",op};check(parse(a).op==op);
    }
    check(protocol_error("unsupported checkpoint schema").code=="session_incompatible");
    check(protocol_error("session mode/fixed-window change refused").status==409);
    check(protocol_error("checkpoint section SHA-256 mismatch").status==422);
    check(protocol_error("session disk budget exceeded").status==507);
    check(protocol_error("DSpark sessions require a worker launched with --draft-vocab full").code=="session_unsupported");
    check(protocol_error("unknown/evicted session (explicit restore required)").status==404);
    try{restore(dir/"missing",s.compat,1024);check(false);}
    catch(const Error& e){check(e.code=="session_not_found"&&e.status==404);}
    // An explicit immutable save/load does not consult automatic disk admission.
    Config disabled;disabled.enabled=false;disabled.disk=false;
    publish(dir/"explicit-disabled",s);
    check(restore(dir/"explicit-disabled",s.compat,1024*1024).attention==s.attention);
    // Progress is driven by completed work, elapsed time, or final completion.
    ProgressCadence cadence;
    const auto clock0=cadence.last;
    check(cadence.ready(0,false,clock0));
    check(!cadence.ready(1,false,clock0+std::chrono::seconds(1)));
    check(cadence.ready(16,false,clock0+std::chrono::seconds(1)));
    check(cadence.ready(17,false,clock0+std::chrono::seconds(4)));
    check(cadence.ready(18,true,clock0+std::chrono::seconds(4)));
    std::vector<std::string> progress_stages;
    const auto observed=restore(dir/"explicit-disabled",s.compat,1024*1024,
        [&](const std::string& stage,std::uint64_t done,std::uint64_t total) {
            require(done<=total,"invalid progress count");progress_stages.push_back(stage);
        });
    check(observed.attention==s.attention && observed.drafter==s.drafter);
    check(std::find(progress_stages.begin(),progress_stages.end(),"restore-verified:frontier.bin")!=progress_stages.end());
    check(std::find(progress_stages.begin(),progress_stages.end(),"restore-read:attention.bin")!=progress_stages.end());
    // Interrupted publications consume quota, including replacement peaks.
    fs::create_directories(cfg.dir/"leftover.tmp");
    write_file(cfg.dir/"leftover.tmp"/"payload",Bytes(128*1024));
    refuses([&]{tiny_disk.put("B",s);});
    check(tiny_disk.used()>=128*1024);
    check(tiny_disk.contains("B")); // a refused replacement preserves the old checkpoint
#ifdef _WIN32
    // A real independent process is refused immediately by the lifetime owner.
    const auto shared=dir/"lock-process";PROCESS_INFORMATION child{};
    {
        Config shared_config;shared_config.dir=shared;shared_config.disk=true;
        shared_config.disk_explicit=true;shared_config.disk_bytes=128*1024;
        Disk owner;owner.open(shared_config,1024);check(owner.enabled());
        STARTUPINFOW startup{};startup.cb=sizeof(startup);
        wchar_t exe[32768];require(GetModuleFileNameW(nullptr,exe,32768)>0,"test executable path");
        const std::wstring command=L"\""+std::wstring(exe)+L"\" --lock-child \""+fs::absolute(shared).wstring()+L"\"";
        const auto launch=[&] {
            auto cmd=command;
            require(CreateProcessW(nullptr,cmd.data(),nullptr,nullptr,FALSE,CREATE_NO_WINDOW,nullptr,nullptr,&startup,&child)!=0,"test child start");
            check(WaitForSingleObject(child.hProcess,10000)==WAIT_OBJECT_0);
            DWORD code=1;GetExitCodeProcess(child.hProcess,&code);
            CloseHandle(child.hThread);CloseHandle(child.hProcess);return code;
        };
        check(launch()==9);check(!fs::exists(shared/"child-acquired"));
    }
    // The exited owner's handle releases the tier for a fresh worker.
    {
        Config released;released.dir=shared;released.disk=true;released.disk_explicit=true;released.disk_bytes=128*1024;
        Disk next;next.open(released,1024);check(next.enabled());next.put("released",s);
    }
#endif
#ifdef _WIN32
    legacy_process_locks(dir/"external-locks");process_publication_boundaries(dir/"process-boundaries");
#else
    std::cout<<"NOT RUN Windows byte-range/process interoperability on this platform\n";
#endif
    std::cout<<"PASS "<<checks<<" CPU checks; artifacts "<<dir.string()<<"\n";return 0;
 } catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
