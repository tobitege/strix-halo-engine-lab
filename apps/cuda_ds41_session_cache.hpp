#pragma once
// Host-only envelope around the existing attention serializer. No CUDA dependency.
#include "strix/model_content_hash.hpp"
#include "strix/session/checkpoint_io.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <charconv>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#endif
namespace strix::ds41::session {
namespace io = strix::session::checkpoint_io;
#ifdef STRIX_SESSION_CONSTRUCTION_TEST
inline std::uint64_t session_filesystem_operations=0;
// Observe family filesystem policy. Shared mechanics have their own activity()
// witness; tests must check both. Production uses the standard namespace directly.
namespace fs {
using namespace std::filesystem;
#define STRIX_SESSION_FS(name) template<class... A> auto name(A&&... a) { \
    ++session_filesystem_operations;return std::filesystem::name(std::forward<A>(a)...);}
STRIX_SESSION_FS(absolute) STRIX_SESSION_FS(weakly_canonical) STRIX_SESSION_FS(exists)
STRIX_SESSION_FS(create_directories) STRIX_SESSION_FS(create_directory) STRIX_SESSION_FS(file_size)
STRIX_SESSION_FS(is_regular_file) STRIX_SESSION_FS(last_write_time) STRIX_SESSION_FS(remove)
STRIX_SESSION_FS(remove_all) STRIX_SESSION_FS(rename) STRIX_SESSION_FS(space)
STRIX_SESSION_FS(directory_iterator) STRIX_SESSION_FS(recursive_directory_iterator)
#undef STRIX_SESSION_FS
}
#else
namespace fs = std::filesystem;
#endif
// Test instrumentation observes real constructor calls, including empty MSVC maps.
#ifdef STRIX_SESSION_CONSTRUCTION_TEST
inline std::uint64_t resource_constructions=0;
inline std::uint64_t session_system_queries=0;
inline void resource_constructed(){++resource_constructions;}
#else
inline void resource_constructed(){}
#endif
constexpr std::uint64_t GiB=1024ULL*1024*1024;
// Window completion or elapsed time, never a heartbeat for blocked device work.
struct ProgressCadence {
    using Clock=std::chrono::steady_clock;
    Clock::time_point last=Clock::now();
    bool ready(std::uint32_t windows,bool final=false,Clock::time_point now=Clock::now()) {
        if(windows%16U==0 || final || now-last>=std::chrono::seconds(3)) {last=now;return true;}
        return false;
    }
};
using Progress=std::function<void(const std::string&,std::uint64_t,std::uint64_t)>;
inline std::uint64_t host_memory(bool available) {
#ifdef STRIX_SESSION_CONSTRUCTION_TEST
    extern std::uint64_t session_system_queries;
    ++session_system_queries;
#endif
#ifdef _WIN32
    MEMORYSTATUSEX m{};m.dwLength=sizeof m;
    if(GlobalMemoryStatusEx(&m))return available?m.ullAvailPhys:m.ullTotalPhys;
#else
    std::ifstream in("/proc/meminfo");std::string key,unit;std::uint64_t n;
    while(in>>key>>n>>unit)if(key==(available?"MemAvailable:":"MemTotal:"))return n*1024;
#endif
    return 0;
}
inline fs::path default_directory() {
    const auto x=std::getenv("XDG_CACHE_HOME");if(x&&*x)return fs::path(x)/"strix-engine"/"sessions";
    auto home=std::getenv("HOME");
#ifdef _WIN32
    if(!home||!*home)home=std::getenv("USERPROFILE");
#endif
    return home&&*home?fs::path(home)/".cache"/"strix-engine"/"sessions":fs::path{};
}
inline std::uint64_t default_disk_budget(fs::path dir) {
    std::error_code ec;
    while(!dir.empty()&&!fs::exists(dir,ec)) {
        if(ec||dir.parent_path()==dir)return 0;
        dir=dir.parent_path();
    }
    if(dir.empty()||ec)return 0;
    auto space=fs::space(dir,ec);return ec?0:std::min(64*GiB,space.available/4);
}
struct Config {
    // UINT64_MAX means unresolved. No host query or path construction at startup.
    bool enabled=true,disk=true;std::uint64_t ram=UINT64_MAX,disk_bytes=0,max_parked=8;
    bool explicit_opt_in=false;
    bool disk_explicit=false;fs::path dir;std::string reason;
    Config(){if(auto v=std::getenv("STRIX_SESSION_CACHE"))enabled=std::string(v)!="0";
        if(auto v=std::getenv("STRIX_SESSION_DISK"))disk=std::string(v)!="0";}
    void off(const std::string& why){enabled=false;reason=why;}
    void resolve_ram(){if(ram==UINT64_MAX)ram=std::min(8*GiB,host_memory(false)/10);}
    void resolve_directory(){if(dir.empty())dir=default_directory();}
};
// Only this table admits automatic routes after complete local GPU qualification.
inline constexpr std::array<const char*,1> gpu_qualified_routes{"rtx-tp1-decode-rows"};
inline void qualify(Config& c,const std::string& route) {
    if(c.enabled&&!c.explicit_opt_in&&
       std::none_of(gpu_qualified_routes.begin(),gpu_qualified_routes.end(),
                    [&](const char* v){return route==v;}))
        c.off("route not yet GPU-qualified: "+route);
}
inline std::uint64_t unsigned_value(const std::string& v) {
    std::uint64_t n=0;auto r=std::from_chars(v.data(),v.data()+v.size(),n);
    if(v.empty()||r.ec!=std::errc{}||r.ptr!=v.data()+v.size())throw std::runtime_error("invalid session budget");
    return n;
}
inline void ignore_name(std::vector<std::string>& args) {
    for(std::size_t i=0;i<args.size();) {
        if(args[i]=="--session"){
            if(i+1>=args.size())throw std::runtime_error("missing session id");
            args.erase(args.begin()+i,args.begin()+i+2);
        } else ++i;
    }
}
inline bool explicit_request(const std::vector<std::string>& args) {
    return std::any_of(args.begin(),args.end(),[](const auto& a){
        return a.rfind("--session-",0)==0;
    });
}
using Bytes = std::vector<std::uint8_t>;
struct Error : std::runtime_error {
    std::string code;int status;
    io::Operation operation=io::Operation::none;
    Error(std::string c,int n,const std::string& m):std::runtime_error(m),code(std::move(c)),status(n){}
};
inline Error protocol_error(const std::string& m) {
    if(m.find("DSpark sessions require a worker launched with --draft-vocab full")!=std::string::npos)
        return {"session_unsupported",422,m};
    if(m.find("compatibility")!=std::string::npos||m.find("schema")!=std::string::npos||
       m.find("unsupported session producer")!=std::string::npos||m.find("mode/fixed")!=std::string::npos)
        return {"session_incompatible",409,m};
    if(m.find("budget")!=std::string::npos||m.find("free space")!=std::string::npos)
        return {"session_budget",507,m};
    if(m.find("unknown/evicted")!=std::string::npos||m.find("not found")!=std::string::npos||m=="session_not_found")
        return {"session_not_found",404,m};
    return {"session_corrupt",422,m};
}
inline void require(bool ok, const char* message) { if (!ok) throw protocol_error(message); }
// One shared lock serializes quota accounting, publication, reads and eviction.
class DirectoryLock {
    std::unique_ptr<io::DirectoryLock> lock_;
public:
    explicit DirectoryLock(fs::path dir,bool owner=false) {
        if(dir.empty())dir=".";
        try {
            lock_=std::make_unique<io::DirectoryLock>(dir,
                owner?io::LockMode::nonblocking:io::LockMode::blocking,
                owner?".owner.lock":".session.lock",io::LockRange::first_byte);
        } catch(const io::Error& e) {
            if(owner&&e.lock_busy)throw Error("session_tier_busy",409,"managed session tier is owned by another worker");
            auto error=protocol_error(e.what());error.operation=e.operation;throw error;
        }
    }
    DirectoryLock(const DirectoryLock&)=delete;
    DirectoryLock& operator=(const DirectoryLock&)=delete;
};
inline std::string sha(std::span<const std::uint8_t> b) {
    return model_content_identity_hex(sha256_bytes_fast(std::as_bytes(b)));
}
inline Bytes bytes(const std::string& s) { return Bytes(s.begin(), s.end()); }
inline void u64(Bytes& b, std::uint64_t n) {
    for (unsigned i=0;i<8;++i) b.push_back(static_cast<std::uint8_t>(n>>(8*i)));
}
inline void blob(Bytes& b, std::span<const std::uint8_t> v) { u64(b,v.size()); b.insert(b.end(),v.begin(),v.end()); }
inline void text(Bytes& b,const std::string& v) { blob(b,bytes(v)); }
// Operational paths and device selector spelling do not describe numerical state.
inline bool compat_operational(const std::string& arg) {
    for(const auto* option : {"--serve","--serve-idle","--serve-max","--serve-session-mib",
        "--qual-dump","--json","--timeline","--draft-trace","--controller-save",
        "--ref","--seqs-file","--chats","--chat-prompt","--chat-steps","--steps",
        "--phases","--import-state","--residency-save","--residency-seed","--warm",
        "--draft-device","--session-cache","--session-disk","--session-ram-bytes",
        "--session-max-parked","--session-dir","--session-disk-bytes"}) if(arg==option)return true;
    return false;
}
inline bool compat_environment(std::wstring v) {
    std::transform(v.begin(),v.end(),v.begin(),[](wchar_t c)->wchar_t{return c>=L'a'&&c<=L'z'?static_cast<wchar_t>(c-(L'a'-L'A')):c;});
    const auto key=v.substr(0,v.find(L'='));
    if(key==L"CUDA_VISIBLE_DEVICES"||key==L"STRIX_HASH_CACHE"||key==L"STRIX_SESSION_CACHE"||key==L"STRIX_SESSION_DISK")return false;
    return key.rfind(L"STRIX_",0)==0||key.rfind(L"CUDA_",0)==0||key.rfind(L"NVIDIA_",0)==0;
}
#ifdef _WIN32
// Call only after restoring the launch options and applying their environment.
// No environment block, strings or retained snapshot exists before session use.
inline Bytes compat_environment_record() {
    const auto env=GetEnvironmentStringsW();require(env!=nullptr,"cannot read session environment");
    const auto release=[](wchar_t* p){FreeEnvironmentStringsW(p);};
    std::unique_ptr<wchar_t,decltype(release)> owner(env,release);
    std::vector<std::wstring> values;
    for(auto q=env;*q;q+=wcslen(q)+1)if(compat_environment(q))values.emplace_back(q);
    std::sort(values.begin(),values.end());
    Bytes record;
    for(const auto& v:values)blob(record,std::span(reinterpret_cast<const std::uint8_t*>(v.data()),v.size()*sizeof(wchar_t)));
    return record;
}
#endif
inline void compat_hardware(Bytes& record, std::span<const std::uint8_t> uuid,
                            int major,int minor,int driver,int runtime) {
    text(record,"cuda-resolved-v1");blob(record,uuid);
    for(auto n:{major,minor,driver,runtime})u64(record,static_cast<std::uint64_t>(n));
}
template<class HashFile>
inline void compat_arguments(Bytes& record,const std::vector<std::string>& args,HashFile hash_file) {
    for(std::size_t i=0;i<args.size();++i) {
        if(compat_operational(args[i])){if(i+1<args.size())++i;continue;}
        if(args[i]=="--serve-keep-on-fail")continue;
        text(record,args[i]);
        auto file_arg=args[i];const auto colon=file_arg.rfind(':');
        if(colon!=std::string::npos&&colon>1&&colon+1<file_arg.size()&&
           file_arg.find_first_not_of("0123456789",colon+1)==std::string::npos)file_arg.erase(colon);
        if(fs::is_regular_file(fs::path(file_arg)))text(record,hash_file(file_arg));
    }
}
struct Reader {
    std::span<const std::uint8_t> b;
    std::uint64_t integer() {
        require(b.size()>=8,"truncated session integer"); std::uint64_t n=0;
        for(unsigned i=0;i<8;++i)n|=std::uint64_t(b[i])<<(8*i);
        b=b.subspan(8); return n;
    }
    std::span<const std::uint8_t> data() {
        auto n=integer(); require(n<=b.size(),"truncated session section");
        auto out=b.first(static_cast<std::size_t>(n)); b=b.subspan(static_cast<std::size_t>(n)); return out;
    }
    std::string string(){auto v=data();return std::string(v.begin(),v.end());}
    void end()const{require(b.empty(),"session trailing bytes");}
};
inline void valid_id(const std::string& id) {
    require(!id.empty()&&id.size()<=128&&id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")==std::string::npos,
            "session id must be 1..128 ASCII letters/digits/_/-");
}
// Full-file hash cache keyed by canonical path, size, mtime, volume/inode.
// Model artifacts must remain immutable while the loaded worker owns them.
inline std::string file_identity(const fs::path& p) {
    auto key=fs::weakly_canonical(p).generic_string()+":"+std::to_string(fs::file_size(p))+":"+
        std::to_string(fs::last_write_time(p).time_since_epoch().count());
#ifdef _WIN32
    HANDLE h=CreateFileW(p.c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
                         nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    require(h!=INVALID_HANDLE_VALUE,"cannot identify session artifact");
    BY_HANDLE_FILE_INFORMATION i{}; const bool ok=GetFileInformationByHandle(h,&i)!=0; CloseHandle(h);
    require(ok,"cannot stat session artifact");
    key+=":"+std::to_string(i.dwVolumeSerialNumber)+":"+std::to_string(i.nFileIndexHigh)+":"+std::to_string(i.nFileIndexLow);
#else
    struct stat s{}; require(::stat(p.c_str(),&s)==0,"cannot stat session artifact");
    key+=":"+std::to_string(s.st_dev)+":"+std::to_string(s.st_ino);
#endif
    return key;
}
class FileHashes {
public:
    FileHashes(){resource_constructed();}
    std::string full(const fs::path& p,const Progress& progress={}) {
        const auto t=std::chrono::steady_clock::now();
        const auto key=file_identity(p);
        FileContentHashStats stats;
        auto last=t;std::uint64_t last_bytes=0;
        if(progress)progress("hash:"+p.filename().string()+" cache=lookup ms=0",0,fs::file_size(p));
        auto digest=model_content_identity_hex(sha256_file_content(p,&stats,[&](auto done,auto total) {
            const auto now=std::chrono::steady_clock::now();
            if(progress&&(done==total||done-last_bytes>=256ULL*1024*1024||now-last>=std::chrono::seconds(3))) {
                progress("hash:"+p.filename().string()+" cache="+(stats.cache_hit?"hit":"miss")+
                    " ms="+std::to_string(std::chrono::duration<double,std::milli>(now-t).count()),done,total);
                last=now;last_bytes=done;
            }
        }));
        require(file_identity(p)==key,"artifact changed during full-content hash");
        std::cerr<<"session_hash path="<<p.string()<<" cache="<<(stats.cache_hit?"hit":"miss")
                 <<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()<<"\n";
        return digest;
    }
};
// Committed frontier: pending is selected, unconsumed. No RNG. Fixed DSpark
// decisions depend only on k and remaining turn length; no timing controller.
struct Frontier {
    std::uint64_t consumed=0,emitted=0,pending=0,accepted=0,draft=0,turns=0,fixed=0;
    bool dspark=false;
    Bytes encode()const {
        Bytes b;text(b,"rtx-frontier-v1/greedy/fixed/decode-rows");
        for(auto v:{consumed,emitted,pending,accepted,draft,turns,fixed,std::uint64_t(dspark)})u64(b,v);
        return b;
    }
    static Frontier decode(std::span<const std::uint8_t> b) {
        Reader r{b};require(r.string()=="rtx-frontier-v1/greedy/fixed/decode-rows","unsupported session producer");
        Frontier f;f.consumed=r.integer();f.emitted=r.integer();f.pending=r.integer();f.accepted=r.integer();
        f.draft=r.integer();f.turns=r.integer();f.fixed=r.integer();auto d=r.integer();r.end();
        require(d<=1&&f.fixed<=5&&f.pending<129280&&f.accepted==f.consumed,"invalid session frontier");
        f.dspark=d!=0;require(f.dspark||(f.fixed==0&&f.draft==0),"invalid non-draft frontier");
        require(!f.dspark||(f.fixed>0&&f.draft==f.consumed),"invalid draft frontier");return f;
    }
};
struct Snapshot {
    static std::string new_lineage() {
        std::random_device random;std::string out;const char* hex="0123456789abcdef";
        for(unsigned i=0;i<16;++i){const auto b=random()&255;out+=hex[b>>4];out+=hex[b&15];}return out;
    }
    std::string compat,lineage=new_lineage();std::uint64_t committed_generation=0;
    Bytes attention,drafter; Frontier frontier;
    std::uint64_t size()const{return attention.size()+drafter.size()+compat.size()+lineage.size()+8+frontier.encode().size();}
};
inline bool same_location(const fs::path& a,const fs::path& b) {
#ifdef _WIN32
    const auto x=a.native(),y=b.native();
    return CompareStringOrdinal(x.c_str(),static_cast<int>(x.size()),y.c_str(),static_cast<int>(y.size()),TRUE)==CSTR_EQUAL;
#else
    return a==b;
#endif
}
// Keep the launch seed lazy, with no work for workers that never use sessions.
// Session seed = the launch --import-state argument's content, the same content every ordinary
// request reads for that argument. Arguments are compared as given, without filesystem queries.
// After an ordinary request replaced the imported state with another argument, the first session
// use reloads the launch argument instead of copying anything ahead of time. Only a synthetic
// launch seed (no file) is preserved before replacement; it has no file to reload. State's copy
// constructor must own its nested storage independently of imported.
template<class State> class LaunchSeed {
    std::unique_ptr<State> seed_;
    fs::path launch_path_;
    std::function<State(const fs::path&)> load_;
    bool imported_is_launch_=true;
public:
    explicit LaunchSeed(bool /*automatic sessions*/,const fs::path& file={},
                        std::function<State(const fs::path&)> load={}):load_(std::move(load)){
        if(!file.empty()&&file!="none")launch_path_=file;
    }
    const State& materialize(const State& imported) {
        if(!seed_) {
            if(imported_is_launch_)seed_=std::make_unique<State>(imported);
            else {
                require(!launch_path_.empty()&&bool(load_),"launch session seed was not preserved before import replacement");
                seed_=std::make_unique<State>(load_(launch_path_));
            }
        }
        return *seed_;
    }
    // Before an ordinary request loads its import: keep a synthetic launch seed (nothing to reload).
    void before_replace(const State& imported,bool /*automatic sessions*/,const fs::path& /*incoming*/={}) {
        if(launch_path_.empty()&&!seed_&&imported_is_launch_)materialize(imported);
    }
    // Only after that load succeeded: the imported state is the launch argument's content exactly
    // when the argument matches. A failed load leaves both the state and this provenance unchanged.
    void after_replace(const fs::path& incoming) {
        imported_is_launch_=!launch_path_.empty()&&incoming==launch_path_;
    }
    bool materialized()const{return bool(seed_);}
};

// Bounded diagnostics only. Admission never relies on these.
class Names {
    std::map<std::string,std::uint64_t> names_;std::uint64_t tick_=0;std::size_t bound_;
public:
    explicit Names(std::size_t bound=4096):bound_(std::max<std::size_t>(1,bound)){resource_constructed();}
    void admitted(const std::string& id){names_.erase(id);}
    void evicted(const std::string& id){
        names_[id]=++tick_;
        while(names_.size()>bound_) {
            auto oldest=std::min_element(names_.begin(),names_.end(),[](const auto& a,const auto& b){return a.second<b.second;});
            names_.erase(oldest);
        }
    }
    void deleted(const std::string& id){evicted(id);}
    void turn(const std::string& id)const {
        throw Error("session_not_found",404,names_.count(id)?"session evicted; create explicitly":"session not found");
    }
    void missing(const std::string& id)const{turn(id);}
    std::size_t size()const{return names_.size();}
    bool known(const std::string& id)const{return names_.count(id)!=0;}
};
template<class F> inline void checkpoint_io_call(F&& f) {
    try {f();}catch(const io::Error& e){auto error=protocol_error(e.what());error.operation=e.operation;throw error;}
}
inline void write_file(const fs::path& p,std::span<const std::uint8_t> b) {
    checkpoint_io_call([&]{io::atomic_write(p,std::as_bytes(b));});
}
inline Bytes read_file(const fs::path& p,std::uint64_t limit,const Progress& progress={}) {
    Bytes b;
    checkpoint_io_call([&]{io::read(p,0,limit,[&](auto n){
        b.resize(static_cast<std::size_t>(n));return std::as_writable_bytes(std::span(b));
    },io::DirectIo::buffered,[&](auto chunk,auto at){
        if(progress)progress("restore-read:"+p.filename().string(),at+chunk.size(),b.size());
    },8U*1024U*1024U);});
    return b;
}
// New immutable directory only. Interrupted .tmp directories remain inspectable.
// Publish manifest last, flush every file, then atomic directory rename.
inline void publish_unlocked(const fs::path& dest,const Snapshot& s,const std::function<void()>& flushed={}) {
    require(!fs::exists(dest),"checkpoint already exists (immutable)");
    if(!dest.parent_path().empty())fs::create_directories(dest.parent_path());
    const auto tmp=fs::path(dest.string()+".tmp");require(fs::create_directory(tmp),"checkpoint temporary directory already exists");
    Bytes manifest;text(manifest,"strix-rtx-session-v2");text(manifest,s.compat);
    text(manifest,s.lineage);u64(manifest,s.committed_generation);const auto f=s.frontier.encode();
    const std::array<std::pair<const char*,std::span<const std::uint8_t>>,3> sections{{
        {"attention.bin",s.attention},{"drafter.bin",s.drafter},{"frontier.bin",f}}};
    u64(manifest,sections.size());
    for(const auto& [name,b]:sections) {
        text(manifest,name);u64(manifest,0);text(manifest,"replicated");u64(manifest,0);u64(manifest,b.size());text(manifest,sha(b));
        write_file(tmp/name,b);
    }
    const auto root=sha(manifest);text(manifest,root);write_file(tmp/"manifest.bin",manifest);checkpoint_io_call([&]{io::sync_directory(tmp);});
    if(flushed)flushed();
    checkpoint_io_call([&]{io::publish_directory(tmp,dest);});
}
inline void publish(const fs::path& dest,const Snapshot& s) {
    DirectoryLock lock(dest.parent_path());
    if(fs::exists(fs::path(dest.string()+".current")))
        throw Error("session_conflict",409,"path inside the managed session tier; use durability disk without a path");
    publish_unlocked(dest,s);
}
inline Snapshot restore_unlocked(const fs::path& dir,const std::string& compat,std::uint64_t limit,const Progress& progress={}) {
    auto m=read_file(dir/"manifest.bin",64*1024,progress);Reader r{m};
    require(r.string()=="strix-rtx-session-v2","unsupported checkpoint schema");
    Snapshot s;s.compat=r.string();require(s.compat==compat,"session compatibility mismatch");
    s.lineage=r.string();s.committed_generation=r.integer();
    require(s.lineage.size()==32&&s.lineage.find_first_not_of("0123456789abcdef")==std::string::npos,"invalid session lineage");
    require(r.integer()==3,"invalid section count");
    std::uint64_t total=0;
    for(const char* expected:{"attention.bin","drafter.bin","frontier.bin"}) {
        const auto name=r.string();require(name==expected,"invalid section name/order");
        require(r.integer()==0&&r.string()=="replicated"&&r.integer()==0,"invalid geometry/offset");
        const auto n=r.integer();const auto digest=r.string();require(n<=limit-total,"session payload exceeds host budget");total+=n;
        auto b=read_file(dir/name,n,progress);
        if(progress)progress("restore-verify:"+name,0,n);
        require(b.size()==n&&sha(b)==digest,"checkpoint section SHA-256 mismatch");
        if(progress)progress("restore-verified:"+name,n,n);
        if(name=="attention.bin")s.attention=std::move(b);else if(name=="drafter.bin")s.drafter=std::move(b);else s.frontier=Frontier::decode(b);
    }
    const auto root_at=m.size()-r.b.size();const auto root=r.string();r.end();
    require(sha(std::span(m).first(root_at))==root,"checkpoint manifest SHA-256 mismatch");
    require(s.size()<=limit,"session snapshot exceeds host budget");return s;
}
// A managed name is a small, checksummed pointer to one complete generation.
struct Pointer {
    std::string id,directory,lineage,manifest;std::uint64_t generation=0;
    Bytes encode()const {
        Bytes b;text(b,"strix-rtx-session-pointer-v1");text(b,id);text(b,directory);
        text(b,lineage);u64(b,generation);text(b,manifest);text(b,sha(b));return b;
    }
    static Pointer decode(std::span<const std::uint8_t> b,const std::string& id) {
        Reader r{b};require(r.string()=="strix-rtx-session-pointer-v1","unsupported session pointer schema");
        Pointer p;p.id=r.string();p.directory=r.string();p.lineage=r.string();p.generation=r.integer();p.manifest=r.string();
        const auto root_at=b.size()-r.b.size();const auto root=r.string();r.end();
        require(sha(b.first(root_at))==root,"checkpoint pointer SHA-256 mismatch");
        valid_id(id);const auto prefix=id+".g";
        require(p.id==id&&p.directory.rfind(prefix,0)==0&&p.directory.size()>prefix.size()&&
                p.directory.find_first_not_of("0123456789",prefix.size())==std::string::npos,"invalid checkpoint pointer name");
        return p;
    }
};
inline fs::path pointer_path(const fs::path& name){return fs::path(name.string()+".current");}
inline Pointer read_pointer(const fs::path& name) {
    return Pointer::decode(read_file(pointer_path(name),65536),name.filename().string());
}
inline Snapshot restore(const fs::path& dir,const std::string& compat,std::uint64_t limit,const Progress& progress={},bool managed=true) {
    const auto parent=dir.parent_path().empty()?fs::path("."):dir.parent_path();
    std::optional<DirectoryLock> lock;
    if(fs::exists(parent/".session.lock"))lock.emplace(parent);
    const bool directory=fs::exists(dir),pointer=fs::exists(pointer_path(dir));
    if(pointer&&(!managed||directory))
        throw Error("session_conflict",409,"path inside the managed session tier; use durability disk without a path");
    if(!directory&&!pointer)throw Error("session_not_found",404,"session checkpoint not found");
    try {
        if(pointer) {
            const auto p=read_pointer(dir);const auto generation=parent/p.directory;
            require(sha(read_file(generation/"manifest.bin",65536))==p.manifest,"checkpoint manifest SHA-256 mismatch");
            auto s=restore_unlocked(generation,compat,limit,progress);
            require(s.lineage==p.lineage&&s.committed_generation==p.generation,"checkpoint pointer generation mismatch");return s;
        }
        return restore_unlocked(dir,compat,limit,progress);
    } catch(const Error&){throw;}
    catch(const std::exception& e){throw Error("session_corrupt",422,e.what());}
}
struct ExplicitPath {fs::path path;bool managed=false;};
inline ExplicitPath route_explicit_path(const fs::path& requested,const fs::path& tier,const std::string& id) {
    valid_id(id);
    const auto canonical=[](const fs::path& p){return fs::weakly_canonical(fs::absolute(p));};
    ExplicitPath out{canonical(requested)};
    const auto conflict=[](){throw Error("session_conflict",409,"path inside the managed session tier; use durability disk without a path");};
    if(!tier.empty()) {
        const auto root=canonical(tier),entry=canonical(tier/id);
        if(same_location(out.path,entry)) {
            // Managed names select pointers only. A directory here cannot prove
            // ownership and must never be hidden by a later managed publication.
            if(fs::exists(entry))conflict();
            out.path=entry;out.managed=true;return out;
        }
        // A managed entry may itself be a link to a location outside the tier.
        if(same_location(canonical(requested.parent_path().empty()?fs::path("."):requested.parent_path()),root))conflict();
        // This also covers pointer/generation names and aliases into their payloads.
        for(auto p=out.path;!p.empty();) {
            if(same_location(p,root))conflict();
            const auto parent=p.parent_path();if(parent==p)break;p=parent;
        }
    }
    // An owner-managed directory must never compete with a pointer at that name.
    if(fs::exists(pointer_path(out.path)))conflict();
    return out;
}
class Lru {
    struct Entry{Snapshot snapshot;std::uint64_t tick,generation;};
    std::map<std::string,Entry> entries_;std::uint64_t budget_,used_=0,tick_=0,evicted_=0,max_entries_;
    std::uint64_t generation_=0,active_generation_=0;
    std::string active_;
    Names* names_=nullptr;
public:
    // A pending admission owns its CPU copy. Discarding it cannot change LRU
    // order, live identity or disk state. Generation zero always needs import.
    struct Pending {Snapshot snapshot;std::uint64_t generation=0;};
    explicit Lru(std::uint64_t budget,std::uint64_t max_entries=8,Names* names=nullptr):budget_(budget),max_entries_(max_entries),names_(names){resource_constructed();}
    bool contains(const std::string& id)const{return entries_.count(id)!=0;}
    Pending stage(const std::string& id)const {
        auto it=entries_.find(id);require(it!=entries_.end(),"unknown/evicted session (explicit restore required)");
        return {it->second.snapshot,it->second.generation};
    }
    bool keep_live(const std::string& id,const Pending& pending)const {
        auto it=entries_.find(id);
        return pending.generation!=0&&it!=entries_.end()&&it->second.generation==pending.generation&&
            active_==id&&active_generation_==pending.generation;
    }
    void admit(const std::string& id,const Pending& pending) {
        if(pending.generation==0)put(id,pending.snapshot);
        else {
            auto it=entries_.find(id);
            require(it!=entries_.end()&&it->second.generation==pending.generation,"stale session admission");
            it->second.tick=++tick_;
        }
    }
    void clear_active(){active_.clear();active_generation_=0;}
    const std::string& active()const{return active_;}
    void activate(const std::string& id) {
        auto it=entries_.find(id);require(it!=entries_.end(),"cannot activate missing session");
        active_=id;active_generation_=it->second.generation;
    }
    const Snapshot& get(const std::string& id) {
        auto it=entries_.find(id);require(it!=entries_.end(),"unknown/evicted session (explicit restore required)");
        it->second.tick=++tick_;return it->second.snapshot;
    }
    void put(const std::string& id,Snapshot s) {
        valid_id(id);const auto n=s.size();require(n<=budget_&&max_entries_>0,"session exceeds host byte budget");erase(id);
        while(used_>budget_-n||entries_.size()>=max_entries_) {
            auto v=std::min_element(entries_.begin(),entries_.end(),[](const auto& a,const auto& b){return a.second.tick<b.second.tick;});
            // Capacity eviction is RAM-only, even when a durable older snapshot exists.
            if(names_)names_->evicted(v->first);
            erase(v->first);++evicted_;
        }
        entries_.emplace(id,Entry{std::move(s),++tick_,++generation_});used_+=n;
    }
    bool erase(const std::string& id) {
        auto it=entries_.find(id);if(it==entries_.end())return false;
        if(active_==id)clear_active();
        used_-=it->second.snapshot.size();entries_.erase(it);return true;
    }
    std::vector<std::string> list()const{std::vector<std::string> ids;for(auto& [id,e]:entries_)ids.push_back(id);return ids;}
    std::uint64_t used()const{return used_;} std::uint64_t budget()const{return budget_;} std::uint64_t evicted()const{return evicted_;}
};
// Publication is serialized with reads and quota eviction. The pointer is the
// commit record; orphan generations never become visible on their own.
class Disk {
    fs::path dir_;std::uint64_t budget_=0,bound_=0;Config* deferred_=nullptr;bool opened_=false,busy_=false;
    std::unique_ptr<DirectoryLock> owner_;
    std::optional<Error> unavailable_;
    void check_owner()const{if(busy_)throw Error("session_tier_busy",409,"managed session tier is owned by another worker");}
    static bool linked(const fs::directory_entry& e) {
#ifdef _WIN32
#ifdef STRIX_SESSION_CONSTRUCTION_TEST
        ++session_filesystem_operations;
#endif
        // MSVC reports directory junctions as directories, not symlinks. They
        // are still aliases and must never become cleanup victims or quota owners.
        const auto attributes=GetFileAttributesW(e.path().c_str());
        require(attributes!=INVALID_FILE_ATTRIBUTES,"checkpoint entry attributes failed");
        if(attributes&FILE_ATTRIBUTE_REPARSE_POINT)return true;
#endif
        return e.is_symlink();
    }
    static bool generation_name(const std::string& name,std::string& id) {
        auto at=name.rfind(".g");if(at==std::string::npos)return false;
        auto digits=name.substr(at+2);if(digits.ends_with(".tmp"))digits.resize(digits.size()-4);
        if(digits.empty()||digits.find_first_not_of("0123456789")!=std::string::npos)return false;
        id=name.substr(0,at);try{valid_id(id);}catch(...){return false;}return true;
    }
    static bool pointer_temporary(const std::string& filename) {
        const auto at=filename.find(".current.tmp");if(at==std::string::npos)return false;
        try{valid_id(filename.substr(0,at));}catch(...){return false;}
        const auto tail=filename.substr(at+12);
        if(tail.empty())return true;
        if(tail.front()!='-')return false;
        const auto split=tail.find('-',1);
        return split!=std::string::npos&&split>1&&split+1<tail.size()&&
            tail.substr(1,split-1).find_first_not_of("0123456789")==std::string::npos&&
            tail.substr(split+1).find_first_not_of("0123456789")==std::string::npos;
    }
    std::vector<fs::path> entries()const {
        std::vector<fs::path> out;if(dir_.empty())return out;
        for(const auto& e:fs::directory_iterator(dir_))if(!linked(e)&&e.is_regular_file()&&e.path().extension()==".current") {
            const auto id=e.path().stem().string();try{valid_id(id);out.push_back(dir_/id);}catch(...){}
        }
        return out;
    }
    static std::uint64_t size(const fs::path& p) {
        if(fs::is_regular_file(p))return fs::file_size(p);
        std::uint64_t n=0;for(auto& e:fs::recursive_directory_iterator(p))if(e.is_regular_file())n+=e.file_size();return n;
    }
    std::uint64_t used_unlocked()const {
        std::uint64_t n=0;
        if(!dir_.empty())for(auto& e:fs::directory_iterator(dir_))if(!linked(e)&&
            (e.is_directory()||e.path().extension()==".current"||pointer_temporary(e.path().filename().string())))n+=size(e.path());
        return n;
    }
    void erase_unlocked(const fs::path& name) {
        const auto p=read_pointer(name);fs::remove(pointer_path(name));checkpoint_io_call([&]{io::sync_directory(dir_);});
        fs::remove_all(dir_/p.directory);
    }
    void cleanup_unlocked() {
        for(const auto& e:fs::directory_iterator(dir_)) {
            if(linked(e))continue;
            const auto filename=e.path().filename().string();std::string id;
            if(e.is_directory()&&generation_name(filename,id)) {
                const auto name=dir_/id;
                if(fs::exists(pointer_path(name))) {
                    // A corrupt pointer must refuse restore, never expose an orphan.
                    try{if(read_pointer(name).directory==filename)continue;}catch(...){continue;}
                }
                fs::remove_all(e.path());
            } else if(e.is_regular_file()&&pointer_temporary(filename))fs::remove(e.path());
        }
    }
public:
    enum class Point {files_flushed,generation_published,pointer_flushed,pointer_committed,old_removed};
    using Fault=std::function<void(Point)>;
    Disk(){resource_constructed();}
    bool enabled()const{check_owner();return budget_!=0;}
    bool opened()const{return opened_;}
    bool available()const{return budget_!=0&&!busy_;}
    // Configuration alone must not stat, create, lock, probe, scan or evict.
    void configure(Config& c,std::uint64_t bound){deferred_=&c;bound_=bound;}
    void ensure_open(){check_owner();if(!opened_&&deferred_)open(*deferred_,bound_);}
    void require_available()const {
        check_owner();if(unavailable_)throw *unavailable_;
        if(!budget_)throw Error("session_tier_unavailable",503,"managed session tier disabled");
    }
    void open(Config& c,std::uint64_t bound) {
        check_owner();if(opened_)return;opened_=true;if(!c.disk)return;
        try {
            c.resolve_directory();
            require(!c.dir.empty(),"no home/cache directory");owner_=std::make_unique<DirectoryLock>(c.dir,true);
            DirectoryLock lock(c.dir);
            auto free=fs::space(c.dir).available;require(bound<=free/2,"insufficient checkpoint free space");
            auto probe=c.dir/(".probe-"+Snapshot::new_lineage());require(fs::create_directory(probe),"disk probe failed");
            try{write_file(probe/"write",{});}catch(...){fs::remove_all(probe);throw;}fs::remove_all(probe);
            budget_=c.disk_explicit?c.disk_bytes:std::min(64*GiB,free/4);require(budget_>=bound,"disk budget below one checkpoint");
            dir_=c.dir;c.disk_bytes=budget_;cleanup_unlocked();
            auto all=entries();std::sort(all.begin(),all.end(),[](auto& a,auto& b){return fs::last_write_time(pointer_path(a))<fs::last_write_time(pointer_path(b));});
            for(auto& name:all)if(used_unlocked()>budget_)erase_unlocked(name);
        }catch(const Error& e){
            if(e.code=="session_tier_busy"){busy_=true;throw;}
            budget_=0;unavailable_.emplace("session_tier_unavailable",e.status==507?507:503,e.what());
            std::cerr<<"session disk: unavailable ("<<e.what()<<")\n";
        }catch(const std::exception& e){budget_=0;unavailable_.emplace("session_tier_unavailable",503,e.what());
            std::cerr<<"session disk: unavailable ("<<e.what()<<")\n";}
    }
    fs::path path(const std::string& id)const{valid_id(id);require(enabled(),"session disk disabled");return dir_/id;}
    bool contains(const std::string& id)const{if(!enabled())return false;DirectoryLock lock(dir_);return fs::is_regular_file(pointer_path(path(id)));}
    std::vector<std::string> list()const{std::vector<std::string> out;if(!enabled())return out;DirectoryLock lock(dir_);for(auto& e:entries())out.push_back(e.filename().string());return out;}
    std::uint64_t used()const{if(!enabled())return 0;DirectoryLock lock(dir_);return used_unlocked();}
    std::uint64_t budget()const{return budget_;}
    void touch(const std::string& id){check_owner();DirectoryLock lock(dir_);fs::last_write_time(pointer_path(path(id)),fs::file_time_type::clock::now());}
    bool erase(const std::string& id){if(!enabled())return false;DirectoryLock lock(dir_);auto name=path(id);if(!fs::exists(pointer_path(name)))return false;erase_unlocked(name);return true;}
    void put(const std::string& id,const Snapshot& s,const Fault& fault={}) {
        const auto n=s.attention.size()+s.drafter.size()+s.frontier.encode().size()+64*1024;
        require(enabled(),"session disk budget exceeded");DirectoryLock lock(dir_);
        const auto name=path(id);std::optional<Pointer> previous;
        if(fs::exists(pointer_path(name))) {
            previous=read_pointer(name);
            if(previous->lineage!=s.lineage || s.committed_generation<=previous->generation)
                throw Error("session_conflict",409,"stale session writer or conflicting lineage");
        }
        require(n<=budget_,"session disk budget exceeded");
        auto all=entries();std::sort(all.begin(),all.end(),[](auto& a,auto& b){return fs::last_write_time(pointer_path(a))<fs::last_write_time(pointer_path(b));});
        auto remaining=used_unlocked();
        for(auto& e:all)if(e!=name){const auto p=read_pointer(e);remaining-=size(dir_/p.directory)+size(pointer_path(e));}
        require(remaining<=budget_-n,"session disk budget exceeded by replacement peak or interrupted writes");
        for(auto& e:all)if(e!=name&&used_unlocked()>budget_-n)erase_unlocked(e);
        // Directory serials name publications, independent of lineage turn counters.
        std::uint64_t serial=0;
        for(const auto& e:fs::directory_iterator(dir_)) {
            const auto filename=e.path().filename().string();std::string owner;
            if(generation_name(filename,owner)&&owner==id) {
                auto digits=filename.substr(id.size()+2);if(digits.ends_with(".tmp"))digits.resize(digits.size()-4);
                serial=std::max(serial,unsigned_value(digits));
            }
        }
        require(serial<UINT64_MAX,"session generation exhausted");
        const auto generation=dir_/(id+".g"+std::to_string(serial+1));
        const auto point=[&](Point p){if(fault)fault(p);};
        publish_unlocked(generation,s,[&]{point(Point::files_flushed);});point(Point::generation_published);
        Pointer next{id,generation.filename().string(),s.lineage,sha(read_file(generation/"manifest.bin",65536)),s.committed_generation};
        const auto current=pointer_path(name);const auto encoded=next.encode();
        checkpoint_io_call([&]{
            io::Publication pointer(current);pointer.write(std::as_bytes(std::span(encoded)));
            pointer.commit([&]{point(Point::pointer_flushed);});
        });
        point(Point::pointer_committed);
        if(previous)fs::remove_all(dir_/previous->directory);
        point(Point::old_removed);
    }
};
// Constructed only by the first session request, never by ordinary startup.
struct Resources {
    Names names;Lru ram;Disk disk;FileHashes hashes;std::string compat;
    Resources(Config& c,std::uint64_t budget,std::uint64_t bound)
        :ram(budget,c.max_parked==0?0:(c.max_parked==UINT64_MAX?UINT64_MAX:c.max_parked+1),&names){disk.configure(c,bound);}
};
// Extract text fields first: numeric failbit+eofbit must never drop a final ID.
inline std::vector<std::uint32_t> read_tokens(std::istream& in,std::size_t limit) {
    std::vector<std::uint32_t> out;std::string field;
    while(in>>field) {
        std::uint32_t id=0;
        const auto result=std::from_chars(field.data(),field.data()+field.size(),id);
        require(result.ec==std::errc{}&&result.ptr==field.data()+field.size()&&id<129280&&out.size()<limit,
                "invalid/oversized session token file");
        out.push_back(id);
    }
    require(in.eof()&&!in.bad(),"invalid token file (whitespace separated unsigned ids)");
    return out;
}
struct Command {
    std::string id,op,durability="ram",path,tokens,history,dump,control_prefix;std::uint32_t steps=0,fixed=0;bool dspark=false,used=false;
};
// Implicit continuation has exactly one source: this worker's resident RAM.
// Full histories may start afresh; suffix-only requests cannot recreate state.
inline void prepare_ram_command(Command& c,const Lru& ram,const Names& names,bool automatic=false) {
    if((c.op=="turn"||c.op=="resume")&&!ram.contains(c.id)) {
        if(automatic||!c.history.empty())c.op="create";
        else names.missing(c.id);
    }
}
inline bool history_prefix(std::span<const std::uint32_t> saved,std::span<const std::uint32_t> full) {
    return saved.size()<=full.size()&&std::equal(saved.begin(),saved.end(),full.begin());
}
// Only ordinary prompt/observer fields accompany a name-only history request.
// Numerical overrides cannot reuse an identity built from the launch settings.
inline std::string automatic_name(std::vector<std::string>& args) {
    std::string id;std::vector<std::string> rest;
    for(std::size_t i=0;i<args.size();++i) {
        const auto& a=args[i];
        if(a=="--session") {require(id.empty()&&i+1<args.size(),"duplicate/missing session name");id=args[++i];valid_id(id);continue;}
        rest.push_back(a);
        if(a=="--qual"||a=="--chat-logits")continue;
        require(a=="--phases"||a=="--seqs-file"||a=="--chats"||a=="--chat-prompt"||a=="--chat-steps"||
                a=="--qual-dump"||a=="--draft-trace"||a=="--json"||a=="--timeline","unsupported named prompt option");
        require(i+1<args.size(),"missing named prompt option value");rest.push_back(args[++i]);
    }
    args=std::move(rest);return id;
}
inline Command parse(std::vector<std::string>& args) {
    Command c;std::vector<std::string> rest;std::map<std::string,bool> seen;
    for(std::size_t i=0;i<args.size();++i) {
        auto a=args[i];if(a.rfind("--session",0)!=0){rest.push_back(a);continue;}
        require(!seen[a],"duplicate session field");seen[a]=true;c.used=true;
        require(i+1<args.size(),"missing session field value");auto v=args[++i];
        auto number=[&]{require(!v.empty()&&v.find_first_not_of("0123456789")==std::string::npos,"invalid session integer");
            auto n=std::stoull(v);require(n<=0xffffffffULL,"session integer overflow");return static_cast<std::uint32_t>(n);};
        if(a=="--session")c.id=v;else if(a=="--session-op")c.op=v;else if(a=="--session-durability")c.durability=v;
        else if(a=="--session-path")c.path=v;else if(a=="--session-tokens")c.tokens=v;else if(a=="--session-dump")c.dump=v;
        else if(a=="--session-history")c.history=v;
        else if(a=="--session-control-prefix")c.control_prefix=v;
        else if(a=="--session-steps")c.steps=number();else if(a=="--session-fixed")c.fixed=number();
        else if(a=="--session-dspark"){require(v=="on"||v=="off","session dspark on|off");c.dspark=v=="on";}
        else throw std::runtime_error("unknown session option "+a);
    }
    if(!c.used)return c;
    require(rest.empty(),"session requests cannot override launch runtime/numerical options");
    if(c.op.empty())c.op="turn";
    if(c.op=="resume"&&(!c.path.empty()||c.durability=="disk"))c.op="restore";
    require(c.op=="create"||c.op=="turn"||c.op=="checkpoint"||c.op=="park"||c.op=="resume"||c.op=="restore"||c.op=="list"||c.op=="stats"||c.op=="delete","unknown session operation");
    if(c.op!="list"&&c.op!="stats")valid_id(c.id);
    require(c.durability=="ram"||c.durability=="disk","session durability ram|disk");
    require(c.path.empty()||c.op=="restore"||c.op=="checkpoint","session path requires restore/resume or checkpoint");
    require(c.op!="restore"||!c.path.empty()||c.durability=="disk","managed restore requires explicit durability disk");
    require(c.fixed<=5&&(!c.dspark?c.fixed==0:c.fixed>0),"DSpark sessions require fixed 1..5 (adaptive unsupported)");
    require((c.op=="turn"||c.op=="create")||(c.tokens.empty()&&c.steps==0&&c.dump.empty()&&!c.dspark&&c.fixed==0),"turn fields on control operation");
    require(c.history.empty()||((c.op=="turn"||c.op=="create")&&c.tokens.empty()),"history requires turn/create and excludes suffix tokens");
    require(c.control_prefix.empty()||c.op=="create","control prefix requires create");
    args=std::move(rest);return c;
}
} // namespace strix::ds41::session
