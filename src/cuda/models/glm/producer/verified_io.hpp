#pragma once
// Windows cache ownership: verify the held file, then keep denying writes and
// deletion for its entire use. Persistent records are authenticated by a
// current-user DPAPI protected authority (same-user execution is out of scope).
// Threat boundary of the held handles and name bindings (unit_auth.hpp): same-user
// accidental or hostile renames, junction swaps and replacements on local NTFS are
// in scope; DOS device remapping (DefineDosDevice, per-session drive letters) and
// privileged attackers (administrator, backup/restore privilege, kernel, raw
// volume writes) are out of scope. The tierstream library opens the unit file
// through its own held ancestor handles and checks the expected identity we pass;
// our held ancestor/name binding stays as defence in depth.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>
#include "strix/model_content_hash.hpp"
#include "hash_authority.hpp"
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace strix::glm::rtxpf::producer {
inline constexpr std::size_t io_alignment=65536,io_chunk=8ULL<<20;
inline std::size_t io_round(std::size_t n){return (n+io_alignment-1)&~(io_alignment-1);}
inline void io_need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct WinHandle {
    HANDLE h=INVALID_HANDLE_VALUE;
    WinHandle()=default;WinHandle(const WinHandle&)=delete;
    ~WinHandle(){if(h!=INVALID_HANDLE_VALUE&&h)CloseHandle(h);}
};
struct AlignedMemory {
    void* p;
    explicit AlignedMemory(std::size_t n):p(VirtualAlloc(nullptr,io_round(n),MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE)){io_need(p!=nullptr,"aligned IO allocation");}
    ~AlignedMemory(){VirtualFree(p,0,MEM_RELEASE);}
};
inline std::filesystem::path default_hash_cache(){
    std::wstring local(32768,L'\0');auto n=GetEnvironmentVariableW(L"LOCALAPPDATA",local.data(),DWORD(local.size()));
    if(!n||n>=local.size())return {};local.resize(n);
    return std::filesystem::path(local)/"Strix"/"glm-verified-hashes-v2";
}
// Absence of a readable, active NTFS journal disables only the accelerator.
// Query the file's USN on the same held deny-write/delete handle used for bytes.
#ifdef STRIX_HASH_AUTHORITY_TESTING
inline bool journal_unavailable=false;
inline std::optional<std::string> journal_test_identity;
#endif
inline std::optional<std::string> file_journal_identity(HANDLE file,DWORD* failure=nullptr){
    if(failure)*failure=ERROR_SUCCESS;
    auto unavailable=[&](DWORD error)->std::optional<std::string>{if(failure)*failure=error;return {};};
#ifdef STRIX_HASH_AUTHORITY_TESTING
    if(journal_unavailable)return unavailable(ERROR_JOURNAL_NOT_ACTIVE);
    if(journal_test_identity)return journal_test_identity;
#endif
    wchar_t fs[32]{};
    if(!GetVolumeInformationByHandleW(file,nullptr,0,nullptr,nullptr,nullptr,fs,32))return unavailable(GetLastError());
    if(std::wcscmp(fs,L"NTFS"))return unavailable(ERROR_INVALID_FUNCTION);
    // Query the journal through the held file handle: a volume handle needs
    // administrator rights (ERROR_INVALID_FUNCTION for an unprivileged open),
    // while the same FSCTL on a file handle reports that file's volume journal.
    WinHandle event;event.h=CreateEventW(nullptr,TRUE,FALSE,nullptr);if(!event.h)return unavailable(GetLastError());
    USN_JOURNAL_DATA_V0 journal{};DWORD got=0;
    OVERLAPPED jov{};jov.hEvent=event.h;
    auto queried=DeviceIoControl(file,FSCTL_QUERY_USN_JOURNAL,nullptr,0,&journal,sizeof journal,&got,&jov);
    if(!queried&&(GetLastError()!=ERROR_IO_PENDING||!GetOverlappedResult(file,&jov,&got,TRUE)))return unavailable(GetLastError());
    if(got<sizeof journal)return unavailable(ERROR_INVALID_DATA);
    ResetEvent(event.h);
    OVERLAPPED ov{};ov.hEvent=event.h;
    alignas(8) std::array<std::byte,4096> buffer{};
    READ_FILE_USN_DATA request{2,3};
    auto ok=DeviceIoControl(file,FSCTL_READ_FILE_USN_DATA,&request,sizeof request,buffer.data(),DWORD(buffer.size()),&got,&ov);
    if(!ok&&(GetLastError()!=ERROR_IO_PENDING||!GetOverlappedResult(file,&ov,&got,TRUE)))return unavailable(GetLastError());
    if(got<8)return unavailable(ERROR_INVALID_DATA);auto* v2=reinterpret_cast<const USN_RECORD_V2*>(buffer.data());USN usn=0;
    if(v2->RecordLength>got)return unavailable(ERROR_INVALID_DATA);
    // Only the fixed fields through Usn are required; a nameless record need
    // not include the C struct's FileName[1] or its trailing alignment padding.
    if(v2->MajorVersion==2&&v2->RecordLength>=offsetof(USN_RECORD_V2,Usn)+sizeof(USN))usn=v2->Usn;
    else if(v2->MajorVersion==3&&v2->RecordLength>=offsetof(USN_RECORD_V3,Usn)+sizeof(USN))usn=reinterpret_cast<const USN_RECORD_V3*>(buffer.data())->Usn;
    else return unavailable(ERROR_INVALID_DATA);
    if(usn<=0)return unavailable(ERROR_JOURNAL_NOT_ACTIVE);
    return std::to_string(journal.UsnJournalID)+" "+std::to_string(usn);
}
// Persistent record policy. With an active USN journal the record key carries the
// journal id and the file's USN (any write moves it). Without a journal a record
// is used only when the caller opts in (no_usn_record): the key is then the
// engine's Windows file-hash stamp v3 (src/model_content_hash.cpp hash_cache_key:
// volume serial, file index, size, last-write and
// creation times; ChangeTime excluded because metadata-only hard-link operations
// move it), read on our held handle, plus the caller's pin (the unit manifest
// digest), under the same DPAPI-keyed MAC (the engine cache has no MAC). Weaker: a same-user rewrite that restores every timestamp between runs is
// not detected (while held, the deny-write handle excludes writes).
struct RecordOptions {bool no_usn_record=false;std::string pin;};
class VerifiedFile {
    WinHandle file_;
    std::uint64_t bytes_;
#ifdef STRIX_HASH_AUTHORITY_TESTING
    mutable std::atomic<std::uint64_t> bytes_read_=0;
#endif
    std::string key()const {
        FILE_ID_INFO id{};FILE_BASIC_INFO basic{};FILE_STANDARD_INFO size{};
        io_need(GetFileInformationByHandleEx(file_.h,FileIdInfo,&id,sizeof id)&&
                GetFileInformationByHandleEx(file_.h,FileBasicInfo,&basic,sizeof basic)&&
                GetFileInformationByHandleEx(file_.h,FileStandardInfo,&size,sizeof size),"file identity metadata");
        io_need(!size.Directory&&std::uint64_t(size.EndOfFile.QuadPart)==bytes_,"cache file size mismatch");
        std::ostringstream s;s<<"glm-verified-v3 "<<id.VolumeSerialNumber<<' ';
        for(auto b:id.FileId.Identifier)s<<unsigned(b)<<',';
        s<<' '<<bytes_<<' '<<basic.LastWriteTime.QuadPart<<' '<<basic.CreationTime.QuadPart<<' '<<basic.ChangeTime.QuadPart;
        return s.str();
    }
    // The engine's Windows hash-cache stamp v3 ("<volume>-<file index>" + "v3 size
    // last-write creation"), from the held handle; ChangeTime is not part of it.
    std::string stamp_v3()const {
        BY_HANDLE_FILE_INFORMATION info{};io_need(GetFileInformationByHandle(file_.h,&info)!=0&&!(info.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY),"file stamp v3");
        const auto u64=[](DWORD high,DWORD low){return (std::uint64_t(high)<<32U)|low;};
        return std::to_string(info.dwVolumeSerialNumber)+"-"+std::to_string(u64(info.nFileIndexHigh,info.nFileIndexLow))+" v3 "+std::to_string(u64(info.nFileSizeHigh,info.nFileSizeLow))+" "+
               std::to_string(u64(info.ftLastWriteTime.dwHighDateTime,info.ftLastWriteTime.dwLowDateTime))+" "+std::to_string(u64(info.ftCreationTime.dwHighDateTime,info.ftCreationTime.dwLowDateTime));
    }
    // Two >=8 MiB, sector-aligned requests in flight (except the final request).
    // Each slot is drained before reuse. On exceptions, cancel and drain before
    // the caller can release pinned memory or hashing scratch.
    template<class Address,class Consume>
    void pipeline(std::uint64_t begin,std::uint64_t length,Address address,Consume consume)const {
        io_need(begin%io_alignment==0&&begin<=bytes_&&length<=bytes_-begin,"IO range");
        struct Slot {WinHandle event;OVERLAPPED ov{};std::uint64_t offset=0;DWORD wanted=0;bool live=false;} slots[2];
        for(auto& s:slots){s.event.h=CreateEventW(nullptr,TRUE,FALSE,nullptr);io_need(s.event.h!=nullptr,"IO event");s.ov.hEvent=s.event.h;}
        auto submit=[&](Slot& s,std::uint64_t offset,unsigned parity){
            s.offset=offset;s.wanted=DWORD(std::min<std::uint64_t>(io_chunk,begin+length-offset));
            s.ov.Offset=DWORD(offset);s.ov.OffsetHigh=DWORD(offset>>32);ResetEvent(s.event.h);
            void* dst=address(offset-begin,parity);io_need(reinterpret_cast<std::uintptr_t>(dst)%io_alignment==0,"unaligned IO destination");
            s.live=true;BOOL ok=ReadFile(file_.h,dst,DWORD(io_round(s.wanted)),nullptr,&s.ov);
            if(!ok&&GetLastError()!=ERROR_IO_PENDING){s.live=false;throw std::runtime_error("overlapped cache read");}
        };
        std::uint64_t next=begin,done=0;unsigned turn=0;
        try{
            for(unsigned i=0;i<2&&next<begin+length;++i){submit(slots[i],next,i);next+=slots[i].wanted;}
            while(done<length){auto& s=slots[turn];DWORD got=0;io_need(GetOverlappedResult(file_.h,&s.ov,&got,TRUE)!=0,"cache read completion");s.live=false;
                // A bounded startup read can end before EOF and read aligned
                // padding; the consumer always sees exactly the requested bytes.
                io_need(got>=s.wanted,"short cache read");consume(address(s.offset-begin,turn),s.wanted);done+=s.wanted;
#ifdef STRIX_HASH_AUTHORITY_TESTING
                bytes_read_+=got;
#endif
                if(next<begin+length){submit(s,next,turn);next+=s.wanted;}turn^=1;
            }
        }catch(...){for(auto& s:slots)if(s.live){CancelIoEx(file_.h,&s.ov);DWORD got;GetOverlappedResult(file_.h,&s.ov,&got,TRUE);}throw;}
    }
public:
    // Independent positional cursor over the authenticated object. Small and
    // unaligned shard ranges use bounded aligned scratch; large windows still
    // use the unbuffered OVERLAPPED path. No pathname or shared file seek.
    class Reader {
        const VerifiedFile& file_;
        std::size_t window_;
        AlignedMemory memory_;
        std::uint64_t begin_=0;
        std::size_t size_=0;
    public:
        explicit Reader(const VerifiedFile& file,std::size_t window=io_alignment):file_(file),window_(window),memory_(std::max<std::uint64_t>(1,std::min<std::uint64_t>(window,file.bytes_))){
            io_need(window&&window%io_alignment==0&&window<=2*io_chunk,"source IO window");
        }
        void read_at(std::uint64_t offset,void* destination,std::size_t n){
            io_need(offset<=file_.bytes_&&n<=file_.bytes_-offset,"source IO range");
            auto* out=static_cast<char*>(destination);
            while(n){
                if(!size_||offset<begin_||offset-begin_>=size_){
                    auto begin=offset/window_*window_;auto size=std::size_t(std::min<std::uint64_t>(window_,file_.bytes_-begin));size_=0;
                    file_.pipeline(begin,size,[&](std::uint64_t off,unsigned){return static_cast<char*>(memory_.p)+off;},[](void*,DWORD){});
                    begin_=begin;size_=size;
                }
                auto skip=std::size_t(offset-begin_),take=std::min(n,size_-skip);
                std::memcpy(out,static_cast<const char*>(memory_.p)+skip,take);out+=take;offset+=take;n-=take;
            }
        }
    };
    std::string contents()const {
        std::string result(std::size_t(bytes_),'\0');Reader reader(*this);reader.read_at(0,result.data(),result.size());return result;
    }
#ifdef STRIX_HASH_AUTHORITY_TESTING
    std::uint64_t bytes_read()const {return bytes_read_;}
#endif
    bool journal_available()const {return file_journal_identity(file_.h).has_value();}
    // The held deny-write/delete handle (identity checks and final-path binding only).
    HANDLE held_handle()const {return file_.h;}
    VerifiedFile(const std::filesystem::path& path,std::uint64_t bytes):bytes_(bytes){
        file_.h=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,
                          FILE_FLAG_NO_BUFFERING|FILE_FLAG_OVERLAPPED|FILE_FLAG_SEQUENTIAL_SCAN,nullptr);
        io_need(file_.h!=INVALID_HANDLE_VALUE,"open cache (deny write/delete)");key();
        FILE_STORAGE_INFO storage{};io_need(GetFileInformationByHandleEx(file_.h,FileStorageInfo,&storage,sizeof storage)!=0,"cache sector alignment");
        io_need(storage.PhysicalBytesPerSectorForPerformance&&io_alignment%storage.PhysicalBytesPerSectorForPerformance==0&&
                storage.LogicalBytesPerSector&&io_alignment%storage.LogicalBytesPerSector==0,"unsupported cache sector alignment");
    }
    void read_into(void* destination)const {
        pipeline(0,bytes_,[&](std::uint64_t off,unsigned){return static_cast<char*>(destination)+off;},[](void*,DWORD){});
    }
    // Arbitrary aligned span (decode expert reads): 64 KiB-aligned begin and
    // destination; destination capacity io_round(length). Two requests in flight.
    void read_span(std::uint64_t begin,std::uint64_t length,void* destination)const {
        io_need(reinterpret_cast<std::uintptr_t>(destination)%io_alignment==0,"unaligned span destination");
        pipeline(begin,length,[&](std::uint64_t off,unsigned){return static_cast<char*>(destination)+off;},[](void*,DWORD){});
    }
    void read_at(std::uint64_t offset,void* destination,std::size_t n)const {
        io_need(n<=io_chunk,"startup IO chunk");pipeline(offset,n,[&](std::uint64_t,unsigned){return destination;},[](void*,DWORD){});
    }
    // Manifest consumers parse exactly this buffer: never reopen its pathname.
    // Always hash these bytes, even if a blob's persistent accelerator is warm.
    std::string verified_contents(const std::string& expected)const {
        io_need(bytes_<=16ULL*1024*1024,"manifest exceeds 16 MiB bound");auto before=key();
        AlignedMemory memory(std::max<std::uint64_t>(1,bytes_));read_into(memory.p);
        std::string bytes(static_cast<const char*>(memory.p),std::size_t(bytes_));
        io_need(strix::model_content_identity_hex(strix::sha256_bytes_fast(std::as_bytes(std::span(bytes))))==expected,"manifest SHA-256 mismatch");
        io_need(key()==before,"manifest metadata changed during verification");return bytes;
    }
    struct Digest {std::string sha256;bool recalled;std::string basis;}; // basis: "usn record", "no-usn record (mtime-stamped)", "hashed"
private:
    // Both callers share the same held-handle identity and authenticated record
    // authority. An expected digest is checked BEFORE publishing a cold result.
    Digest resolve_digest(const std::string* expected,const std::filesystem::path& cache_dir,const RecordOptions& o)const {
        auto before=key();auto journal=file_journal_identity(file_.h);
        const bool stamped=!journal&&o.no_usn_record&&!o.pin.empty();
        bool available=false;HashKey authority{};
        if((journal||stamped)&&!cache_dir.empty())try{authority=hash_authority(cache_dir,&available);}catch(const std::exception&){available=false;}
        const auto canonical=journal?before+" "+*journal:stamp_v3()+" no-usn pin="+o.pin;
        const char* recalled_basis=journal?"usn record":"no-usn record (mtime-stamped)";
        std::filesystem::path record;
        if(available){
            auto name=strix::model_content_identity_hex(strix::sha256_bytes_fast(std::as_bytes(std::span(canonical))));
            record=cache_dir/(name+".sha256");
            std::ifstream in(record,std::ios::binary);std::string saved,digest,saved_mac,extra;
            if(std::getline(in,saved)&&std::getline(in,digest)&&std::getline(in,saved_mac)&&!std::getline(in,extra)&&
               saved==canonical&&digest.size()==64&&
               digest.find_first_not_of("0123456789abcdef")==std::string::npos&&
               (!expected||digest==*expected)&&
               equal_record_mac(saved_mac,hash_record_mac(authority,canonical+'\n'+digest+'\n'))&&
               key()==before&&file_journal_identity(file_.h)==journal)return {digest,true,recalled_basis};}
        AlignedMemory memory(2*io_chunk);strix::Sha256Fast sha;
        pipeline(0,bytes_,[&](std::uint64_t,unsigned slot){return static_cast<char*>(memory.p)+slot*io_chunk;},
                 [&](void* p,DWORD n){sha.update(std::span(static_cast<const std::byte*>(p),n));});
        auto digest=strix::model_content_identity_hex(sha.finalize());
        io_need(!expected||digest==*expected,"cache SHA-256 mismatch");
        io_need(key()==before,"cache metadata changed during verification");
        if(!available||file_journal_identity(file_.h)!=journal)return {digest,false,"hashed"};
        auto mac=hash_record_mac(authority,canonical+'\n'+digest+'\n');
        std::error_code ec;
        auto temporary=record;temporary+=L"."+std::to_wstring(GetCurrentProcessId())+L"."+std::to_wstring(GetCurrentThreadId())+L".partial";
        {std::ofstream out(temporary,std::ios::binary|std::ios::trunc);out<<canonical<<'\n'<<digest<<'\n'<<mac<<'\n';out.close();if(!out){std::filesystem::remove(temporary,ec);return {digest,false,"hashed"};}}
        if(!MoveFileExW(temporary.c_str(),record.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))std::filesystem::remove(temporary,ec);
        return {digest,false,"hashed"};
    }
public:
    // Unknown source digest: only an authenticated matching record can supply
    // it. Otherwise hash the complete held file and publish the COMPUTED digest.
    Digest compute_or_recall(const std::filesystem::path& cache_dir)const {
        return resolve_digest(nullptr,cache_dir,{});
    }
    // Known cache digest: retain mismatch rejection and persistent-hit result.
    bool verify(const std::string& expected,const std::filesystem::path& cache_dir)const {
        return resolve_digest(&expected,cache_dir,{}).recalled;
    }
    // Known digest with an explicit record policy (unit file: no-USN record allowed).
    Digest verify_digest(const std::string& expected,const std::filesystem::path& cache_dir,const RecordOptions& o)const {
        return resolve_digest(&expected,cache_dir,o);
    }
};
// One worker per logical core; files are independent whole-file SHA chains.
// Worker failures are joined before propagating, preserving all handle lives.
template<class Job> void parallel_hash_files(std::size_t count,Job job){
    std::atomic<std::size_t> next=0;std::mutex mutex;std::exception_ptr error;
    std::vector<std::thread> workers;
    auto join=[&]{for(auto& w:workers)if(w.joinable())w.join();};
    try{for(std::size_t i=0;i<std::min<std::size_t>(count,std::max(1u,std::thread::hardware_concurrency()));++i)workers.emplace_back([&]{
        try{for(;;){auto j=next.fetch_add(1);if(j>=count)break;job(j);}}
        catch(...){std::lock_guard lock(mutex);if(!error)error=std::current_exception();next.store(count);}
    });}catch(...){join();throw;}
    join();if(error)std::rethrow_exception(error);
}
}
