#pragma once
// Startup authentication of the GLM unit file (tierstream disk tier), shared by
// the runtime and its CPU tests. Order: the deny-write/delete handle first (its
// expected size is checked on that handle); the manifest read once through its
// own held handle against the operator pin (it binds model identity, geometry,
// rates and the whole-file SHA-256); then the contents through the held handle:
// an authenticated verification record (verified_io.hpp authority) is recalled,
// otherwise the whole file is hashed. Mismatches reject startup.
#include "verified_io.hpp"
#include "unit_file.hpp"
#include <chrono>
#include <cstring>
#include <vector>
#include <memory>
namespace strix::glm::rtxpf::producer {
struct UnitAuth {
    std::unique_ptr<VerifiedFile> lock;std::string file_sha256,manifest_sha256,basis="hashed";bool recalled=false;double ms=0;
    const char* how()const{return !recalled?"hashed through the held handle":basis=="usn record"?"authenticated record recalled (USN)":"authenticated record recalled (no USN: mtime-stamped record)";}
};
// ---- Identity-bound opening contract for the cache's own reader ----
// tierstream opens its source by pathname (it takes no handle). The
// name it receives is therefore bound to the authenticated object: the held
// handle's final path (no aliases/junctions left), every ancestor directory held
// with FILE_LIST_DIRECTORY sharing only reads (no rename/delete/write/reparse
// installation while held; reparse ancestors refused), a reparse-refusing name
// handle on the file with the authenticated FileId, and a post-open serving
// handle through that path compared to the authenticated FileId. All handles
// live as long as the cache.
inline void same_file_object(HANDLE a,HANDLE b){
    FILE_ID_INFO x{},y{};
    io_need(GetFileInformationByHandleEx(a,FileIdInfo,&x,sizeof x)&&GetFileInformationByHandleEx(b,FileIdInfo,&y,sizeof y),"unit file FileId query");
    io_need(x.VolumeSerialNumber==y.VolumeSerialNumber&&!std::memcmp(&x.FileId,&y.FileId,sizeof x.FileId),"unit file identity mismatch (bound path no longer names the authenticated file)");
}
inline std::filesystem::path held_final_path(HANDLE h){
    std::wstring s(32768,L'\0');auto n=GetFinalPathNameByHandleW(h,s.data(),DWORD(s.size()),FILE_NAME_NORMALIZED|VOLUME_NAME_DOS);
    io_need(n&&n<s.size(),"held unit file final path");s.resize(n);
    io_need(s.size()>=7&&s.compare(0,4,L"\\\\?\\")==0&&s[5]==L':'&&s[6]==L'\\',"unit file must be on a local drive path");
    // No alternate data stream (or any other ':' after the drive designator).
    io_need(s.find(L':',6)==std::wstring::npos,"unit file alternate data stream refused");
    return s.substr(4);
}
class BoundUnitFile {
    std::vector<std::unique_ptr<WinHandle>> ancestors_;std::unique_ptr<WinHandle> name_,serving_;
public:
    std::filesystem::path path;
    explicit BoundUnitFile(const VerifiedFile& held):path(held_final_path(held.held_handle())){
        for(auto p=path.parent_path();!p.empty();){
            auto h=std::make_unique<WinHandle>();
            // FILE_LIST_DIRECTORY participates in sharing (attribute-only handles do
            // not exclude directory writers); FILE_SHARE_READ denies write/delete/rename.
            h->h=CreateFileW(p.c_str(),FILE_LIST_DIRECTORY|FILE_READ_ATTRIBUTES,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
            io_need(h->h!=INVALID_HANDLE_VALUE,"anchor unit file ancestor directory");FILE_ATTRIBUTE_TAG_INFO info{};
            io_need(GetFileInformationByHandleEx(h->h,FileAttributeTagInfo,&info,sizeof info),"unit file ancestor attributes");
            io_need(!(info.FileAttributes&FILE_ATTRIBUTE_REPARSE_POINT),"unit file ancestor reparse point refused");
            ancestors_.push_back(std::move(h));auto next=p.parent_path();if(next==p)break;p=next;
        }
        name_=std::make_unique<WinHandle>();name_->h=CreateFileW(path.c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        io_need(name_->h!=INVALID_HANDLE_VALUE,"retain unit file name");FILE_ATTRIBUTE_TAG_INFO info{};
        io_need(GetFileInformationByHandleEx(name_->h,FileAttributeTagInfo,&info,sizeof info)&&!(info.FileAttributes&FILE_ATTRIBUTE_REPARSE_POINT),"unit file reparse point refused");
        same_file_object(held.held_handle(),name_->h);
    }
    std::size_t ancestors()const{return ancestors_.size();}
    // The exact bytes the cache receives: lossless UTF-8 of the native bound path
    // (backslashes kept); tierstream decodes source paths as UTF-8, never ANSI.
    std::string utf8()const{
        const auto& w=path.native();io_need(!w.empty()&&w.size()<0x7fffffff,"bound path length");
        int n=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,w.data(),int(w.size()),nullptr,0,nullptr,nullptr);io_need(n>0,"bound path is not valid UTF-16");
        std::string out(std::size_t(n),'\0');io_need(WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,w.data(),int(w.size()),out.data(),n,nullptr,nullptr)==n,"bound path UTF-8 encoding");
        int m=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,out.data(),n,nullptr,0);std::wstring back(std::size_t(m>0?m:0),L'\0');
        io_need(m>0&&MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,out.data(),n,back.data(),m)==m&&back==w,"bound path does not round-trip through UTF-8");return out;
    }
    // After the cache opened its source through `path`: a serving handle opened the
    // same way (read, share read) must still name the authenticated object.
    void verify_serving(const VerifiedFile& held){
        serving_=std::make_unique<WinHandle>();serving_->h=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        io_need(serving_->h!=INVALID_HANDLE_VALUE,"serving unit file handle");same_file_object(held.held_handle(),serving_->h);
    }
    std::string describe()const{return "canonical held path "+path.string()+"; "+std::to_string(ancestors_.size())+" ancestor directories held (FILE_LIST_DIRECTORY, share read: no write/rename/delete); reparse refused; name and serving FileId == authenticated FileId; passed to the cache as lossless UTF-8. "
        "The library (tierstream) opens it through held ancestor handles and requires the authenticated identity "
        "(source_identities: volume serial, file index, size); this binding stays as defence in depth";}
};
// The authenticated handles and the bound name, owned together. The cache that
// consumes the name is owned AFTER it (BoundCache), so it always closes first.
struct UnitBinding {std::unique_ptr<VerifiedFile> lock;std::unique_ptr<BoundUnitFile> bound;};
// The operator's unit path names a file's default data stream: no ':' after the
// root name (a handle's final path may omit a stream name, so this is checked on
// the requested path too).
inline void refuse_stream_path(const std::filesystem::path& unit){
    std::wstring s=unit.native();if(s.rfind(L"\\\\?\\",0)==0||s.rfind(L"\\??\\",0)==0)s.erase(0,4);
    const auto colon=s.find(L':');
    io_need(colon==std::wstring::npos||(colon==1&&s.find(L':',2)==std::wstring::npos),"unit file alternate data stream refused");
}
inline UnitAuth authenticate_unit_file(const std::filesystem::path& unit,const ExpertCatalog& c,const std::string& model_sha256,const std::string& manifest_pin,
                                       const std::filesystem::path& hash_cache=default_hash_cache()){
    expert_need(sha256_hex(manifest_pin),"--unit-manifest-sha256 (pinned unit manifest digest) is required");refuse_stream_path(unit);
    UnitAuth a;a.lock=std::make_unique<VerifiedFile>(unit,c.size()*c.unit_stride());
    const auto mpath=unit_manifest_path(unit);VerifiedFile held_manifest(mpath,std::filesystem::file_size(mpath));
    auto manifest=strix::serve::parse_json(held_manifest.verified_contents(manifest_pin));validate_unit_manifest(manifest,c,model_sha256);
    a.file_sha256=manifest.find("file_sha256")->as_string();a.manifest_sha256=manifest_pin;
    const auto begin=std::chrono::steady_clock::now();
    // No USN journal: an mtime-stamped record bound to the manifest pin (weaker; verified_io.hpp RecordOptions).
    const auto d=a.lock->verify_digest(a.file_sha256,hash_cache,RecordOptions{true,manifest_pin});a.recalled=d.recalled;a.basis=d.basis;
    a.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();return a;
}
}
