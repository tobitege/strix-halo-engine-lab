#pragma once
// GLM expert unit file (tierstream disk tier): 43 x 288 units in numeric order,
// one 64 KiB-strided opaque unit each (expert_source.hpp), plus a unit manifest
// (per-unit SHA-256) and the tierstream desc-v1 descriptor. Optional secondary
// stripes use tierstream's own tool (ts-stripe make with a stripe-ranges-v1
// "blocks:K" range); this file only plans their exact chunks and bytes.
// Nothing here runs implicitly: plan/dry-run never write; copy refuses an
// existing target and requires the exact planned byte count.
#include "expert_source.hpp"
#include "expert_tier.hpp"
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <functional>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
namespace strix::glm::rtxpf::producer {
inline constexpr std::uint64_t stripe_chunk_default=16ULL<<20;
struct StripeTarget {std::filesystem::path dir;unsigned run=64;bool even=false;std::uint64_t chunks=0,bytes=0;};
struct UnitFilePlan {
    std::filesystem::path primary,manifest,descriptor;std::uint64_t units=0,stride=0,bytes=0,chunk=stripe_chunk_default;std::vector<StripeTarget> stripes;
};
inline std::filesystem::path unit_manifest_path(const std::filesystem::path& p){auto m=p;m+=".units.json";return m;}
inline std::filesystem::path unit_descriptor_path(const std::filesystem::path& p){auto m=p;m+=".desc.json";return m;}
// Exactly ts-stripe's placement: chunk c is placed when (c/run)%2 == parity (blocks:K -> 1, blocks-even:K -> 0).
inline std::pair<std::uint64_t,std::uint64_t> stripe_extent(std::uint64_t size,std::uint64_t chunk,unsigned run,bool even){
    expert_need(chunk&&run>=1&&run<=999999,"stripe chunk/run");std::uint64_t count=(size+chunk-1)/chunk,chunks=0,bytes=0;
    for(std::uint64_t c=0;c<count;++c)if((c/run)%2==(even?0u:1u)){++chunks;bytes+=std::min(chunk,size-c*chunk);}
    return {chunks,bytes};
}
// "DIR:K" or "DIR:even:K" (K chunks per run).
inline StripeTarget parse_stripe(const std::string& s){
    auto colon=s.rfind(':');expert_need(colon!=std::string::npos&&colon+1<s.size(),"stripe target must be DIR:K or DIR:even:K");
    auto k=s.substr(colon+1);expert_need(k.size()<=6&&k.find_first_not_of("0123456789")==std::string::npos,"stripe run K must be 1..999999");
    StripeTarget t;t.run=unsigned(std::stoul(k));expert_need(t.run>=1,"stripe run K must be 1..999999");auto dir=s.substr(0,colon);
    if(dir.size()>5&&dir.substr(dir.size()-5)==":even"){t.even=true;dir=dir.substr(0,dir.size()-5);}
    expert_need(!dir.empty()&&!(dir.size()==1&&std::isalpha(static_cast<unsigned char>(dir[0]))),"stripe directory");t.dir=dir;return t;
}
inline UnitFilePlan plan_unit_file(const ExpertCatalog& c,const std::filesystem::path& primary,const std::vector<StripeTarget>& stripes,std::uint64_t chunk=stripe_chunk_default){
    UnitFilePlan p;p.primary=primary;p.manifest=unit_manifest_path(primary);p.descriptor=unit_descriptor_path(primary);p.units=c.size();p.stride=c.unit_stride();p.chunk=chunk;
    expert_need(p.stride%4096==0&&p.stride%unit_alignment==0,"unit offsets must stay 4 KiB (64 KiB) aligned");p.bytes=p.units*p.stride;
    for(auto t:stripes){auto [n,b]=stripe_extent(p.bytes,chunk,t.run,t.even);t.chunks=n;t.bytes=b;p.stripes.push_back(t);}return p;
}
inline Json stripe_ranges(const UnitFilePlan& p,const StripeTarget& t){
    auto j=Json::object();j.set("schema","tierstream/stripe-ranges-v1");j.set("primary",p.primary.filename().string());j.set("size",p.bytes);
    auto r=Json::object();r.set("name","glm-expert-units");r.set("offset",std::uint64_t(0));r.set("bytes",p.bytes);r.set("policy",std::string(t.even?"blocks-even:":"blocks:")+std::to_string(t.run));
    auto a=Json::array();a.push(r);j.set("ranges",a);return j;
}
inline Json plan_json(const UnitFilePlan& p,const std::string& model_sha256){
    auto j=Json::object();j.set("format",expert_unit_format);j.set("model_sha256",model_sha256);j.set("primary",p.primary.string());j.set("manifest",p.manifest.string());j.set("descriptor",p.descriptor.string());
    j.set("units",p.units);j.set("unit_stride",p.stride);j.set("primary_bytes",p.bytes);j.set("chunk_bytes",p.chunk);auto s=Json::array();
    for(auto& t:p.stripes){auto e=Json::object();e.set("dir",t.dir.string());e.set("policy",std::string(t.even?"blocks-even:":"blocks:")+std::to_string(t.run));e.set("chunks",t.chunks);e.set("bytes",t.bytes);
        auto ranges=(t.dir/(p.primary.filename().string()+".ranges.json")).string();e.set("ranges_file",ranges);
        e.set("command","ts-stripe make \""+p.primary.string()+"\" \""+ranges+"\" \""+t.dir.string()+"\" --chunk-mib "+std::to_string(p.chunk>>20));s.push(e);}
    j.set("stripes",s);std::uint64_t total=p.bytes;for(auto& t:p.stripes)total+=t.bytes;j.set("total_write_bytes",total);return j;
}
// Paths handed to tierstream are UTF-8 (its decoding), never the ANSI code page.
inline std::string utf8_path(const std::filesystem::path& p){auto s=p.u8string();return std::string(reinterpret_cast<const char*>(s.data()),s.size());}
inline std::filesystem::path partial_path(const std::filesystem::path& p){auto m=p;m+=".partial";return m;}
// Every path a copy creates (finals and their reserved .partial names): dry-run
// readiness and copy both require all of them to be absent.
inline std::vector<std::filesystem::path> unit_file_outputs(const UnitFilePlan& p){
    std::vector<std::filesystem::path> v;for(auto* f:{&p.primary,&p.manifest,&p.descriptor}){v.push_back(*f);v.push_back(partial_path(*f));}return v;
}
// An output reserved by exclusive creation and owned (deny write/delete to
// others) until it is published by a no-replace rename of the same handle.
class ExclusiveFile {
#ifdef _WIN32
    HANDLE h_=INVALID_HANDLE_VALUE;
#else
    std::FILE* f_=nullptr;
#endif
    std::filesystem::path path_;bool published_=false;
public:
    explicit ExclusiveFile(const std::filesystem::path& p):path_(p){
#ifdef _WIN32
        h_=CreateFileW(p.c_str(),GENERIC_READ|GENERIC_WRITE|DELETE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
        expert_need(h_!=INVALID_HANDLE_VALUE,"output already exists or cannot be reserved (exclusive creation)");
#else
        f_=std::fopen(p.c_str(),"w+bx");expert_need(f_!=nullptr,"output already exists or cannot be reserved (exclusive creation)");
#endif
    }
    ExclusiveFile(const ExclusiveFile&)=delete;
    ~ExclusiveFile(){
#ifdef _WIN32
        if(h_!=INVALID_HANDLE_VALUE){if(!published_){FILE_DISPOSITION_INFO d{TRUE};SetFileInformationByHandle(h_,FileDispositionInfo,&d,sizeof d);}CloseHandle(h_);}
#else
        if(f_){std::fclose(f_);if(!published_)std::filesystem::remove(path_);}
#endif
    }
    void write(const void* data,std::size_t n){auto* p=static_cast<const char*>(data);
        while(n){
#ifdef _WIN32
            DWORD k=DWORD(std::min<std::size_t>(n,1u<<30)),w=0;expert_need(WriteFile(h_,p,k,&w,nullptr)&&w==k,"output write");
#else
            std::size_t k=n,w=std::fwrite(p,1,k,f_);expert_need(w==k,"output write");
#endif
            p+=k;n-=k;}}
    void read_at(std::uint64_t offset,void* data,std::size_t n){
#ifdef _WIN32
        LARGE_INTEGER at;at.QuadPart=LONGLONG(offset);expert_need(SetFilePointerEx(h_,at,nullptr,FILE_BEGIN)!=0,"output seek");auto* p=static_cast<char*>(data);
        while(n){DWORD k=DWORD(std::min<std::size_t>(n,1u<<30)),r=0;expert_need(ReadFile(h_,p,k,&r,nullptr)&&r==k,"output read");p+=k;n-=k;}
        LARGE_INTEGER end{};expert_need(SetFilePointerEx(h_,end,nullptr,FILE_END)!=0,"output seek");
#else
        expert_need(std::fseek(f_,long(offset),SEEK_SET)==0&&std::fread(data,1,n,f_)==n&&std::fseek(f_,0,SEEK_END)==0,"output read");
#endif
    }
    void flush(){
#ifdef _WIN32
        expert_need(FlushFileBuffers(h_)!=0,"output flush");
#else
        expert_need(std::fflush(f_)==0,"output flush");
#endif
    }
    // No-replace publication: fails if `final` exists, even if it appeared during the copy.
    void publish(const std::filesystem::path& final){
#ifdef _WIN32
        auto name=final.wstring();std::vector<std::byte> buf(sizeof(FILE_RENAME_INFO)+(name.size()+1)*sizeof(wchar_t));auto* info=reinterpret_cast<FILE_RENAME_INFO*>(buf.data());
        info->ReplaceIfExists=FALSE;info->RootDirectory=nullptr;info->FileNameLength=DWORD(name.size()*sizeof(wchar_t));std::memcpy(info->FileName,name.c_str(),name.size()*sizeof(wchar_t));
        expert_need(SetFileInformationByHandle(h_,FileRenameInfo,info,DWORD(buf.size()))!=0,"publish refused: target exists (no-replace)");
#else
        expert_need(!std::filesystem::exists(final),"publish refused: target exists (no-replace)");std::filesystem::rename(path_,final);
#endif
        path_=final;published_=true;
    }
};
using UnitReader=std::function<void(std::uint64_t offset,void* dst,std::size_t n)>;
// Rereads every unit: layout/padding/markers, each unit digest and the whole-file digest.
inline std::uint64_t verify_units(const ExpertCatalog& c,const UnitReader& read,const Json& manifest,const std::string& model_sha256){
    validate_unit_manifest(manifest,c,model_sha256);std::vector<std::byte> image(c.unit_stride());const auto& units=manifest.find("units")->as_array();std::size_t j=0;const auto& g=c.geometry();strix::Sha256Fast whole;
    for(unsigned l=g.first_layer;l<=g.last_layer;++l)for(unsigned e=0;e<g.experts;++e,++j){read(c.unit_offset(l,e),image.data(),image.size());
        validate_unit_image(c.at(l,e),image,c.unit_stride());expert_need(unit_sha256(image)==units[j].find("sha256")->as_string(),"unit digest mismatch");whole.update(std::span<const std::byte>(image));}
    expert_need(strix::model_content_identity_hex(whole.finalize())==manifest.find("file_sha256")->as_string(),"unit file digest mismatch");return j;
}
inline std::uint64_t verify_unit_file(const ExpertCatalog& c,const std::filesystem::path& primary,const Json& manifest,const std::string& model_sha256){
    expert_need(std::filesystem::file_size(primary)==c.size()*c.unit_stride(),"unit file size");std::ifstream in(primary,std::ios::binary);expert_need(bool(in),"open unit file");
    return verify_units(c,[&](std::uint64_t off,void* d,std::size_t n){in.seekg(std::streamoff(off));expert_need(bool(in.read(static_cast<char*>(d),std::streamsize(n))),"short unit file");},manifest,model_sha256);
}
// Reserves all outputs exclusively (.partial), writes units in numeric order
// (file offset == unit_offset) with the whole-file digest, writes the manifest
// and descriptor, verifies everything through the owned handle, then publishes
// each output with a no-replace rename. Failure before publication deletes the
// reserved partials; no existing file is ever opened for writing.
inline Json write_unit_file(const ExpertCatalog& c,const UnitFilePlan& p,const std::string& model_sha256,const TensorLocator& locate,const std::vector<std::uint64_t>& file_bytes,const AlignedReader& read){
    for(auto& f:unit_file_outputs(p))expert_need(!std::filesystem::exists(f),"unit file output or its .partial exists; never overwritten");
    ExclusiveFile units(partial_path(p.primary)),manifest_out(partial_path(p.manifest)),descriptor_out(partial_path(p.descriptor));
    std::vector<std::string> digests;strix::Sha256Fast whole;std::uint64_t at=0;
    digests=produce_units(c,locate,file_bytes,read,[&](unsigned l,unsigned e,std::span<const std::byte> image){expert_need(c.unit_offset(l,e)==at,"unit file order");
        units.write(image.data(),image.size());whole.update(image);at+=image.size();});
    expert_need(at==p.bytes,"unit file size");units.flush();
    auto manifest=unit_manifest(c,model_sha256,digests,strix::model_content_identity_hex(whole.finalize()));validate_unit_manifest(manifest,c,model_sha256);
    auto mtext=manifest.dump(),dtext=tierstream_descriptor(c,utf8_path(p.primary),model_sha256).dump();
    manifest_out.write(mtext.data(),mtext.size());manifest_out.flush();descriptor_out.write(dtext.data(),dtext.size());descriptor_out.flush();
    verify_units(c,[&](std::uint64_t off,void* d,std::size_t n){units.read_at(off,d,n);},manifest,model_sha256);
    units.publish(p.primary);manifest_out.publish(p.manifest);descriptor_out.publish(p.descriptor);
    auto result=strix::serve::parse_json(mtext);result.set("manifest_sha256",strix::model_content_identity_hex(strix::sha256_bytes(std::as_bytes(std::span(mtext)))));return result;
}
}
