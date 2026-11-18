#include "strix/weight/source.hpp"
#include "h3/te/te_sha256.hpp"
#include "strix/file_snapshot.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <streambuf>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <fcntl.h>
#endif
namespace strix::weight {
namespace {
Digest256 digest(std::span<const Byte> bytes) {
    h3::te::Sha256 hash; hash.update(bytes.data(),bytes.size()); return {hash.finish()};
}
class FileSource final : public Source {
    struct Snapshot { std::array<std::uint64_t,6> words{}; bool operator==(const Snapshot&) const = default; };
    std::FILE* file_; std::uint64_t size_; std::string shard_; Snapshot snapshot_;
    Snapshot snapshot() const {
#ifdef _WIN32
        auto handle=reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(file_)));
        FILE_BASIC_INFO basic{}; FILE_STANDARD_INFO standard{};
        if(!GetFileInformationByHandleEx(handle,FileBasicInfo,&basic,sizeof basic) || !GetFileInformationByHandleEx(handle,FileStandardInfo,&standard,sizeof standard))
            throw std::runtime_error("cannot snapshot source handle");
        return {{{std::uint64_t(standard.EndOfFile.QuadPart),std::uint64_t(basic.LastWriteTime.QuadPart),std::uint64_t(basic.ChangeTime.QuadPart),0,0,0}}};
#else
        struct stat st{};if(fstat(fileno(file_),&st))throw std::runtime_error("cannot snapshot source handle");
        return {{{std::uint64_t(st.st_size),std::uint64_t(st.st_mtim.tv_sec),std::uint64_t(st.st_mtim.tv_nsec),std::uint64_t(st.st_ctim.tv_sec),std::uint64_t(st.st_ctim.tv_nsec),std::uint64_t(st.st_ino)}}};
#endif
    }
    mutable std::mutex mutex_; mutable std::optional<Digest256> digest_; mutable std::once_flag digest_once_;
public:
    explicit FileSource(const std::string& path,std::string shard) : file_(std::fopen(path.c_str(),"rb")), size_(0),
        shard_(shard.empty()?std::filesystem::path(path).filename().generic_string():std::move(shard)) {
        if (!file_) throw std::runtime_error("cannot open source");
        try {
            snapshot_=snapshot();size_=snapshot_.words[0];
#ifndef _WIN32
            LoaderCapture::record(path,fileno(file_));
#endif
        }catch(...){std::fclose(file_);throw; }
    }
    ~FileSource() override { std::fclose(file_); }
    std::uint64_t size() const noexcept override { return size_; }
    void discard(ByteRange range) const noexcept override {
#ifndef _WIN32
        if(range.offset<=size_ && range.bytes<=size_-range.offset)
            (void)posix_fadvise(fileno(file_),static_cast<off_t>(range.offset),static_cast<off_t>(range.bytes),POSIX_FADV_DONTNEED);
#else
        (void)range;
#endif
    }
    std::string shard_name() const override { return shard_; }
    Status read(ByteRange range,std::span<Byte> out) const override {
        if (range.offset>size_ || range.bytes>size_-range.offset || range.bytes!=out.size())
            return {Error::malformed,"source read outside bounds"};
        if (out.empty()) return {Error::ok,{}};
        std::lock_guard lock(mutex_);
        try {if(!(snapshot()==snapshot_))return {Error::stale_source,"source changed after indexing"};}
        catch(const std::exception& e){return {Error::stale_source,e.what()};}
#ifdef _WIN32
        if (_fseeki64(file_,static_cast<__int64>(range.offset),SEEK_SET))
#else
        if (fseeko(file_,static_cast<off_t>(range.offset),SEEK_SET))
#endif
            return {Error::device_failure,"source seek failed"};
        if (std::fread(out.data(),1,out.size(),file_)!=out.size())
            return {Error::stale_source,"source short read"};
        try {if(!(snapshot()==snapshot_))return {Error::stale_source,"source changed during read"};}
        catch(const std::exception& e){return {Error::stale_source,e.what()};}
        return {Error::ok,{}};
    }
    Digest256 identity() const override {
        {std::lock_guard lock(mutex_);if(!(snapshot()==snapshot_))throw std::runtime_error("source changed after indexing");}
        std::call_once(digest_once_,[&] {
            h3::te::Sha256 hash;std::array<Byte,65536> bytes;
            for(std::uint64_t at=0;at<size_;) {
                const auto n=std::size_t(std::min<std::uint64_t>(bytes.size(),size_-at));
                auto status=read({at,n},{bytes.data(),n});if(!status)throw std::runtime_error(status.detail);
                hash.update(bytes.data(),n);at+=n;
            }
            const auto result=Digest256{hash.finish()};
            std::lock_guard lock(mutex_);if(!(snapshot()==snapshot_))throw std::runtime_error("source changed during digest");digest_=result;
        });
        std::lock_guard lock(mutex_);if(!(snapshot()==snapshot_))throw std::runtime_error("source changed after digest");return *digest_;
    }
};
class NamespaceSource final:public Source {
    std::shared_ptr<const Source> source_;std::string shard_;
public:
    NamespaceSource(std::shared_ptr<const Source> source,std::string shard):source_(std::move(source)),shard_(std::move(shard)){}
    Status read(ByteRange range,std::span<Byte> out) const override{return source_->read(range,out);}
    std::uint64_t size() const noexcept override{return source_->size();}
    Digest256 identity() const override{return source_->identity();}
    std::string shard_name() const override{return shard_;}
    void discard(ByteRange range) const noexcept override{source_->discard(range);}
};
class MemorySource final : public Source {
    std::vector<Byte> data_; std::string shard_;
    mutable std::once_flag identity_once_;mutable Digest256 identity_{};
public:
    MemorySource(std::span<const Byte> bytes,std::string shard):data_(bytes.begin(),bytes.end()),shard_(std::move(shard)){}
    std::uint64_t size() const noexcept override {return data_.size();}
    std::string shard_name() const override {return shard_;}
    Digest256 identity() const override {std::call_once(identity_once_,[&]{identity_=digest(data_);});return identity_;}
    Status read(ByteRange r,std::span<Byte> out) const override {
        if(r.offset>data_.size() || r.bytes>data_.size()-r.offset || r.bytes!=out.size())
            return {Error::malformed,"source read outside bounds"};
        if(!out.empty()) std::memcpy(out.data(),data_.data()+r.offset,out.size());
        return {Error::ok,{}};
    }
};
class ViewSource final : public Source {
    std::span<const Byte> bytes_; std::shared_ptr<const void> owner_; std::string shard_;
    mutable std::once_flag identity_once_; mutable Digest256 identity_{};
public:
    ViewSource(std::span<const Byte> bytes,std::shared_ptr<const void> owner,std::string shard):bytes_(bytes),owner_(std::move(owner)),shard_(std::move(shard)){}
    std::uint64_t size() const noexcept override{return bytes_.size();}
    std::string shard_name() const override{return shard_;}
    Digest256 identity() const override{std::call_once(identity_once_,[&]{identity_=digest(bytes_);});return identity_;}
    Status read(ByteRange range,std::span<Byte> out) const override {
        if(!owner_)return {Error::stale_source,"borrowed view has no owner"};
        if(range.offset>bytes_.size() || range.bytes>bytes_.size()-range.offset || range.bytes!=out.size())return {Error::malformed,"view read outside bounds"};
        if(!out.empty())std::memcpy(out.data(),bytes_.data()+range.offset,out.size());return {Error::ok,{}};
    }
};
class SourceBuffer : public std::streambuf {
    std::shared_ptr<const Source> source_; std::uint64_t at_=0;std::array<char,65536> buffer_{};
public:
    explicit SourceBuffer(std::shared_ptr<const Source> s):source_(std::move(s)){setg(buffer_.data(),buffer_.data(),buffer_.data());}
    int_type underflow() override {
        if(gptr()!=egptr())return traits_type::to_int_type(*gptr());
        if(at_==source_->size())return traits_type::eof();
        const auto n=std::size_t(std::min<std::uint64_t>(buffer_.size(),source_->size()-at_));
        if(!source_->read({at_,n},{reinterpret_cast<Byte*>(buffer_.data()),n}))return traits_type::eof();
        at_+=n;setg(buffer_.data(),buffer_.data(),buffer_.data()+n);return traits_type::to_int_type(*gptr());
    }
    pos_type seekoff(off_type offset,std::ios_base::seekdir dir,std::ios_base::openmode which) override {
        if(!(which&std::ios_base::in))return pos_type(off_type(-1));
        const auto current=at_-std::uint64_t(egptr()-gptr());
        const auto base=dir==std::ios_base::beg?0:dir==std::ios_base::cur?current:source_->size();
        if(offset<0 && std::uint64_t(-(offset+1))+1>base)return pos_type(off_type(-1));
        const auto next=offset<0?base-(std::uint64_t(-(offset+1))+1):base+std::uint64_t(offset);
        if(next>source_->size() || (offset>=0 && next<base))return pos_type(off_type(-1));
        at_=next;setg(buffer_.data(),buffer_.data(),buffer_.data());return pos_type(at_);
    }
    pos_type seekpos(pos_type pos,std::ios_base::openmode which) override {return seekoff(off_type(pos),std::ios_base::beg,which);}
};
class SourceStream final:public std::istream {
    SourceBuffer buffer_;
public: explicit SourceStream(std::shared_ptr<const Source> source):std::istream(nullptr),buffer_(std::move(source)){rdbuf(&buffer_);}
};
}
Digest256 digest_bytes(std::span<const Byte> bytes) {return digest(bytes);}
Status open_source(const std::string& path,std::shared_ptr<const Source>& out,std::string shard) {
    try {auto source=std::make_shared<FileSource>(path,std::move(shard));out=std::move(source);return {Error::ok,{}};}
    catch(const std::exception& e){return {Error::malformed,e.what()};}
}
std::shared_ptr<const Source> memory_source(std::span<const Byte> bytes,std::string shard) {return std::make_shared<MemorySource>(bytes,std::move(shard));}
std::shared_ptr<const Source> view_source(std::span<const Byte> bytes,std::shared_ptr<const void> owner,std::string shard){return std::make_shared<ViewSource>(bytes,std::move(owner),std::move(shard));}
std::unique_ptr<std::istream> source_stream(std::shared_ptr<const Source> source) {return std::make_unique<SourceStream>(std::move(source));}
TensorId tensor_id(const Source& source,std::string_view name,const Digest256& header) {
    h3::te::Sha256 hash;const auto shard=source.shard_name();
    if(shard.empty()){auto d=source.identity();hash.update(d.bytes.data(),d.bytes.size());}
    else hash.update(shard.data(),shard.size());
    const char separator=0;hash.update(&separator,1);hash.update(header.bytes.data(),header.bytes.size());hash.update(name.data(),name.size());
    const auto d=hash.finish();TensorId id=0;for(unsigned i=0;i<8;++i)id|=std::uint64_t(d[i])<<(8*i);return id;
}
std::shared_ptr<const Source> namespace_source(std::shared_ptr<const Source> source,std::string shard){return std::make_shared<NamespaceSource>(std::move(source),std::move(shard));}
Status digest_slice(const Slice& slice,Digest256& out) {
    if(!slice.source || slice.range.offset>slice.source->size() || slice.range.bytes>slice.source->size()-slice.range.offset)return {Error::malformed,"digest slice outside source"};
    h3::te::Sha256 hash;std::array<Byte,65536> bytes;
    for(std::uint64_t at=0;at<slice.range.bytes;) {
        const auto n=std::size_t(std::min<std::uint64_t>(bytes.size(),slice.range.bytes-at));
        auto status=slice.source->read({slice.range.offset+at,n},{bytes.data(),n});if(!status)return status;
        hash.update(bytes.data(),n);at+=n;
    }
    out={hash.finish()};return {Error::ok,{}};
}

Status read_slice(const Slice& slice,std::span<Byte> out) {
    if(!slice.source)return {Error::stale_source,"missing source ownership"};
    return slice.source->read(slice.range,out);
}
}
