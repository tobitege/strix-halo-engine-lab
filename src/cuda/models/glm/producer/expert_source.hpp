#pragma once
// Host-only routed-expert addressing. Never assumes expert_id*average_bytes:
// cache packing sorts names lexicographically ("experts.1." < "experts.10." <
// "experts.100." < "experts.2."), so every (layer,expert) extent is looked up.
//
// GLM expert UNIT (tier cache layout): one opaque, 64 KiB-strided unit per
// (layer,expert) in NUMERIC expert order, carrying the original compressed
// bytes of gate/up/down in a canonical order (matrix gate,up,down; within each
// trellis,suh,svh,mul1), each tensor 256-byte aligned, zero padding elsewhere.
// Unit offsets depend only on the three rates. A unit manifest binds model
// identity, geometry, per-unit rates and SHA-256 of each complete unit image.
#include "strix/serve/json.hpp"
#include "strix/model_content_hash.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
namespace strix::glm::rtxpf::producer {
using strix::serve::Json;
inline void expert_need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
inline constexpr const char* expert_unit_format="glm-expert-unit-v1";
inline constexpr std::uint64_t unit_tensor_alignment=256,unit_alignment=65536;
inline constexpr const char* matrix_names[3]={"gate_proj","up_proj","down_proj"};
inline constexpr const char* role_names[4]={"trellis","suh","svh","mul1"};
struct ExpertGeometry {
    unsigned hidden=4096,inter=2048,experts=288,first_layer=3,last_layer=45;
    unsigned in(unsigned m)const{return m<2?hidden:inter;}
    unsigned out(unsigned m)const{return m<2?inter:hidden;}
    unsigned layers()const{return last_layer-first_layer+1;}
};
inline std::uint64_t trellis_bytes(unsigned in,unsigned out,unsigned rate){return std::uint64_t(in/16)*(out/16)*16*rate*2;}
inline std::uint64_t role_bytes(unsigned role,unsigned in,unsigned out,unsigned rate){return role==0?trellis_bytes(in,out,rate):role==1?std::uint64_t(in)*2:role==2?std::uint64_t(out)*2:4;}
inline std::uint64_t align_up(std::uint64_t v,std::uint64_t a){return (v+a-1)/a*a;}
// Checked JSON access: absent or mistyped fields are controlled integrity
// rejections (never a null dereference or a narrowing of a negative/fractional number).
inline const Json& json_field(const Json& o,const char* k,const char* what){expert_need(o.is_object(),what);auto* v=o.find(k);expert_need(v!=nullptr,what);return *v;}
inline std::uint64_t json_natural(const Json& v,const char* what){expert_need(v.is_number(),what);auto d=v.as_double();expert_need(d>=0&&d<=9007199254740992.0&&double(std::uint64_t(d))==d,what);return std::uint64_t(d);}
inline const std::string& json_text(const Json& v,const char* what){expert_need(v.is_string(),what);return v.as_string();}
inline const Json::Array& json_list(const Json& v,const char* what){expert_need(v.is_array(),what);return v.as_array();}
struct ExpertTensor {std::string name,blob;std::uint64_t offset=0,bytes=0;unsigned matrix=0,role=0;};
// Canonical unit placement. offsets[m][role] relative to the unit base.
struct UnitLayout {std::array<std::array<std::uint64_t,4>,3> offset{};std::array<unsigned,3> rate{};std::uint64_t bytes=0,logical=0;};
inline UnitLayout unit_layout(const ExpertGeometry& g,std::array<unsigned,3> rates){
    UnitLayout u;u.rate=rates;std::uint64_t at=0;
    for(unsigned m=0;m<3;++m){expert_need(rates[m]>=3&&rates[m]<=6,"expert rate must be K3..K6");
        for(unsigned r=0;r<4;++r){at=align_up(at,unit_tensor_alignment);u.offset[m][r]=at;auto n=role_bytes(r,g.in(m),g.out(m),rates[m]);at+=n;u.logical+=n;}}
    u.bytes=at;return u;
}
struct ExpertEntry {
    unsigned layer=0,expert=0;std::string blob;
    std::array<std::array<ExpertTensor,4>,3> t; // [matrix][role]
    std::uint64_t span_begin=0,span_end=0;     // lexicographic blob span (contiguous)
    UnitLayout unit;
};
class ExpertCatalog {
    ExpertGeometry g_;std::vector<ExpertEntry> table_;std::uint64_t stride_=0,max_unit_=0,max_span_=0;
public:
    static std::string prefix(unsigned layer,unsigned expert){return "model.language_model.layers."+std::to_string(layer)+".mlp.experts."+std::to_string(expert)+".";}
    static std::string blob_name(unsigned l){return std::string("experts-")+(l<10?"0":"")+std::to_string(l)+".bin";}
    // tensors: manifest "tensors" object (name -> blob, offset, bytes, dtype, shape).
    ExpertCatalog(const Json& tensors,ExpertGeometry g={}):g_(g){
        // One O(n) name index: Json::find is a linear scan over ~150K manifest tensors.
        expert_need(tensors.is_object(),"manifest tensors must be an object");std::unordered_map<std::string_view,const Json*> index;index.reserve(tensors.as_object().size());
        for(const auto& member:tensors.as_object())index.emplace(member.first,&member.second);
        auto get=[&](const std::string& name)->const Json&{auto it=index.find(name);expert_need(it!=index.end(),"missing routed expert tensor");return *it->second;};
        auto integer=[](const Json& j,const char* k){return json_natural(json_field(j,k,"expert tensor field"),"expert tensor integer");};
        auto text=[](const Json& j,const char* k){return json_text(json_field(j,k,"expert tensor field"),"expert tensor text");};
        table_.reserve(std::size_t(g_.layers())*g_.experts);
        for(unsigned l=g_.first_layer;l<=g_.last_layer;++l){std::vector<std::pair<std::uint64_t,std::uint64_t>> spans;
            for(unsigned e=0;e<g_.experts;++e){ExpertEntry x;x.layer=l;x.expert=e;x.blob=blob_name(l);std::array<unsigned,3> rates{};x.span_begin=~0ULL;
                for(unsigned m=0;m<3;++m)for(unsigned r=0;r<4;++r){
                    auto name=prefix(l,e)+matrix_names[m]+"."+role_names[r];const auto& j=get(name);auto& t=x.t[m][r];
                    t.name=name;t.blob=text(j,"blob");t.offset=integer(j,"offset");t.bytes=integer(j,"bytes");t.matrix=m;t.role=r;
                    expert_need(t.blob==x.blob,"routed expert tensor outside its layer blob");expert_need(t.offset%unit_tensor_alignment==0,"expert tensor alignment");
                    auto dtype=text(j,"dtype");auto* shape=j.find("shape");expert_need(shape&&shape->is_array(),"expert tensor shape");const auto& s=shape->as_array();
                    unsigned in=g_.in(m),out=g_.out(m);
                    if(r==0){expert_need(dtype=="I16"&&s.size()==3&&json_natural(s[0],"expert shape dimension")==in/16&&json_natural(s[1],"expert shape dimension")==out/16,"expert trellis geometry");auto k=json_natural(s[2],"expert shape dimension");expert_need(k%16==0&&k/16>=3&&k/16<=6,"expert trellis rate");rates[m]=unsigned(k/16);}
                    else if(r==1)expert_need(dtype=="F16"&&s.size()==1&&json_natural(s[0],"expert shape dimension")==in,"expert suh geometry");
                    else if(r==2)expert_need(dtype=="F16"&&s.size()==1&&json_natural(s[0],"expert shape dimension")==out,"expert svh geometry");
                    else expert_need(dtype=="I32"&&s.empty(),"expert mul1 geometry");
                    x.span_begin=std::min(x.span_begin,t.offset);x.span_end=std::max(x.span_end,t.offset+t.bytes);
                }
                for(unsigned m=0;m<3;++m)for(unsigned r=0;r<4;++r)expert_need(x.t[m][r].bytes==role_bytes(r,g_.in(m),g_.out(m),rates[m]),"expert tensor byte count");
                x.unit=unit_layout(g_,rates);
                // Lexicographic packing keeps one expert's twelve tensors adjacent;
                // anything wider means interleaving and is rejected.
                expert_need(x.span_end-x.span_begin<=x.unit.bytes+12*unit_tensor_alignment,"noncontiguous expert extent");
                spans.emplace_back(x.span_begin,x.span_end);max_unit_=std::max(max_unit_,x.unit.bytes);max_span_=std::max(max_span_,x.span_end-x.span_begin);
                table_.push_back(std::move(x));
            }
            std::sort(spans.begin(),spans.end());for(std::size_t i=1;i<spans.size();++i)expert_need(spans[i-1].second<=spans[i].first,"overlapping expert extents");
        }
        stride_=align_up(max_unit_,unit_alignment);
    }
    const ExpertGeometry& geometry()const{return g_;}
    const ExpertEntry& at(unsigned layer,unsigned expert)const{expert_need(layer>=g_.first_layer&&layer<=g_.last_layer&&expert<g_.experts,"expert index");return table_[std::size_t(layer-g_.first_layer)*g_.experts+expert];}
    std::uint64_t unit_stride()const{return stride_;}
    std::uint64_t max_unit_bytes()const{return max_unit_;}
    std::uint64_t max_span_bytes()const{return max_span_;}
    std::size_t size()const{return table_.size();}
    // Tier-cache gated ordinal: trunk routed layers 3..44 -> 0..41, MTP 45 -> 42.
    unsigned ordinal(unsigned layer)const{expert_need(layer>=g_.first_layer&&layer<=g_.last_layer,"expert layer");return layer-g_.first_layer;}
    std::uint64_t unit_offset(unsigned layer,unsigned expert)const{return (std::uint64_t(ordinal(layer))*g_.experts+expert)*stride_;}
};
// ----- Aligned, coalesced range reads with physical accounting -----
struct IoCount {std::uint64_t logical=0,physical=0,requests=0;};
struct AlignedRange {std::uint64_t begin=0,length=0,skip=0;};
// Smallest aligned window covering [offset,offset+n), clamped to end of file.
inline AlignedRange aligned_range(std::uint64_t offset,std::uint64_t n,std::uint64_t file_bytes,std::uint64_t align=unit_alignment){
    expert_need(offset<=file_bytes&&n<=file_bytes-offset,"range outside file");
    AlignedRange r;r.begin=offset/align*align;r.skip=offset-r.begin;r.length=std::min(align_up(offset+n,align),file_bytes)-r.begin;return r;
}
struct SourceExtent {unsigned file=0;std::uint64_t offset=0,bytes=0,dst=0;char check=0;}; // check: 'h' F16, 'b' BF16, 'f' F32, 0 none
struct ReadRun {unsigned file=0;std::uint64_t begin=0,end=0;std::vector<SourceExtent> pieces;};
// Group by file, ascending offset; join pieces whose gap is at most `gap` bytes
// while the run stays within `limit` bytes (bounded staging).
inline std::vector<ReadRun> coalesce(std::vector<SourceExtent> x,std::uint64_t gap,std::uint64_t limit){
    std::sort(x.begin(),x.end(),[](auto& a,auto& b){return a.file!=b.file?a.file<b.file:a.offset<b.offset;});std::vector<ReadRun> out;
    for(auto& e:x){expert_need(e.bytes>0&&e.bytes<=limit,"extent exceeds staging bound");
        if(!out.empty()&&out.back().file==e.file&&e.offset>=out.back().end&&e.offset-out.back().end<=gap&&e.offset+e.bytes-out.back().begin<=limit){out.back().end=e.offset+e.bytes;out.back().pieces.push_back(e);continue;}
        expert_need(out.empty()||out.back().file!=e.file||e.offset>=out.back().end,"overlapping source extents");
        out.push_back({e.file,e.offset,e.offset+e.bytes,{e}});
    }
    return out;
}
inline void check_finite_piece(const void* p,std::uint64_t n,char kind){
    if(kind=='h'||kind=='b'){auto* v=static_cast<const std::uint16_t*>(p);unsigned mask=kind=='h'?0x7c00:0x7f80;for(std::uint64_t i=0;i<n/2;++i)expert_need((v[i]&mask)!=mask,"nonfinite expert scale");}
    else if(kind=='f'){auto* v=static_cast<const std::uint32_t*>(p);for(std::uint64_t i=0;i<n/4;++i)expert_need((v[i]&0x7f800000u)!=0x7f800000u,"nonfinite expert FP32");}
}
// read(file,aligned_begin,length,dst): one aligned positional read of `length`
// bytes; dst is 64 KiB aligned with io-rounded capacity. scratch is caller-owned.
using AlignedReader=std::function<void(unsigned file,std::uint64_t begin,std::uint64_t length,void* dst)>;
inline IoCount read_extents(const std::vector<SourceExtent>& extents,const std::vector<std::uint64_t>& file_bytes,const AlignedReader& read,void* dst,void* scratch,std::uint64_t scratch_bytes,std::uint64_t gap=unit_alignment){
    expert_need(scratch_bytes>2*unit_alignment&&reinterpret_cast<std::uintptr_t>(scratch)%unit_alignment==0,"bounded aligned staging");
    IoCount io;for(auto& run:coalesce(extents,gap,scratch_bytes-2*unit_alignment)){expert_need(run.file<file_bytes.size(),"source file index");
        auto r=aligned_range(run.begin,run.end-run.begin,file_bytes[run.file]);expert_need(align_up(r.length,unit_alignment)<=scratch_bytes,"staging bound");
        read(run.file,r.begin,r.length,scratch);io.physical+=align_up(r.length,unit_alignment);++io.requests;
        for(auto& p:run.pieces){auto* src=static_cast<const char*>(scratch)+(p.offset-r.begin);check_finite_piece(src,p.bytes,p.check);std::memcpy(static_cast<char*>(dst)+p.dst,src,p.bytes);io.logical+=p.bytes;}
    }
    return io;
}
// Extents that assemble one canonical unit image from (file,offset) locations.
using TensorLocator=std::function<SourceExtent(const ExpertTensor&)>;
inline std::vector<SourceExtent> unit_extents(const ExpertEntry& x,const TensorLocator& locate){
    std::vector<SourceExtent> out;
    for(unsigned m=0;m<3;++m)for(unsigned r=0;r<4;++r){auto e=locate(x.t[m][r]);expert_need(e.bytes==x.t[m][r].bytes,"located extent size");e.dst=x.unit.offset[m][r];e.check=r==1||r==2?'h':0;out.push_back(e);}
    return out;
}
// Cache blob locator: the layer blob is file 0; offsets are virtual blob offsets.
inline SourceExtent blob_locator(const ExpertTensor& t){return {0,t.offset,t.bytes,0,0};}
inline std::string unit_sha256(std::span<const std::byte> image){return strix::model_content_identity_hex(strix::sha256_bytes(image));}
// Unit image: exactly `stride` bytes, zero outside tensor payloads; mul1 marker checked.
inline void validate_unit_image(const ExpertEntry& x,std::span<const std::byte> image,std::uint64_t stride){
    expert_need(image.size()==stride&&x.unit.bytes<=stride,"unit image extent");
    for(unsigned m=0;m<3;++m){std::uint32_t marker=0;std::memcpy(&marker,image.data()+x.unit.offset[m][3],4);expert_need(marker==0x83dcd12du,"unit mul1 marker");}
    std::vector<bool> used(stride);std::uint64_t covered=0;
    for(unsigned m=0;m<3;++m)for(unsigned r=0;r<4;++r){auto o=x.unit.offset[m][r],n=x.t[m][r].bytes;covered+=n;for(auto i=o;i<o+n;++i)used[i]=true;}
    expert_need(covered==x.unit.logical,"unit logical bytes");
    for(std::uint64_t i=0;i<stride;++i)if(!used[i])expert_need(image[i]==std::byte{0},"unit padding must be zero");
}
// Produces every unit image in numeric order (layer-major, expert ascending),
// validates it and returns per-unit SHA-256 for unit_manifest. The sink owns
// storage (a unit file at offset unit_offset(l,e)); nothing is written here.
using UnitSink=std::function<void(unsigned layer,unsigned expert,std::span<const std::byte>)>;
inline std::vector<std::string> produce_units(const ExpertCatalog& c,const TensorLocator& locate,const std::vector<std::uint64_t>& file_bytes,const AlignedReader& read,const UnitSink& sink){
    const auto& g=c.geometry();const auto stride=c.unit_stride();std::vector<std::byte> image(stride);
    const std::uint64_t scratch_bytes=align_up(c.max_span_bytes(),unit_alignment)+4*unit_alignment;std::vector<std::byte> raw(scratch_bytes+unit_alignment);
    void* scratch=raw.data()+(unit_alignment-reinterpret_cast<std::uintptr_t>(raw.data())%unit_alignment)%unit_alignment;
    std::vector<std::string> digests;digests.reserve(c.size());
    for(unsigned l=g.first_layer;l<=g.last_layer;++l)for(unsigned e=0;e<g.experts;++e){auto& x=c.at(l,e);std::fill(image.begin(),image.end(),std::byte{0});
        auto io=read_extents(unit_extents(x,locate),file_bytes,read,image.data(),scratch,scratch_bytes);expert_need(io.logical==x.unit.logical,"unit logical bytes read");
        validate_unit_image(x,image,stride);digests.push_back(unit_sha256(image));sink(l,e,image);}
    return digests;
}
inline bool sha256_hex(const std::string& s){return s.size()==64&&s.find_first_not_of("0123456789abcdef")==std::string::npos;}
// file_sha256: SHA-256 of the complete unit file (all units in numeric order),
// the digest startup authenticates through the held handle.
inline Json unit_manifest(const ExpertCatalog& c,const std::string& model_sha256,const std::vector<std::string>& digests,const std::string& file_sha256){
    const auto& g=c.geometry();expert_need(digests.size()==c.size(),"unit digest count");
    auto m=Json::object();m.set("format",expert_unit_format);m.set("model_sha256",model_sha256);m.set("unit_stride",c.unit_stride());m.set("unit_alignment",unit_alignment);m.set("tensor_alignment",unit_tensor_alignment);
    m.set("order","layer-major, numeric expert id; gate,up,down x trellis,suh,svh,mul1");m.set("first_layer",int(g.first_layer));m.set("last_layer",int(g.last_layer));m.set("experts",int(g.experts));m.set("hidden",int(g.hidden));m.set("inter",int(g.inter));
    auto units=Json::array();std::size_t j=0;
    for(unsigned l=g.first_layer;l<=g.last_layer;++l)for(unsigned e=0;e<g.experts;++e,++j){auto& x=c.at(l,e);auto u=Json::object();u.set("layer",int(l));u.set("expert",int(e));
        auto r=Json::array();for(auto v:x.unit.rate)r.push(int(v));u.set("rates",r);u.set("bytes",x.unit.bytes);u.set("sha256",digests[j]);units.push(u);}
    expert_need(sha256_hex(file_sha256),"unit file digest");m.set("file_sha256",file_sha256);m.set("units",units);return m;
}
// Rejects any manifest whose geometry, order, rates, extents or digest format
// differs from the catalog derived from the authenticated cache/shard metadata.
inline void validate_unit_manifest(const Json& m,const ExpertCatalog& c,const std::string& model_sha256){
    const auto& g=c.geometry();auto num=[&](const char* k){return json_natural(json_field(m,k,"unit manifest field"),"unit manifest integer");};
    expert_need(sha256_hex(json_text(json_field(m,"file_sha256","unit manifest field"),"unit manifest file digest")),"unit manifest file digest");
    expert_need(json_text(json_field(m,"format","unit manifest field"),"unit manifest format")==expert_unit_format&&json_text(json_field(m,"model_sha256","unit manifest field"),"unit manifest model")==model_sha256,"unit manifest format/model");
    expert_need(num("unit_stride")==c.unit_stride()&&num("unit_alignment")==unit_alignment&&num("tensor_alignment")==unit_tensor_alignment,"unit manifest stride/alignment");
    expert_need(num("first_layer")==g.first_layer&&num("last_layer")==g.last_layer&&num("experts")==g.experts&&num("hidden")==g.hidden&&num("inter")==g.inter,"unit manifest geometry");
    const auto& units=json_list(json_field(m,"units","unit manifest field"),"unit manifest units");expert_need(units.size()==c.size(),"unit manifest count");std::size_t j=0;
    for(unsigned l=g.first_layer;l<=g.last_layer;++l)for(unsigned e=0;e<g.experts;++e,++j){const auto& u=units[j];auto& x=c.at(l,e);
        auto unum=[&](const char* k){return json_natural(json_field(u,k,"unit entry field"),"unit entry integer");};
        expert_need(unum("layer")==l&&unum("expert")==e,"unit manifest order");
        const auto& r=json_list(json_field(u,"rates","unit entry field"),"unit entry rates");expert_need(r.size()==3,"unit manifest rates");
        for(unsigned k=0;k<3;++k)expert_need(json_natural(r[k],"unit entry rate")==x.unit.rate[k],"unit manifest rate");
        expert_need(unum("bytes")==x.unit.bytes,"unit manifest bytes");const auto& sha=json_text(json_field(u,"sha256","unit entry field"),"unit entry digest");
        expert_need(sha.size()==64&&sha.find_first_not_of("0123456789abcdef")==std::string::npos,"unit manifest digest");}
}
}
