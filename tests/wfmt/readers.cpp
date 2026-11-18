#include "strix/weight/admission.hpp"
#include "strix/weight/registry.hpp"
#include "strix/ds41_native_fp8.hpp"
#include "strix/vision/gguf_weights.hpp"
#include "gemma4/model/model.hpp"
#include "h3/block/safetensors.hpp"
#include <cstring>
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace strix::weight;
namespace {
void check(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void u64(std::vector<Byte>& b,std::uint64_t n,unsigned width=8){for(unsigned i=0;i<width;++i)b.push_back(Byte((n>>(8*i))&255));}
void text(std::vector<Byte>& b,std::string_view s){u64(b,s.size());for(auto c:s)b.push_back(Byte(c));}
std::vector<Byte> safe(std::string header,std::size_t payload){std::vector<Byte> b;u64(b,header.size());for(char c:header)b.push_back(Byte(c));for(std::size_t i=0;i<payload;++i)b.push_back(Byte(i&255));return b;}
}
bool wfmt_readers_test() try {
    std::vector<Byte> gg;for(char c:std::string("GGUF"))gg.push_back(Byte(c));u64(gg,3,4);u64(gg,2);u64(gg,0);
    for(unsigned type:{1U,8U}){text(gg,type==1?"a.weight":"b.weight");u64(gg,2,4);u64(gg,32);u64(gg,2);u64(gg,type,4);u64(gg,type==1?0:128);}
    while(gg.size()%32)gg.push_back(Byte{0});const auto payload=gg.size();for(unsigned i=0;i<196;++i)gg.push_back(Byte(i&255));
    auto source=memory_source(gg,"mixed.gguf");TensorIndex index;check(bool(index_gguf(source,index)),"mixed GGUF index");
    check(index.tensors.size()==2 && index.tensors[0].wire_type==1 && index.tensors[1].wire_type==8,"mixed wire identities preserved");
    check(index.tensors[0].shape.rank==2 && index.tensors[0].shape.dims[0]==2 && index.tensors[0].shape.dims[1]==32,"canonical logical axes");
    check(index.tensors[1].payload.range.offset==payload+128 && index.tensors[1].payload.range.bytes==68,"GGUF exact byte range");
    auto retained=index.tensors[1].payload;index={};source.reset();std::array<Byte,68> bytes{};check(bool(read_slice(retained,bytes)),"source survives index destruction");
    check(std::equal(bytes.begin(),bytes.end(),gg.begin()+payload+128),"GGUF payload unchanged");
    WeightMap vision_owned;
    {
        auto vision_source=memory_source(gg);auto vision_stream=source_stream(vision_source);auto info=strix::gguf::read(*vision_stream,gg.size());
        auto malformed=info;malformed.tensors[0].dimensions={1,1,1,2,32};TensorIndex impossible;
        const auto rank_status=index_gguf(malformed,vision_source,impossible);check(!rank_status && rank_status.code==Error::unsupported_shape,"externally supplied GGUF rank is bounded before copying axes");
        bool rank_refused=false;try{strix::vision::GgufWeights bad(malformed,gg);(void)bad.tensor("a.weight");}catch(const std::invalid_argument&){rank_refused=true;}check(rank_refused,"vision rejects over-rank external metadata before indexing fixed axes");
        strix::vision::GgufWeights legacy_vision(std::move(info),std::move(gg));const auto tensor=legacy_vision.tensor("a.weight");
        check(tensor.rows==2 && tensor.cols==32,"vision legacy shape unchanged");
        check(std::memcmp(tensor.data,legacy_vision.bytes().data()+payload,128)==0,"vision legacy operand bytes unchanged");
        vision_owned=legacy_vision.weight_map();check(vision_owned.size()==1,"vision requested tensor enters common admission");
        bool unsupported=false;try{(void)legacy_vision.tensor("b.weight");}catch(const std::invalid_argument&){unsupported=true;}check(unsupported,"vision unsupported quant remains rejected");
    }
    std::array<Byte,128> vision_bytes{};check(bool(read_slice(vision_owned[0].weight.payload,vision_bytes)),"vision common map retains memory after legacy source destruction");
    auto wire=safe(R"({"a.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"aux":{"dtype":"I64","shape":[1],"data_offsets":[8,16]}})",16);
    TensorIndex st;check(bool(index_safetensors(memory_source(wire,"sub-a/model.safetensors"),st)),"safetensors typed index");check(st.tensors.size()==2 && st.tensors[1].wire_dtype=="I64","typed dtype preserved");
    auto same=memory_source(wire,"sub-b/model.safetensors");TensorIndex other;check(bool(index_safetensors(same,other)),"other shard");check(st.tensors[0].id!=other.tensors[0].id,"same-basename relative shard paths remain distinct");
    TensorIndex copied;check(bool(index_safetensors(memory_source(wire,"sub-a/model.safetensors"),copied)),"copied source index");
    check(copied.tensors[0].id==st.tensors[0].id,"deployment copies preserve stable tensor identity");
    auto bad=wire;bad.pop_back();TensorIndex untouched;untouched.schema="unchanged";check(!index_safetensors(memory_source(bad),untouched),"truncation rejected");check(untouched.schema=="unchanged","failure does not publish partial index");
    check(!index_safetensors(memory_source(safe(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})",8)),other),"duplicate name rejected");
    check(!index_safetensors(memory_source(safe(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",4)),other),"overlap rejected");
    check(!index_safetensors(memory_source(safe(R"({"a":{"dtype":"F32","shape":[9223372036854775807,4],"data_offsets":[0,4]}})",4)),other),"shape overflow rejected");
    check(!index_safetensors(memory_source(safe(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})",8)),other),"unclaimed hole rejected");
    {
        // A rank-5 tensor (Conv3d patch embedding of a vision tower) must not fail the index; it is kept opaque.
        auto rank5=safe(R"({"vision_tower.patch_embedding.weight":{"dtype":"BF16","shape":[2,3,2,1,1],"data_offsets":[0,24]},"model.layers.0.mlp.down_proj.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[24,32]}})",32);
        TensorIndex r5;check(bool(index_safetensors(memory_source(rank5,"model-vision.safetensors"),r5)),"rank-5 safetensors indexed");
        check(r5.tensors.size()==2,"rank-5 index census");
        const auto& v=r5.tensors[0].name.starts_with("vision")?r5.tensors[0]:r5.tensors[1];
        const auto& d=r5.tensors[0].name.starts_with("vision")?r5.tensors[1]:r5.tensors[0];
        check(v.opaque() && v.shape.rank==0 && v.extents==std::vector<std::uint64_t>({2,3,2,1,1}),"rank-5 tensor opaque with full extents");
        check(v.wire_dtype=="BF16" && v.payload.range.bytes==24,"opaque tensor keeps dtype and byte range");
        check(!d.opaque() && d.shape.rank==2 && d.shape.dims[0]==2 && d.shape.dims[1]==2 && d.extents==std::vector<std::uint64_t>({2,2}),"viewable tensor shape unchanged");
        WeightMap mm;check(bool(bind_minimax(r5,mm)) && mm.size()==1,"family binding skips the opaque tensor and admits the rest");
        TensorBinding b{};b.tensor=v.id;b.role=Role::embedding;b.expert_axis=UINT32_MAX;b.admitted_formats={{"plain.bf16",1}};
        WeightMap direct;const auto status=admit(builtin_registry(),{&r5,{&b,1},{}},direct);
        check(!status && status.code==Error::unsupported_shape,"explicit view of an opaque tensor rejected per tensor");
        {
            // File-backed legacy tables (shared archive used by Gemma / Qwen vision, H3) keep the full rank-5 extents.
            const auto dir=std::filesystem::temp_directory_path()/("wfmt-rank5-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            std::filesystem::create_directory(dir);const auto file=dir/"model.safetensors";
            struct Gone{std::filesystem::path p;~Gone(){std::error_code ec;std::filesystem::remove(p,ec);std::filesystem::remove(p.parent_path(),ec);}} gone{file};
            {std::ofstream f(file,std::ios::binary);f.write(reinterpret_cast<const char*>(rank5.data()),std::streamsize(rank5.size()));}
            {
                auto archive=strix::gemma4::read_safetensors(file);check(archive.tensors.size()==2,"rank-5 archive census");
                const auto& t=archive.tensors[0].name.starts_with("vision")?archive.tensors[0]:archive.tensors[1];
                check(t.shape==std::vector<std::uint64_t>({2,3,2,1,1}) && t.bytes==24,"archive keeps rank-5 extents");
                strix::h3::block::SafeTensors h3(file.string());const auto& h=h3.at("vision_tower.patch_embedding.weight");
                check(h.shape==std::vector<std::int64_t>({2,3,2,1,1}),"H3 table keeps rank-5 extents");
            }
        }
        std::string ones;for(unsigned k=0;k<256;++k)ones+=k?",1":"1";
        check(!index_safetensors(memory_source(safe(R"({"a":{"dtype":"F32","shape":[)"+ones+R"(],"data_offsets":[0,4]}})",4)),other),"absurd rank rejected");
    }
    check(!index_pack(memory_source(wire),other),"unverified pack schema rejected");
    auto packed=safe(R"({"__metadata__":{"strix.weight.schema":"exl3.mul1.tiles16@1"},"x":{"dtype":"I16","shape":[1],"data_offsets":[0,2]}})",2);
    check(bool(index_pack(memory_source(packed),other)) && other.container==Container::pack,"declared pack indexed without advertising compute");
    // The existing DS41 API consumes exactly the same positioned bytes through its retained Source.
    const auto directory=std::filesystem::temp_directory_path()/("wfmt-source-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);const auto path=directory/"model.safetensors";
    struct Cleanup{std::filesystem::path p;~Cleanup(){std::error_code ec;std::filesystem::remove(p,ec);std::filesystem::remove(p.parent_path(),ec);}} cleanup{path};
    {std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(wire.data()),std::streamsize(wire.size()));}
    std::weak_ptr<strix::gemma4::MappedFile> mapped_owner;
    Slice mapped_slice;
    {
        auto gemma=strix::gemma4::read_safetensors(path);check(gemma.tensors.size()==2,"Gemma common parser tensor census");
        const auto& tensor=gemma.tensors[0];check(tensor.dtype=="BF16" && tensor.shape==std::vector<std::uint64_t>({2,2}) && tensor.bytes==8,"Gemma legacy type/shape preserved");
        check(std::memcmp(gemma.storage->data()+tensor.offset,wire.data()+tensor.offset,8)==0,"Gemma mapped operand bytes unchanged");
        strix::h3::block::SafeTensors h3(path.string());const auto values=h3.read_vec<std::uint16_t>("a.weight",strix::h3::block::StDtype::bf16);
        check(values.size()==4 && std::memcmp(values.data(),wire.data()+tensor.offset,8)==0,"H3 legacy read_vec bytes unchanged");
        mapped_owner=gemma.storage;mapped_slice=gemma.index.tensors[0].payload;
    }
    check(!mapped_owner.expired(),"Gemma admission retains the execution mapping itself");
    std::array<Byte,8> mapped_bytes{};
    check(bool(read_slice(mapped_slice,mapped_bytes)),"Gemma slice survives wrapper destruction");
    {
        const auto mapping=mapped_owner.lock();
        check(std::memcmp(mapped_bytes.data(),mapping->data()+mapped_slice.range.offset,8)==0,"Gemma admission and execution use the same backing bytes");
    }
    mapped_slice={};
    check(mapped_owner.expired(),"Gemma mapping released after its last slice");
    std::shared_ptr<const Source> file;check(bool(open_source(path.string(),file)),"file source open");
    const auto digest=file->identity();check(digest.bytes==digest_bytes(wire).bytes,"source identity is SHA256 content, not metadata");
    TensorIndex file_index;check(bool(index_safetensors(file,file_index)),"file index");
    strix::ds41::SafetensorsCheckpoint::Tensor legacy;legacy.file=path;legacy.source=file_index.tensors[0].payload;legacy.offset=legacy.source.range.offset;legacy.bytes=legacy.source.range.bytes;
    strix::ds41::SafetensorsCheckpoint checkpoint(directory);
    const auto* old=checkpoint.find("a.weight");check(old!=nullptr,"DS41 wrapper indexes existing tensor name");
    std::array<std::uint8_t,8> legacy_bytes{};checkpoint.read(*old,legacy_bytes.data());
    check(std::equal(legacy_bytes.begin(),legacy_bytes.end(),wire.begin()+old->offset,[](auto a,auto b){return a==std::to_integer<std::uint8_t>(b);}),"legacy DS41 wrapper bytes unchanged");
    // Positioned read through the source has the same bytes as the prior offset-based read API.
    std::array<Byte,8> observed{};check(bool(read_slice(legacy.source,observed)),"legacy slice read");
    check(std::equal(observed.begin(),observed.end(),wire.begin()+legacy.offset),"legacy payload byte identity");
    {std::ofstream f(path,std::ios::binary|std::ios::app);f.put('x');}
    check(!file->read({0,1},{observed.data(),1}),"retained source mutation is stale");
    bool stale=false;try{(void)file->identity();}catch(const std::exception&){stale=true;}check(stale,"cached identity cannot mask source mutation");
    return true;
}catch(const std::exception& e){std::cerr<<"readers: "<<e.what()<<'\n';return false;}
