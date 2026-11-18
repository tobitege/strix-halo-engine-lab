#include "cases.hpp"
#include "strix/ds41_native_fp8.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/vision/gguf_weights.hpp"
#include "gemma4/model/model.hpp"
#include "h3/block/safetensors.hpp"
#ifndef _WIN32
#include "glm/load/glm_rank_plan.hpp"
#include "minimax/engine/mm_common.hpp"
#endif
#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using Bytes=std::vector<std::byte>;
namespace {
void integer(Bytes& bytes,std::uint64_t n,unsigned width=8){for(unsigned i=0;i<width;++i)bytes.push_back(std::byte((n>>(8*i))&255));}
void string(Bytes& bytes,std::string_view text){integer(bytes,text.size());bytes.insert(bytes.end(),reinterpret_cast<const std::byte*>(text.data()),reinterpret_cast<const std::byte*>(text.data()+text.size()));}
void write(const std::filesystem::path& path,const Bytes& bytes){std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));if(!out)throw std::runtime_error("capture fixture write failed");}
void record(const std::filesystem::path& output,std::string_view case_name,std::string_view dtype,const std::vector<std::uint64_t>& shape,std::uint64_t offset,std::uint64_t count,const Bytes& source,const Bytes& payload,bool accepted=true){
    Bytes observed;integer(observed,accepted?1:0);string(observed,dtype);integer(observed,shape.size());for(auto d:shape)integer(observed,d);integer(observed,offset);integer(observed,count);observed.insert(observed.end(),payload.begin(),payload.end());
    if(!strix::weight::test::write_capture_record(output.string(),case_name,"legacy-wrapper",shape,source,observed))throw std::runtime_error("capture record write failed");
}
}
int main(int argc,char** argv) try {
    if(argc!=5 || std::string_view(argv[1])!="--wfmt-capture" || std::string_view(argv[3])!="--wfmt-matrix")return 2;
    const std::filesystem::path output(argv[2]);const auto directory=output/"fixtures";std::filesystem::create_directories(directory);
    const std::string header=R"({"a.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"b.weight":{"dtype":"F32","shape":[1],"data_offsets":[8,12]},"aux":{"dtype":"I64","shape":[1],"data_offsets":[12,20]},"codes":{"dtype":"U8","shape":[4],"data_offsets":[20,24]}})";
    Bytes safe;integer(safe,header.size());safe.insert(safe.end(),reinterpret_cast<const std::byte*>(header.data()),reinterpret_cast<const std::byte*>(header.data()+header.size()));for(unsigned i=0;i<24;++i)safe.push_back(std::byte(i));
    const auto file=directory/"weights.safetensors";write(file,safe);
    const std::string index=R"({"weight_map":{"a.weight":"weights.safetensors","b.weight":"weights.safetensors","aux":"weights.safetensors","codes":"weights.safetensors"}})";write(directory/"model.safetensors.index.json",{reinterpret_cast<const std::byte*>(index.data()),reinterpret_cast<const std::byte*>(index.data()+index.size())});
    const std::array<std::string,5> names={"a.weight","b.weight","aux","codes","missing"};
    {
        strix::ds41::SafetensorsCheckpoint checkpoint(directory);
        for(const auto& name:names){const auto* t=checkpoint.find(name);if(!t){record(output,"ds41-"+name,"",{},0,0,safe,{},false);continue;}Bytes payload(std::size_t(t->bytes));checkpoint.read(*t,reinterpret_cast<std::uint8_t*>(payload.data()));record(output,"ds41-"+name,t->dtype,t->shape,t->offset,t->bytes,safe,payload);}
    }
    {
        strix::h3::block::SafeTensors checkpoint(file.string());
        for(const auto& name:names){const auto* t=checkpoint.find(name);if(!t){record(output,"h3-"+name,"",{},0,0,safe,{},false);continue;}Bytes payload(std::size_t(t->nbytes()));checkpoint.read_raw(*t,payload.data());std::vector<std::uint64_t> shape(t->shape.begin(),t->shape.end());record(output,"h3-"+name,t->dtype_name,shape,t->begin,t->nbytes(),safe,payload);}
    }
    {
        const auto checkpoint=strix::gemma4::read_safetensors(file);
        for(const auto& name:names){const auto found=std::find_if(checkpoint.tensors.begin(),checkpoint.tensors.end(),[&](const auto& t){return t.name==name;});if(found==checkpoint.tensors.end()){record(output,"gemma-"+name,"",{},0,0,safe,{},false);continue;}const auto& t=*found;Bytes payload(std::size_t(t.bytes));std::memcpy(payload.data(),checkpoint.storage->data()+t.offset,payload.size());record(output,"gemma-"+name,t.dtype,t.shape,t.offset,t.bytes,safe,payload);}
    }
#ifndef _WIN32
    {
        strix::glm::load::Index checkpoint(directory.string());
        for(const auto& name:names){const auto* t=checkpoint.find(name);if(!t){record(output,"glm-"+name,"",{},0,0,safe,{},false);continue;}std::ifstream in(checkpoint.files().at(t->file),std::ios::binary);in.seekg(std::streamoff(t->offset));Bytes payload(std::size_t(t->bytes));in.read(reinterpret_cast<char*>(payload.data()),std::streamsize(payload.size()));if(!in)throw std::runtime_error("GLM capture raw read failed");record(output,"glm-"+name,t->dtype,t->shape,t->offset,t->bytes,safe,payload);}
    }
    {
        strix::minimax::engine::detail::PackReader checkpoint(directory.string());
        for(const auto& name:names){if(name=="missing"){bool refused=false;try{(void)checkpoint.info(name);}catch(const std::exception&){refused=true;}if(!refused)throw std::runtime_error("MiniMax accepted missing tensor");record(output,"minimax-"+name,"",{},0,0,safe,{},false);continue;}const auto& t=checkpoint.info(name);Bytes payload(std::size_t(t.bytes));checkpoint.read(t,0,t.bytes,payload.data());record(output,"minimax-"+name,t.dtype,t.shape,t.offset,t.bytes,safe,payload);}
    }
#endif
    Bytes gg;for(char c:std::string("GGUF"))gg.push_back(std::byte(c));integer(gg,3,4);integer(gg,4);integer(gg,0);
    const std::array<unsigned,4> types={0,1,30,8};const std::array<std::uint64_t,4> offsets={0,256,384,512};const std::array<std::uint64_t,4> counts={256,128,128,68};
    for(unsigned i=0;i<4;++i){string(gg,"v"+std::to_string(i)+".weight");integer(gg,2,4);integer(gg,32);integer(gg,2);integer(gg,types[i],4);integer(gg,offsets[i]);}
    while(gg.size()%32)gg.push_back(std::byte{});for(unsigned i=0;i<580;++i)gg.push_back(std::byte(i&255));
    const auto ggfile=directory/"vision.gguf";write(ggfile,gg);const auto info=strix::gguf::read_file(ggfile);strix::vision::GgufWeights vision(info,gg);
    for(unsigned i=0;i<4;++i){const auto& t=info.tensors[i];Bytes payload;bool accepted=false;std::vector<std::uint64_t> shape={2,32};try{auto tensor=vision.tensor(t.name);shape={tensor.rows,tensor.cols};payload.resize(std::size_t(t.byte_size));std::memcpy(payload.data(),tensor.data,payload.size());accepted=true;}catch(const std::invalid_argument&){}record(output,"vision-"+std::to_string(types[i]),"GGUF:"+std::to_string(types[i]),shape,t.absolute_offset,t.byte_size,gg,payload,accepted);}
    std::cout<<"legacy wrapper capture complete\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
