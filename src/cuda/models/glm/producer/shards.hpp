#pragma once
// CPU-only virtual cache. The native K16/N16 decoder consumes shard payloads
// verbatim; retiling here is sorted placement plus zero padding, never decoding.
#include "strix/serve/json.hpp"
#include "strix/model_content_hash.hpp"
#include <filesystem>
#include <fstream>
#include <map>
#include <algorithm>
#include <functional>
#include <cstring>
#include <cmath>
#include <memory>
#include "expert_source.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "verified_io.hpp"
#endif

namespace strix::glm::rtxpf::producer {
using strix::serve::Json;
inline void shard_need(bool b,const char* message){if(!b)throw std::runtime_error(message);}
inline const Json& member(const Json& j,const std::string& key){auto p=j.find(key);shard_need(p!=nullptr,"missing shard field");return *p;}
inline std::uint64_t natural(const Json& j){auto v=j.as_int();shard_need(v>=0,"negative shard integer");return std::uint64_t(v);}
#ifndef _WIN32
inline std::string file_text(const std::filesystem::path& p){std::ifstream f(p,std::ios::binary);shard_need(bool(f),"open source metadata");return {(std::istreambuf_iterator<char>(f)),{}};}
#endif
inline Json sorted_json(const Json& j){
    if(j.is_object()){std::map<std::string,Json> m;for(auto& [k,v]:j.as_object()){shard_need(!m.contains(k),"duplicate JSON key");m.emplace(k,sorted_json(v));}auto out=Json::object();for(auto& [k,v]:m)out.set(k,v);return out;}
    if(j.is_array()){auto out=Json::array();for(auto& v:j.as_array())out.push(sorted_json(v));return out;}return j;
}
inline std::string json_hash(const Json& j){auto s=sorted_json(j).dump();return strix::model_content_identity_hex(strix::sha256_bytes(std::as_bytes(std::span(s))));}
#ifndef _WIN32
inline std::string source_sha256(const std::filesystem::path& path){
    std::ifstream f(path,std::ios::binary);shard_need(bool(f),"open source for hash");strix::Sha256Fast sha;std::vector<std::byte> buffer(8<<20);std::uint64_t bytes=0;
    while(f){f.read(reinterpret_cast<char*>(buffer.data()),buffer.size());auto n=std::size_t(f.gcount());if(n){sha.update({buffer.data(),n});bytes+=n;}}
    shard_need(f.eof()&&bytes==std::filesystem::file_size(path),"source changed during hash");return strix::model_content_identity_hex(sha.finalize());
}
#endif
// Shared source inventory: deterministic filename order, independent whole-file
// hash chains, and held deny-write/delete handles for the worker's entire life.
// Small host fixtures exercise this without allocating full-model tensors.
class ShardSources {
public:
    struct File {
        std::string name;std::filesystem::path path;std::uint64_t bytes;
#ifdef _WIN32
        std::unique_ptr<VerifiedFile> held;
#endif
        class Reader {
#ifdef _WIN32
            VerifiedFile::Reader reader_;
#else
            std::ifstream reader_;
#endif
        public:
            explicit Reader(const File& file,std::size_t window=65536):reader_(
#ifdef _WIN32
                *file.held,window
#else
                file.path,std::ios::binary
#endif
            ){}
            void read_at(std::uint64_t offset,void* destination,std::size_t n){
#ifdef _WIN32
                reader_.read_at(offset,destination,n);
#else
                reader_.seekg(offset);shard_need(bool(reader_.read(static_cast<char*>(destination),n)),"short source read");
#endif
            }
        };
    };
    std::vector<File> files;
    unsigned verified_hash_hits=0;
    // One aligned positional read of [begin,begin+length) into 64 KiB-aligned
    // dst with io_round(length) capacity, through the retained handle.
    static void read_span(const File& f,std::uint64_t begin,std::uint64_t length,void* dst){
#ifdef _WIN32
        f.held->read_span(begin,length,dst);
#else
        std::ifstream in(f.path,std::ios::binary);in.seekg(begin);shard_need(bool(in.read(static_cast<char*>(dst),length)),"short source span read");
#endif
    }
    explicit ShardSources(const std::filesystem::path& root){
        std::map<std::string,std::filesystem::path> paths;
        for(auto& f:std::filesystem::directory_iterator(root))if(f.path().extension()==".safetensors")paths.emplace(f.path().filename().string(),f.path());
        shard_need(!paths.empty(),"no source shards");
        for(auto& [name,path]:paths){
            File f{name,path,std::filesystem::file_size(path)};
#ifdef _WIN32
            f.held=std::make_unique<VerifiedFile>(path,f.bytes);
#endif
            files.push_back(std::move(f));
        }
    }
    const File& file(const std::string& name)const {
        auto i=std::lower_bound(files.begin(),files.end(),name,[](const File& f,const std::string& n){return f.name<n;});
        shard_need(i!=files.end()&&i->name==name,"unknown source shard");return *i;
    }
    Json records(bool hash_sources=true
#ifdef _WIN32
                 ,const std::filesystem::path& hash_cache=default_hash_cache()
#endif
                 ){
        std::vector<std::string> digests(files.size(),"UNHASHED-TEST-ONLY");verified_hash_hits=0;
#ifdef _WIN32
        std::atomic<unsigned> hits=0;
        if(hash_sources)parallel_hash_files(files.size(),[&](std::size_t j){
            auto result=files[j].held->compute_or_recall(hash_cache);digests[j]=std::move(result.sha256);if(result.recalled)++hits;
        });verified_hash_hits=hits.load();
#else
        if(hash_sources)for(std::size_t j=0;j<files.size();++j)digests[j]=source_sha256(files[j].path);
#endif
        auto sources=Json::array();
        for(std::size_t j=0;j<files.size();++j){auto s=Json::object();s.set("name",files[j].name);s.set("bytes",files[j].bytes);s.set("sha256",digests[j]);sources.push(s);}
        return sources;
    }
};
// Decode-capable inventories add the root-level vocabulary head. The "head"
// field enters the cache key, so a headless prefill cache never matches it.
inline constexpr const char* head_contract="lm_head-root-k6-v1";
inline bool head_tensor(const std::string& name){return name.rfind("lm_head.",0)==0;}
inline Json shard_content_manifest(const Json& config,const Json& sources,bool head=false){
    auto manifest=Json::object();manifest.set("format","glm-native-cache-v1");if(head)manifest.set("head",head_contract);manifest.set("layout","k16-n16-msb-mul1");manifest.set("decoder","mul1-8tiles-v2");manifest.set("tile","cutlass-c1-16experts-v1");manifest.set("transform","h128-rn16-v1");
    auto source=Json::object();source.set("config",sorted_json(config));source.set("files",sources);manifest.set("sources",sorted_json(source));manifest.set("model_sha256",json_hash(source));
    auto contract=sorted_json(manifest);contract.as_object().erase(std::remove_if(contract.as_object().begin(),contract.as_object().end(),[](auto& p){return p.first=="sources";}),contract.as_object().end());manifest.set("cache_key",json_hash(contract));return manifest;
}
struct ShardTensor {std::string file,dtype;std::uint64_t offset,bytes;Json shape;};
struct ShardPiece {std::string name;std::uint64_t offset;};
class Shards {
    std::filesystem::path root_;
    // Deny source writes/deletes for the lifetime of the worker on Windows.
    // Every Windows cursor consumes the retained authenticated object.
#ifdef _WIN32
    std::unique_ptr<VerifiedFile> config_lock_;
#endif
    std::unique_ptr<ShardSources> sources_;
#ifdef STRIX_HASH_AUTHORITY_TESTING
    friend struct ShardsFixture;
#endif
    struct InventoryOnly {};
    Shards(const std::string& root,InventoryOnly):root_(root){
#ifdef _WIN32
        config_lock_=std::make_unique<VerifiedFile>(root_/"config.json",std::filesystem::file_size(root_/"config.json"));
#endif
        sources_=std::make_unique<ShardSources>(root_);
    }
    Json config()const {
#ifdef _WIN32
        return strix::serve::parse_json(config_lock_->contents());
#else
        return strix::serve::parse_json(file_text(root_/"config.json"));
#endif
    }
    static std::string full(const std::string& n){return head_tensor(n)?n:"model.language_model."+n;}
    void required(const std::string& n,const char* dtype,std::initializer_list<unsigned> shape){
        auto i=tensors.find(full(n));shard_need(i!=tensors.end(),"incomplete full-model shards");auto& t=i->second;
        shard_need(t.dtype==dtype&&t.shape.size()==shape.size(),"source tensor dtype/shape");unsigned j=0;for(auto d:shape){auto v=natural(t.shape.as_array()[j++]);shard_need(d?v==d:(v==48||v==64||v==80||v==96),"source tensor dimensions/rate");}
        selected.emplace(i->first,t);
    }
    void matrix(const std::string& n,unsigned in,unsigned out){
        required(n+".mul1","I32",{});required(n+".trellis","I16",{in/16,out/16,0});required(n+".suh","F16",{in});required(n+".svh","F16",{out});
        for(auto suffix:{"bias","mcg","su","sv","out_scales","3inst"})shard_need(!tensors.contains(full(n+"."+suffix)),"unsupported source transform");
        // The payload marker is a 64 KiB-window read per matrix (~2.3 GiB in total):
        // deferred until after admission and authentication when hashing is deferred.
        if(defer_markers_)pending_markers_.push_back(full(n+".mul1"));else check_marker(full(n+".mul1"));
    }
    void check_marker(const std::string& name){auto& t=selected.at(name);ShardSources::File::Reader reader(sources_->file(t.file));std::uint32_t marker=0;reader.read_at(t.offset,&marker,4);shard_need(marker==0x83dcd12d,"source mul1 marker");}
    bool defer_markers_=false;std::vector<std::string> pending_markers_;
    void schema(){
        required("embed_tokens.weight","BF16",{154880,4096});required("norm.weight","BF16",{4096});
        for(unsigned l=0;l<46;++l){auto p="layers."+std::to_string(l)+".";required(p+"input_layernorm.weight","BF16",{4096});required(p+"post_attention_layernorm.weight","BF16",{4096});
            if(l<45)for(auto site:{"hc_attn_","hc_ffn_"}){auto h=p+site;required(h+"fn","F32",{24,16384});required(h+"base","F32",{24});required(h+"scale","F32",{3});}
            auto a=p+"self_attn.";
            if(l<45&&l%4!=3){matrix(a+"qkv_proj",4096,24576);matrix(a+"o_proj",8192,4096);
                required(a+"b_proj.weight","F16",{64,4096});for(auto s:{"f_a_proj.weight","g_a_proj.weight"})required(a+s,"F16",{128,4096});for(auto s:{"f_b_proj.weight","g_b_proj.weight"})required(a+s,"F16",{8192,128});
                required(a+"conv1d.weight","BF16",{24576,1,4});required(a+"o_norm.weight","BF16",{128});required(a+"A_log","F32",{64});required(a+"dt_bias","F32",{8192});
            }else{matrix(a+"q_a_proj",4096,1536);matrix(a+"q_b_proj",1536,16384);matrix(a+"kv_a_proj_with_mqa",4096,512);matrix(a+"o_proj",16384,4096);matrix(a+"indexer.wq_b",1536,4096);
                required(a+"q_a_layernorm.weight","BF16",{1536});required(a+"kv_a_layernorm.weight","BF16",{512});required(a+"kv_b_proj.weight","F16",{32768,512});
                for(auto s:{"indexer.wk.weight","indexer.index_kpool_compress_gate"})required(a+s,"F16",{128,4096});required(a+"indexer.weights_proj.weight","F16",{32,4096});for(auto s:{"indexer.k_norm.weight","indexer.k_norm.bias"})required(a+s,"F16",{128});required(a+"indexer.index_kpool_compress_ape","F32",{4,128});
            }
            auto m=p+"mlp.";auto expert=[&](const std::string& b,unsigned inter){matrix(b+"gate_proj",4096,inter);matrix(b+"up_proj",4096,inter);matrix(b+"down_proj",inter,4096);};
            if(l<3)expert(m,12288);else{required(m+"gate.weight","F16",{288,4096});required(m+"gate.e_score_correction_bias","F32",{288});expert(m+"shared_experts.",2048);for(unsigned e=0;e<288;++e)expert(m+"experts."+std::to_string(e)+".",2048);}
            if(l==45){matrix(p+"eh_proj",8192,4096);for(auto s:{"enorm","hnorm","shared_head.norm"})required(p+s+".weight","BF16",{4096});}
        }
        if(head_){matrix("lm_head",4096,154880);shard_need(natural(selected.at("lm_head.trellis").shape.as_array()[2])==96,"lm_head must be K6");
            shard_need(selected.at("lm_head.trellis").bytes==475791360&&selected.at("lm_head.suh").bytes==8192&&selected.at("lm_head.svh").bytes==309760,"lm_head K6 byte contract");}
    }
    void read_headers(){
        for(auto& file:sources_->files){auto& filename=file.name;auto bytes=file.bytes;
            ShardSources::File::Reader reader(file);std::uint64_t n=0;shard_need(bytes>=8,"invalid shard header length");reader.read_at(0,&n,8);shard_need(n>=2&&n<=64ULL*1024*1024&&n<=bytes-8,"invalid shard header length");std::string raw(n,'\0');reader.read_at(8,raw.data(),raw.size());auto header=strix::serve::parse_json(raw);std::vector<std::pair<std::uint64_t,std::uint64_t>> ranges;
            for(auto& [name,t]:header.as_object()){if(name=="__metadata__")continue;auto& offsets=member(t,"data_offsets").as_array();shard_need(offsets.size()==2,"shard offsets");auto lo=natural(offsets[0]),hi=natural(offsets[1]);auto dtype=member(t,"dtype").as_string();unsigned width=dtype=="F16"||dtype=="BF16"||dtype=="I16"?2:dtype=="F32"||dtype=="I32"?4:0;shard_need(width>0,"source dtype");std::uint64_t size=width;
                auto shape=member(t,"shape");for(auto& d:shape.as_array()){auto dim=natural(d);shard_need(dim&&size<=bytes/dim,"source shape overflow");size*=dim;}
                shard_need(lo<=hi&&hi<=bytes-8-n&&size==hi-lo&&!tensors.contains(name),"source bounds/duplicate tensor");tensors.emplace(name,ShardTensor{filename,dtype,8+n+lo,size,shape});ranges.emplace_back(lo,hi);
            }std::sort(ranges.begin(),ranges.end());for(std::size_t i=1;i<ranges.size();++i)shard_need(ranges[i-1].second<=ranges[i].first,"source overlap");
        }
    }
    void pack(){
        pieces.clear();auto ts=Json::object(),bs=Json::object();std::map<std::string,std::uint64_t> lengths;
        for(auto& [name,t]:selected){auto marker=name.find(".mlp.experts.");std::string blob=head_tensor(name)?"head.bin":"base.bin";if(marker!=std::string::npos){auto start=name.find(".layers.")+8;auto l=std::stoul(name.substr(start));blob="experts-"+std::string(l<10?"0":"")+std::to_string(l)+".bin";}auto& off=lengths[blob];off=(off+255)&~std::uint64_t(255);pieces[blob].push_back({name,off});auto v=Json::object();v.set("blob",blob);v.set("offset",off);v.set("bytes",t.bytes);v.set("dtype",t.dtype);v.set("shape",t.shape);ts.as_object().emplace_back(name,v);off+=t.bytes;}
        for(auto& [name,size]:lengths){auto v=Json::object();v.set("bytes",size);v.set("sha256","");bs.set(name,v);}manifest.set("tensors",ts);manifest.set("blobs",bs);
    }
    bool head_=false;
public:
    unsigned verified_hash_hits=0;
    std::map<std::string,ShardTensor> tensors,selected;
    std::map<std::string,std::vector<ShardPiece>> pieces;
    Json manifest;
    explicit Shards(const std::string& root,bool hash_sources=true,bool head=false):Shards(root,InventoryOnly{}){head_=head;defer_markers_=!hash_sources;
        auto config=this->config();auto* text=config.find("text_config");const auto& cfg=text?*text:config;
        for(auto [name,n]:std::initializer_list<std::pair<const char*,unsigned>>{{"hidden_size",4096},{"vocab_size",154880},{"num_hidden_layers",45},{"hc_mult",4},{"q_lora_rank",1536},{"kv_lora_rank",512},{"num_attention_heads",64},{"qk_nope_head_dim",256},{"qk_rope_head_dim",0},{"v_head_dim",256},{"index_n_heads",32},{"index_head_dim",128},{"index_kpool",4},{"index_topk",2048},{"n_routed_experts",288},{"moe_intermediate_size",2048},{"intermediate_size",12288},{"num_nextn_predict_layers",1}})shard_need(natural(member(cfg,name))==n,"unsupported model configuration");
        auto& linear=member(cfg,"linear_attn_config");shard_need(natural(member(linear,"num_heads"))==64&&natural(member(linear,"head_dim"))==128&&member(linear,"gate_lower_bound").as_double()==-5.0&&natural(member(linear,"short_conv_kernel_size"))==4,"unsupported linear attention");
        auto& lt=member(cfg,"layer_types").as_array();auto& mt=member(cfg,"mlp_layer_types").as_array();shard_need(lt.size()==45&&mt.size()==45,"source layer configuration");
        Json kda=Json::array();for(unsigned l=0;l<45;++l){shard_need(lt[l].as_string()==(l%4!=3?"linear_attention":"deepseek_sparse_attention")&&mt[l].as_string()==(l<3?"dense":"sparse"),"source layer types");if(l%4!=3)kda.push(int(l));}shard_need(member(linear,"kda_layers").dump()==kda.dump(),"source KDA layers");
        for(auto [name,value]:std::initializer_list<std::pair<const char*,double>>{{"routed_scaling_factor",2.5},{"num_experts_per_tok",8},{"rms_norm_eps",1e-5}})if(auto v=cfg.find(name))shard_need(v->as_double()==value,"source arithmetic configuration");
        read_headers();
        schema();manifest=shard_content_manifest(config,sources_->records(hash_sources),head_);verified_hash_hits=sources_->verified_hash_hits;
        pack();
    }
    // Deferred source authentication: an inventory opened with hash_sources=false
    // (sizes and layout only, e.g. for an early budget plan) hashes or recalls
    // every held source now and rebuilds the identical manifest with digests.
    void authenticate_sources(
#ifdef _WIN32
                              const std::filesystem::path& hash_cache=default_hash_cache()
#endif
                              ){
#ifdef _WIN32
        manifest=shard_content_manifest(config(),sources_->records(true,hash_cache),head_);
#else
        manifest=shard_content_manifest(config(),sources_->records(true),head_);
#endif
        verified_hash_hits=sources_->verified_hash_hits;pack();
        // Payload markers only now, after the sources were authenticated.
        for(const auto& n:pending_markers_)check_marker(n);pending_markers_.clear();defer_markers_=false;
    }
    // Metadata-only inventories (plan mode) never read payload markers.
    std::size_t deferred_marker_count()const{return pending_markers_.size();}
    // Bounded caller-owned staging; emit receives virtual cache offsets. Shard
    // readers have independent offset-based scratch windows (at most 16 MiB).
    void stream(const std::string& blob,void* staging,std::size_t capacity,const std::function<void(std::uint64_t,const void*,std::size_t)>& emit)const{
        shard_need(capacity>=256&&capacity%4==0,"staging alignment");std::unique_ptr<ShardSources::File::Reader> reader;const ShardSources::File* active=nullptr;std::uint64_t cursor=0;
        for(auto& piece:pieces.at(blob)){auto& t=selected.at(piece.name);if(piece.offset>cursor){std::memset(staging,0,std::size_t(piece.offset-cursor));emit(cursor,staging,std::size_t(piece.offset-cursor));}auto& file=sources_->file(t.file);if(active!=&file){reader.reset();reader=std::make_unique<ShardSources::File::Reader>(file,16ULL<<20);active=&file;}
            for(std::uint64_t off=0;off<t.bytes;){auto n=std::size_t(std::min<std::uint64_t>(capacity,t.bytes-off));reader->read_at(t.offset+off,staging,n);
                if(t.dtype=="F16"||t.dtype=="BF16"){auto* p=static_cast<std::uint16_t*>(staging);unsigned mask=t.dtype=="F16"?0x7c00:0x7f80;for(std::size_t i=0;i<n/2;++i)shard_need((p[i]&mask)!=mask,"nonfinite source weight");}else if(t.dtype=="F32"){auto* p=static_cast<float*>(staging);for(std::size_t i=0;i<n/4;++i)shard_need(std::isfinite(p[i]),"nonfinite source FP32");}
                emit(piece.offset+off,staging,n);off+=n;
            }cursor=piece.offset+t.bytes;
        }
    }
    // Shared CPU reader for resident loads and request-scoped file-tier read-ahead.
    // Its cursors are private to this call; source handles live with Shards.
    const ShardSources& sources()const{return *sources_;}
    // Selected-tensor ranges at caller-chosen destination offsets (compact
    // expert units). Coalesced aligned reads through the retained handles;
    // scales are checked finite exactly as the streaming loader does.
    IoCount read_ranges(const std::vector<std::pair<std::string,std::uint64_t>>& wanted,void* destination,void* scratch,std::uint64_t scratch_bytes)const{
        std::vector<SourceExtent> extents;std::vector<std::uint64_t> sizes;std::map<std::string,unsigned> index;
        for(auto& [name,dst]:wanted){auto& t=selected.at(name);auto [it,fresh]=index.emplace(t.file,unsigned(sizes.size()));if(fresh)sizes.push_back(sources_->file(t.file).bytes);
            extents.push_back({it->second,t.offset,t.bytes,dst,char(t.dtype=="F16"?'h':t.dtype=="BF16"?'b':t.dtype=="F32"?'f':0)});}
        std::vector<const ShardSources::File*> files(sizes.size());for(auto& [name,j]:index)files[j]=&sources_->file(name);
        return read_extents(extents,sizes,[&](unsigned f,std::uint64_t begin,std::uint64_t length,void* dst){ShardSources::read_span(*files[f],begin,length,dst);},destination,scratch,scratch_bytes);
    }
    void read_into(const std::string& blob,void* destination)const {
        std::vector<std::byte> staging(8<<20);
        stream(blob,staging.data(),staging.size(),[&](std::uint64_t off,const void* p,std::size_t n){std::memcpy(static_cast<char*>(destination)+off,p,n);});
    }

};
}
