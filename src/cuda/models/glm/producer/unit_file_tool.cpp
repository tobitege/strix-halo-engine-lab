// Host-only GLM expert unit file tool. No CUDA, no implicit writes.
//   plan    --model DIR --out PRIMARY [--stripe DIR:K | DIR:even:K]... [--chunk-mib 16]
//   dry-run (same arguments): also checks every source extent, targets and free space
//   copy    (same arguments) --confirm-bytes N   N = dry-run "primary_bytes"
//   verify  --model DIR --out PRIMARY
// Stripes are planned only: the dry-run writes nothing and prints the exact
// ts-stripe make command and ranges JSON for each secondary target.
#include "shards.hpp"
#include "unit_file.hpp"
#include <cctype>
#include <iostream>
using namespace strix::glm::rtxpf::producer;
namespace {
std::uint64_t free_bytes(const std::filesystem::path& p){std::error_code ec;auto at=p;while(!at.empty()&&!std::filesystem::exists(at,ec))at=at.parent_path();if(at.empty())return 0;auto s=std::filesystem::space(at,ec);return ec?0:s.available;}
}
int main(int argc,char** argv){try{
    shard_need(argc>=2,"usage: glm_unit_file plan|dry-run|copy|verify --model DIR --out PRIMARY [--stripe DIR:K]... [--chunk-mib 16] [--confirm-bytes N]");
    std::string mode=argv[1],model,out;std::vector<StripeTarget> stripes;std::uint64_t chunk=stripe_chunk_default,confirm=0;
    for(int i=2;i<argc;++i){std::string a=argv[i];shard_need(i+1<argc,"missing option value");std::string v=argv[++i];
        if(a=="--model")model=v;else if(a=="--out")out=v;else if(a=="--stripe")stripes.push_back(parse_stripe(v));
        else if(a=="--chunk-mib"){auto m=std::stoull(v);shard_need(m>=1&&m<=256&&(m&(m-1))==0,"--chunk-mib: power of two 1..256");chunk=m<<20;}
        else if(a=="--confirm-bytes")confirm=std::stoull(v);else throw std::runtime_error("unknown option "+a);}
    shard_need(mode=="plan"||mode=="dry-run"||mode=="copy"||mode=="verify","unknown mode");shard_need(!model.empty()&&!out.empty(),"--model and --out are required");
    // Only copy computes (or authenticates) full source identities; plan/dry-run read headers only.
    Shards source(model,mode=="copy"||mode=="verify");ExpertCatalog catalog(member(source.manifest,"tensors"));
    auto model_sha=member(source.manifest,"model_sha256").as_string();auto plan=plan_unit_file(catalog,out,stripes,chunk);auto report=plan_json(plan,model_sha);
    if(mode=="verify"){std::ifstream m(plan.manifest,std::ios::binary);shard_need(bool(m),"open unit manifest");std::string text((std::istreambuf_iterator<char>(m)),{});
        auto n=verify_unit_file(catalog,plan.primary,strix::serve::parse_json(text),model_sha);report.set("verified_units",std::uint64_t(n));
        report.set("unit_manifest_sha256",strix::model_content_identity_hex(strix::sha256_bytes(std::as_bytes(std::span(text)))));report.set("verdict","VERIFIED");std::cout<<report.dump()<<"\n";return 0;}
    // Source extents: every unit tensor resolves to a selected source tensor of the right size.
    std::vector<std::string> files;std::map<std::string,unsigned> index;std::vector<std::uint64_t> sizes;std::uint64_t logical=0;
    auto locate=[&](const ExpertTensor& t)->SourceExtent{auto& s=source.selected.at(t.name);shard_need(s.bytes==t.bytes,"source tensor size");auto [it,fresh]=index.emplace(s.file,unsigned(files.size()));
        if(fresh){files.push_back(s.file);sizes.push_back(source.sources().file(s.file).bytes);}return {it->second,s.offset,s.bytes,0,0};};
    const auto& g=catalog.geometry();for(unsigned l=g.first_layer;l<=g.last_layer;++l)for(unsigned e=0;e<g.experts;++e){auto x=unit_extents(catalog.at(l,e),locate);for(auto& v:x)logical+=v.bytes;}
    report.set("logical_payload_bytes",logical);report.set("source_files",std::uint64_t(files.size()));
    auto targets=Json::array();bool room=true;
    auto check_target=[&](const std::filesystem::path& p,std::uint64_t need){auto t=Json::object();t.set("path",p.string());t.set("exists",std::filesystem::exists(p));t.set("required_bytes",need);auto f=free_bytes(p);t.set("available_bytes",f);if(f<need||std::filesystem::exists(p))room=false;targets.push(t);};
    check_target(plan.primary,plan.bytes);for(auto& t:plan.stripes)check_target(t.dir/plan.primary.filename(),t.bytes);
    // Sidecars and every reserved .partial must be absent too (exclusive creation, no-replace publication).
    auto outputs=Json::array();for(auto& f:unit_file_outputs(plan)){auto o=Json::object();o.set("path",f.string());o.set("exists",std::filesystem::exists(f));if(std::filesystem::exists(f))room=false;outputs.push(o);}
    report.set("outputs",outputs);report.set("targets",targets);report.set("targets_ready",room);
    if(mode=="plan"||mode=="dry-run"){report.set("mode",mode);report.set("writes","none");std::cout<<report.dump()<<"\n";
        if(mode=="dry-run")for(auto& t:plan.stripes)std::cout<<stripe_ranges(plan,t).dump()<<"\n";return room||mode=="plan"?0:2;}
    shard_need(confirm==plan.bytes,"copy requires --confirm-bytes equal to the dry-run primary_bytes");shard_need(room,"targets exist or lack free space");
    std::vector<const ShardSources::File*> handles;for(auto& f:files)handles.push_back(&source.sources().file(f));
    auto manifest=write_unit_file(catalog,plan,model_sha,locate,sizes,[&](unsigned f,std::uint64_t b,std::uint64_t n,void* d){ShardSources::read_span(*handles[f],b,n,d);});
    // write_unit_file verified every unit and the whole-file digest through its
    // owned handle before publishing. Now publish the authenticated verification
    // record (file identity + USN + digest) through a held deny-write handle so
    // worker startups recall it instead of rehashing the file.
    const auto file_sha=manifest.find("file_sha256")->as_string(),manifest_sha=manifest.find("manifest_sha256")->as_string();
    VerifiedFile held(plan.primary,plan.bytes);held.verify(file_sha,default_hash_cache());
    report.set("verified_units",std::uint64_t(catalog.size()));report.set("unit_file_sha256",file_sha);report.set("unit_manifest_sha256",manifest_sha);
    report.set("launch","--unit-file "+plan.primary.string()+" --unit-manifest-sha256 "+manifest_sha);report.set("mode","copy");report.set("verdict","WRITTEN-VERIFIED");std::cout<<report.dump()<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
