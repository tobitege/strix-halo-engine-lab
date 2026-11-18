// Host-only production virtual-cache reader. No CUDA imports or device work.
#include "cuda/models/glm/producer/shards.hpp"
#include <iostream>
#include <chrono>
using namespace strix::glm::rtxpf::producer;
int main(int argc,char** argv){try{
    shard_need(argc>=3,"usage: shards MODEL --inventory | BLOB...");
    auto begin=std::chrono::steady_clock::now();
    // Real-shard equivalence tests deliberately avoid rehashing all 166 GB.
    // Service construction always computes or authenticates source digests.
    bool hash=std::string(argv[2])=="--hash-inventory";Shards source(argv[1],hash);
    auto inventory=Json::object();inventory.set("sources",member(source.manifest,"sources"));inventory.set("model_sha256",member(source.manifest,"model_sha256"));inventory.set("cache_key",member(source.manifest,"cache_key"));inventory.set("tensors",std::uint64_t(source.selected.size()));inventory.set("blobs",member(source.manifest,"blobs"));inventory.set("inventory_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());
    std::cout<<inventory.dump()<<"\n";
    std::vector<std::byte> staging(8<<20);
    for(int i=2;i<argc;++i){std::string blob=argv[i];if(blob=="--inventory"||blob=="--hash-inventory")continue;strix::Sha256Fast sha;std::uint64_t bytes=0;begin=std::chrono::steady_clock::now();
        source.stream(blob,staging.data(),staging.size(),[&](std::uint64_t offset,const void* p,std::size_t n){shard_need(offset==bytes,"noncontiguous virtual blob");sha.update({static_cast<const std::byte*>(p),n});bytes+=n;});
        auto result=Json::object();result.set("blob",blob);result.set("bytes",bytes);result.set("sha256",strix::model_content_identity_hex(sha.finalize()));result.set("read_pack_hash_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());std::cout<<result.dump()<<"\n";
    }return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
