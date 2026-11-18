// CPU-only cache verification/prewarm. Never links a CUDA library.
#include "verified_manifest.hpp"
#include "strix/serve/json.hpp"
#include <chrono>
#include <iostream>
#include <memory>
using namespace strix::glm::rtxpf::producer;
int main(int argc,char** argv){try{
    io_need(argc==3,"usage: glm_native_integrity CACHE_DIR MANIFEST_SHA256");
    const auto begin=std::chrono::steady_clock::now();auto directory=std::filesystem::path(argv[1]);auto path=directory/"manifest.json";
    auto cache=default_hash_cache();VerifiedManifest manifest_file(path,argv[2]);
    const auto& manifest=manifest_file.parsed;io_need(manifest.find("format")&&manifest.find("format")->as_string()=="glm-native-cache-v1","cache format");
    struct Job {std::unique_ptr<VerifiedFile> file;std::string sha;};std::vector<Job> jobs;std::uint64_t bytes=0;
    io_need(manifest.find("blobs")!=nullptr,"cache blobs");
    for(auto& [name,blob]:manifest.find("blobs")->as_object()){
        io_need(!name.empty()&&std::filesystem::path(name).filename()==name,"cache path traversal");
        auto* size=blob.find("bytes");auto* sha=blob.find("sha256");io_need(size&&sha&&size->as_int()>=0,"blob fields");
        auto n=std::uint64_t(size->as_int());jobs.push_back({std::make_unique<VerifiedFile>(directory/name,n),sha->as_string()});bytes+=n;
    }
    std::atomic<unsigned> hits=0,no_journal=0;parallel_hash_files(jobs.size(),[&](std::size_t j){if(!jobs[j].file->journal_available())++no_journal;if(jobs[j].file->verify(jobs[j].sha,cache))++hits;});
    auto result=strix::serve::Json::object();result.set("verdict","VERIFIED");result.set("blobs",int(jobs.size()));result.set("verified_hash_hits",int(hits.load()));result.set("bytes",bytes);
    result.set("journal_unavailable_blobs",int(no_journal.load()));
    result.set("hash_workers",int(std::max(1u,std::thread::hardware_concurrency())));result.set("elapsed_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());
    std::cout<<result.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
