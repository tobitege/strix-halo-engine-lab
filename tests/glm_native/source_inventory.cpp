// CPU-only synthetic source inventory probe. No schema bypass in the service.
#define STRIX_HASH_AUTHORITY_TESTING
#include "cuda/models/glm/producer/shards.hpp"
#include <iostream>
using namespace strix::glm::rtxpf::producer;
int main(int argc,char** argv){try{
    shard_need(argc>=3,"usage: source_inventory MODEL HASH_CACHE [--no-journal|--authority-failure]");
    if(argc==4){
        auto option=std::string(argv[3]);
        if(option=="--no-journal")journal_unavailable=true;
        else if(option=="--authority-failure")authority_fault=[](const char*){throw std::runtime_error("injected authority failure");};
        else throw std::runtime_error("unknown test option");
    }
    auto config_path=std::filesystem::path(argv[1])/"config.json";
    VerifiedFile config_file(config_path,std::filesystem::file_size(config_path));
    ShardSources sources(argv[1]);auto config=strix::serve::parse_json(config_file.contents());
    auto manifest=shard_content_manifest(config,sources.records(true,argv[2]));
    std::uint64_t read=0;unsigned no_journal=0;
    for(auto& file:sources.files){read+=file.held->bytes_read();if(!file.held->journal_available())++no_journal;
        WinHandle writer;writer.h=CreateFileW(file.path.c_str(),GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,0,nullptr);
        shard_need(writer.h==INVALID_HANDLE_VALUE&&GetLastError()==ERROR_SHARING_VIOLATION,"source must deny writes");
        shard_need(!DeleteFileW(file.path.c_str())&&GetLastError()==ERROR_SHARING_VIOLATION,"source must deny deletes");
    }
    auto result=Json::object();result.set("manifest",manifest);result.set("bytes_read",read);result.set("verified_hash_hits",int(sources.verified_hash_hits));result.set("journal_unavailable_files",int(no_journal));
    std::cout<<result.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
