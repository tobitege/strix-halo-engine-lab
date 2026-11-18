#include "glm/load/glm_rank_plan.hpp"
#include "strix/weight/admission.hpp"
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main() try {
    const auto directory=std::filesystem::temp_directory_path()/("wfmt-glm-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);const auto file=directory/"model.safetensors";
    struct Cleanup {std::filesystem::path file;~Cleanup(){std::error_code e;std::filesystem::remove(file,e);std::filesystem::remove(file.parent_path(),e);}} cleanup{file};
    const std::string header=R"({"layer.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
    {std::ofstream out(file,std::ios::binary);const auto n=std::uint64_t(header.size());for(unsigned i=0;i<8;++i)out.put(char((n>>(8*i))&255));out.write(header.data(),std::streamsize(header.size()));float one=1;out.write(reinterpret_cast<const char*>(&one),4);}
    strix::weight::WeightMap retained;
    {
        strix::glm::load::Index index(directory.string());const auto* legacy=index.find("layer.weight");
        if(!legacy || legacy->dtype!="F32" || legacy->shape!=std::vector<std::uint64_t>{1} || legacy->offset!=8+header.size() || legacy->bytes!=4 || index.files().size()!=1)
            throw std::runtime_error("GLM legacy tensor header/range changed");
        retained=index.weight_map();
        if(retained.size()!=1 || retained[0].weight.payload.range.offset!=legacy->offset || retained[0].weight.payload.range.bytes!=legacy->bytes)
            throw std::runtime_error("GLM shared map differs from legacy operand range");
    }
    std::array<std::byte,4> bytes{};auto status=strix::weight::read_slice(retained[0].weight.payload,bytes);float one=1;
    if(!status || std::memcmp(bytes.data(),&one,4))throw std::runtime_error("GLM source ownership or bytes changed");
    std::cout<<"GLM legacy/common range and bytes PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
