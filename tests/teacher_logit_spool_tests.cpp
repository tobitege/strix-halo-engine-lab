#include "strix/teacher_logit_spool.hpp"
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class F>void rejects(F function){bool caught=false;try{function();}
    catch(const std::exception&){caught=true;}require(caught,"invalid spool operation accepted");}
}
int main(){
    const auto filename="strix-logit-spool-"+std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count())+".bin";
    const auto path=std::filesystem::temp_directory_path()/filename;
    try {
        const std::array<std::uint32_t,3> tokens{1U,2U,0U};
        const std::array<float,3> first{0.0F,-2.0F,3.0F},second{1.0F,0.0F,-1.0F};
        {
            strix::TeacherLogitSpool file(path.string(),3U,tokens);
            rejects([&]{strix::TeacherLogitSpool duplicate(path.string(),3U,tokens);});
            rejects([&]{file.finish();});
            rejects([&]{file.append(2U,0U,first);});
            rejects([&]{file.append(1U,1U,first);});
            auto nonfinite=first;nonfinite[2]=std::numeric_limits<float>::infinity();
            rejects([&]{file.append(1U,0U,nonfinite);});
            file.append(1U,0U,first);
            file.append(2U,1U,second);
            rejects([&]{file.append(0U,2U,second);});
            file.finish();
            rejects([&]{file.finish();});
        }
        std::ifstream stream(path,std::ios::binary);
        const std::string bytes{std::istreambuf_iterator<char>{stream},{}};
        stream.close();
        require(bytes.size()==108U,"spool row/header/footer extent differs");
        require(bytes.substr(0U,8U)=="TMLLOG01" && bytes.substr(92U,8U)=="TMLEOF01",
            "spool completion framing differs");
        const auto u32=[&](std::size_t offset){std::uint32_t result=0U;
            for(unsigned i=0;i<4U;++i)result|=static_cast<std::uint32_t>(
                static_cast<unsigned char>(bytes.at(offset+i)))<<(8U*i);return result;};
        require(u32(8U)==3U && u32(16U)==2U && u32(24U)==3U,"spool retained dimensions differ");
        require(u32(32U)==1U && u32(36U)==2U && u32(40U)==0U,"spool teacher sequence differs");
        require(u32(44U)==1U && u32(68U)==2U,"spool row inputs differ");
        require(u32(56U)==std::bit_cast<std::uint32_t>(first[0]) &&
            u32(64U)==std::bit_cast<std::uint32_t>(first[2]) &&
            u32(88U)==std::bit_cast<std::uint32_t>(second[2]),"spool full vocabulary order/bits differ");
        std::filesystem::remove(path);
        const std::array<std::uint32_t,2> invalid{0U,3U};
        rejects([&]{strix::TeacherLogitSpool file(path.string(),3U,invalid);});
        require(!std::filesystem::exists(path),"invalid spool created an output");
        // Row selection: every row is appended and checked, only the selected positions are written.
        const std::array<std::uint32_t,4> four{1U,2U,0U,1U};
        const std::array<std::uint64_t,2> repeated{1U,1U};
        const std::array<std::uint64_t,1> outside{3U},middle{1U};
        rejects([&]{strix::TeacherLogitSpool file(path.string(),3U,four,repeated);});
        rejects([&]{strix::TeacherLogitSpool file(path.string(),3U,four,outside);});
        require(!std::filesystem::exists(path),"invalid selection created an output");
        {
            strix::TeacherLogitSpool file(path.string(),3U,four,middle);
            rejects([&]{file.append(2U,1U,second);});
            file.append(1U,0U,first);
            file.append(2U,1U,second);
            rejects([&]{file.finish();});
            file.append(0U,2U,first);
            file.finish();
        }
        std::ifstream selected_stream(path,std::ios::binary);
        const std::string kept{std::istreambuf_iterator<char>{selected_stream},{}};
        selected_stream.close();
        const auto at=[&](std::size_t offset){std::uint32_t result=0U;
            for(unsigned i=0;i<4U;++i)result|=static_cast<std::uint32_t>(
                static_cast<unsigned char>(kept.at(offset+i)))<<(8U*i);return result;};
        require(kept.size()==96U && kept.substr(0U,8U)=="TMLLOG02" && kept.substr(80U,8U)=="TMLEOF01",
            "selected spool framing differs");
        require(at(16U)==1U && at(24U)==4U && at(48U)==1U && at(56U)==2U && at(60U)==1U && at(88U)==1U,
            "selected spool header/row/footer differ");
        require(at(68U)==std::bit_cast<std::uint32_t>(second[0]) && at(76U)==std::bit_cast<std::uint32_t>(second[2]),
            "selected spool row bits differ");
        std::filesystem::remove(path);
        std::cout<<"teacher logit spool tests passed\n";
    }catch(const std::exception& error){std::filesystem::remove(path);std::cerr<<error.what()<<'\n';return 1;}
    return 0;
}
