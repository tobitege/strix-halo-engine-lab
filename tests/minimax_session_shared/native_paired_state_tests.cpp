// Explicit operator-only native driver. Not registered with CTest and excluded
// from ALL. Both ranks must run this instead of an HTTP/named-session owner.
#include "minimax/serve/serve_runner.hpp"
#include "minimax/engine/mm_defaults.hpp"
#include "strix/dist/open_transport.hpp"
#include <charconv>
#include <fstream>
#include <iostream>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <cstdlib>

static std::uint64_t number(const char* text,std::uint64_t maximum) {
    const std::string_view value(text);std::uint64_t out=0;
    const auto parsed=std::from_chars(value.data(),value.data()+value.size(),out);
    if(parsed.ec!=std::errc{} || parsed.ptr!=value.data()+value.size() || out>maximum)
        throw std::invalid_argument("invalid native probe integer");
    return out;
}
int main(int argc,char** argv) try {
    // Hard process wall bound includes loading, transport, native work and all
    // waits. Timeout exits 124 and is NEVER a qualification pass.
    struct Watchdog {
        std::mutex mutex;std::condition_variable wake;bool done=false;
        std::thread worker{[this]{std::unique_lock lock(mutex);
            if(!wake.wait_for(lock,std::chrono::seconds(120),[this]{return done;}))std::_Exit(124);}};
        ~Watchdog(){{std::lock_guard lock(mutex);done=true;}wake.notify_all();worker.join();}
    } watchdog;
    if(argc!=8 && argc!=9)throw std::invalid_argument(
        "usage: minimax-native-paired-state MODEL_DIR R TRANSPORT PROMPT.u32 HOST_BUDGET CAPACITY MAX_ROWS [state|input|fault0|fault1]");
    const std::string_view ticket=argc==9?argv[8]:"state";
    if(ticket!="state" && ticket!="input" && ticket!="fault0" && ticket!="fault1")throw std::invalid_argument("native ticket");
    namespace mm=strix::minimax::engine;
    mm::Options options;options.model_dir=argv[1];options.rank=static_cast<std::uint32_t>(number(argv[2],1));
    options.slots=3;options.capacity=number(argv[6],UINT32_MAX);
    options.max_rows=static_cast<std::uint32_t>(number(argv[7],4096));
    const auto budget=number(argv[5],UINT64_MAX-1);
    if(!budget || !options.max_rows || options.capacity<8)throw std::invalid_argument("native probe limits");
    // Read a bounded complete little-endian fixture before transport/device work.
    std::ifstream input(argv[4],std::ios::binary);std::vector<std::uint32_t> prompt;
    for(std::array<unsigned char,4> bytes;input.read(reinterpret_cast<char*>(bytes.data()),4);) {
        std::uint32_t token=0;for(unsigned j=0;j<4;++j)token|=std::uint32_t(bytes[j])<<(j*8);
        if(token>=200064 || prompt.size()+5>=options.capacity || prompt.size()>=options.max_rows)
            throw std::invalid_argument("native probe fixture range");
        prompt.push_back(token);
    }
    if(!input.eof() || input.gcount() || prompt.empty())throw std::invalid_argument("native probe incomplete fixture");
    if(ticket!="state" && (prompt.size()>8 || prompt.size()+8>=options.capacity || options.max_rows<8))
        throw std::invalid_argument("input tickets require <=8 input IDs, context headroom and MAX_ROWS>=8");
    strix::minimax::defaults::apply();
    strix::dist::OpenTransportOptions open;open.spellings=strix::dist::hyphen_transport;
    open.usb.rank=options.rank;open.usb.transport_capabilities=0;
    auto transport=strix::dist::open_transport(argv[3],open);
    strix::dist::Group group;group.rank=options.rank;group.world=2;group.plan=0x4d4d5331ULL;group.transport_capabilities=0;
    const auto maximum=std::max<std::size_t>(std::size_t{options.max_rows}*6144,256*782*128)*sizeof(float);
    auto comm=transport.direct
        ?std::make_unique<strix::dist::Communicator>(group,std::move(transport.host),std::move(transport.direct),maximum)
        :std::make_unique<strix::dist::Communicator>(group,std::move(transport.host),maximum);
    mm::Engine engine(options,*comm);(void)engine.load();engine.set_taps(false);
    strix::minimax::serve::native::Runner runner{engine,*comm};
    const auto result=strix::minimax::serve::native::paired_state_probe(engine,runner,options,"/proc/self/exe",budget,prompt,ticket);
    std::cout<<"rank "<<options.rank<<" bounded operator ticket completed; wider native gates remain\n";
    return result;
} catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
