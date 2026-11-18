// Included only in the diagnostic test executable. Observes production buffers;
// it never reconstructs an intermediate or changes the production kernel.
#include "strix/cuda/qwen_quant.cuh"
#include <fstream>
#include <iomanip>
#include <sstream>

static void wfmt_record(const std::filesystem::path& directory, const std::string& name,
                        const char* arithmetic, u32 T,
                        const std::vector<std::pair<std::string,std::vector<u8>>>& sections) {
    std::ostringstream header;
    header << "{\"case\":\"" << name << "\",\"arithmetic\":\"" << arithmetic
           << "\",\"shape\":[" << T << ",2560,640],\"sections\":[";
    bool first=true;
    for (const auto& [key,data] : sections) {
        u64 hash=14695981039346656037ull;
        for (auto byte:data) { hash^=byte; hash*=1099511628211ull; }
        if (!first) header << ',';
        first=false;
        header << "{\"name\":\"" << key << "\",\"bytes\":" << std::dec << data.size()
               << ",\"fnv1a64\":\"" << std::hex << std::setw(16) << std::setfill('0') << hash << "\"}";
    }
    header << "]}";
    const auto text=header.str();
    std::filesystem::create_directories(directory);
    std::ofstream out(directory/(name+".wfmt"),std::ios::binary);
    out.write("WFMT\0\1\r\n",8);
    for(unsigned i=0;i<8;++i) out.put(char((u64(text.size())>>(8*i))&255));
    out.write(text.data(),std::streamsize(text.size()));
    for(const auto& [key,data]:sections) out.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()));
    if(!out) throw std::runtime_error("MoE capture write failed");
}

static HostWeights wfmt_native_weights(u32 gt,u32 dt,std::mt19937& rng) {
    HostWeights w{gt,dt,3,{},{},{}};
    const auto fill=[&](std::vector<u8>& bytes,u32 type,u32 n,u32 k) {
        const auto geo=strix::cuda::quant::geometry(type);
        if(!geo.block_elements || k%geo.block_elements) throw std::invalid_argument("capture shape");
        bytes.resize(size_t(3)*n*(k/geo.block_elements)*geo.block_bytes);
        for(auto& x:bytes) x=u8(rng());
        // Small finite scales keep the graph's intermediate arithmetic in range.
        for(size_t off=0;off<bytes.size();off+=geo.block_bytes) {
            const u16 d=half_of(0x1p-10F);
            std::memcpy(bytes.data()+off,&d,2);
        }
    };
    fill(w.gate,gt,mo::ffn,mo::hidden); fill(w.up,gt,mo::ffn,mo::hidden);
    fill(w.down,dt,mo::hidden,mo::ffn);
    return w;
}

static void wfmt_capture_case(const std::filesystem::path& directory,const am::Tables& tables,
                              const HostWeights& w,u32 T,bool fast,std::mt19937& rng) {
    State state; init_state(state,T); synthetic_inputs(state,2,1|2|4,rng);
    Device d; upload_weights(d,w); upload_state(d,state,true);
    mo::enqueue(d.rows_view,T,mo::hidden,d.weights,tables,d.record,nullptr,fast);
    CK(cudaDeviceSynchronize());
    std::vector<std::pair<std::string,std::vector<u8>>> sections;
    auto append=[&](const char* key,const std::vector<u8>& data) {
        auto found=std::find_if(sections.begin(),sections.end(),[&](const auto& s){return s.first==key;});
        if(found==sections.end()) { sections.push_back({key,data}); return; }
        found->second.insert(found->second.end(),data.begin(),data.end());
    };
    append("source",w.gate); append("source",w.up); append("source",w.down);
    for(const int v:{V_input,V_routes,V_rw}) append("state",state.view[v]);
    for(const int v:{V_codes,V_dcodes}) append("activation_codes",snap(d,state,v));
    for(const int v:{V_deltas,V_sums,V_ddeltas,V_dsums}) append("activation_scales",snap(d,state,v));
    append("partials",snap(d,state,V_inter)); append("partials",snap(d,state,V_raw));
    append("output",snap(d,state,V_out)); append("output",snap(d,state,V_acc));
    append("state",snap(d,state,V_status));
    std::vector<u8> summary(128); CK(cudaMemcpy(summary.data(),d.summary,summary.size(),cudaMemcpyDeviceToHost));
    append("state",summary);
    std::sort(sections.begin(),sections.end(),[](const auto& a,const auto& b){return a.first<b.first;});
    const bool native=mo::native_types(d.weights);
    const std::string name="moe-g"+std::to_string(w.gate_type)+"-d"+std::to_string(w.down_type)+"-m"+std::to_string(T)+(fast?"-fast":"-exact");
    wfmt_record(directory,name,native?"cuda-native-iq-fp32@1":fast?"cuda-moe-mma@1":"cuda-four-chain-imma@1",T,sections);
    free_device(d);
}

static void wfmt_capture(const am::Tables& tables,const char* directory) {
    std::mt19937 rng(0x11ca);
    unsigned count=0;
    for(const auto gt:{mo::type_q4_k,mo::type_q5_k}) for(const auto dt:{mo::type_q5_1,mo::type_q8_0}) {
        const auto w=synthetic_weights(gt,dt,3,rng);
        for(const auto T:{1U,8U,9U,33U,65U}) for(const bool fast:{false,true}) {
            wfmt_capture_case(directory,tables,w,T,fast,rng); ++count;
        }
    }
    for(const auto gt:{18U,21U,22U,23U,42U}) for(const auto dt:{20U,42U}) {
        const auto w=wfmt_native_weights(gt,dt,rng);
        for(const auto T:{1U,9U}) { wfmt_capture_case(directory,tables,w,T,false,rng); ++count; }
    }
    std::printf("WFMT MoE capture: %u records\n",count);
}
