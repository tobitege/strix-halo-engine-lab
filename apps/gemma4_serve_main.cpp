#include "strix/device/hip_gate.hpp"
#include "gemma4/engine/api.hpp"
#include "gemma4/engine/runtime.hpp"
#include "gemma4/engine/deployment.hpp"
#include <cstdlib>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    namespace e = strix::gemma4::engine;
    try {
        auto prepared=e::prepare_serve(argc,argv);
        if(prepared.exit_code){std::cout<<prepared.output;return *prepared.exit_code;}
        auto& options=prepared.options;
        auto& deployment=prepared.deployment;
        e::Runtime runtime(options);
        if (options.inspect) { std::cout << e::effective_config(*runtime.backend,options.request,bool(runtime.drafts),options.streams).dump() << '\n'; return 0; }
        if (!runtime.tokenizer) throw std::invalid_argument("serve requires --tokenizer");
        // The deployment plan's slot count reaches the API explicitly (0: STRIX_G4_SERVE_SLOTS as without a plan).
        e::Api api(*runtime.backend,runtime.tokenizer,runtime.drafts,options.request.dflash,"nemotron-bolt-e4b",options.request,deployment,
                   e::Api::Owner::standalone,{},prepared.deployment_slots);
        strix::serve::HttpServerConfig config; config.bind_address = options.bind; config.port = options.port;
        strix::serve::HttpServer server(config,[&](const auto& request, auto& writer) { api.handle(request,writer); });
        server.listen(); std::cerr << "Gemma 4 service ready\n"; server.serve();
        return 0;
    } catch (const std::exception& error) { std::cerr << "gemma4 serve: " << error.what() << '\n'; return 1; }
}
