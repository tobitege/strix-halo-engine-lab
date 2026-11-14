// Optional complete-workload gate. Real assets are explicit; never a ctest.
#include "strix/device/hip_gate.hpp"
#include "gemma4/engine/runtime.hpp"

#include <chrono>
#include <iostream>
#include <sstream>

namespace e = strix::gemma4::engine;
using Json = strix::serve::Json;
using Clock = std::chrono::steady_clock;
int main(int argc, char** argv) {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    try {
        const auto options = e::parse_options(argc,argv);
        if (options.help) {
            std::cout << "strix-gemma4-gate MODEL --backend device --rope-tables TABLES --draft DRAFT --draft-rope TABLE --tokenizer TOKENIZER --chat REQUEST --generate 256\n"
                "Runs three interleaved auto/OFF rounds, comparing all raw IDs and final committed state. Natural stop IDs are intentionally ignored.\n";
            return 0;
        }
        e::Runtime runtime(options);
        if (options.inspect) { std::cout << e::effective_config(*runtime.backend,options.request,bool(runtime.drafts)).dump() << '\n'; return 0; }
        if (e::effective_config(*runtime.backend,options.request,bool(runtime.drafts)).find("dflash_environment")->as_string() == "on")
            throw std::invalid_argument("auto gate requires STRIX_G4_DFLASH unset; on forces fixed windows");
        if (!runtime.drafts || options.request.max_tokens < 256 || (options.streams != 1 && options.streams != 4) || !strix::gemma4::dflash::enabled_from_environment())
            throw std::invalid_argument("gate requires a drafter, >=256 tokens, C1/C4 and DFlash enabled");
        auto request = options.request;
        if (!options.chat.empty() && runtime.tokenizer) {
            const auto chat = e::parse_chat_request(strix::serve::parse_json_unique(e::read_text(options.chat)),e::DraftMode::automatic,request);
            request = chat.generation;
            request.prompt = strix::gemma4::serve::Adapter(runtime.tokenizer).prepare(chat.input).ids;
        } else if (!options.ids.empty()) {
            std::istringstream input(e::read_text(options.ids)); std::uint64_t id = 0;
            while (input >> id) {
                if (id >= runtime.backend->config().vocab) throw std::invalid_argument("gate prompt ID outside vocabulary");
                request.prompt.push_back(std::uint32_t(id));
            }
            if (!input.eof()) throw std::invalid_argument("invalid gate prompt IDs");
        } else throw std::invalid_argument("gate requires --chat with tokenizer or --ids");
        if (request.max_tokens < 256 || request.sampling.temperature != 0) throw std::invalid_argument("gate requires >=256 greedy tokens");
        request.stop_ids.clear();
        const auto multi = e::MultiStreamConfig::from_environment();
        std::vector<std::uint32_t> reference;
        std::string digest;
        auto warmup = request; warmup.max_tokens = 16; warmup.dflash = e::DraftMode::off;
        { e::Coordinator coordinator(*runtime.backend); coordinator.submit(warmup); coordinator.run();
          if (coordinator.result(0).status == e::Status::failed) throw std::runtime_error(coordinator.result(0).error); }
        for (unsigned round = 0; round < 3; ++round) for (unsigned mode = 0; mode < 2; ++mode) {
            const bool automatic = (round%2 ? 1-mode : mode) != 0;
            auto r = request; r.dflash = automatic ? e::DraftMode::automatic : e::DraftMode::off;
            const auto start = Clock::now();
            e::Coordinator coordinator(*runtime.backend,runtime.drafts);
            std::vector<std::size_t> slots;
            for (unsigned stream = 0; stream < options.streams; ++stream) slots.push_back(coordinator.submit(r));
            coordinator.run();
            const double wall = std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            for (const auto slot : slots) {
                const auto& out = coordinator.result(slot);
                if (out.status != e::Status::length || out.ids.size() != r.max_tokens) throw std::runtime_error("gate incomplete: "+out.error);
                if (reference.empty()) { reference = out.ids; digest = out.digest; }
                const bool equal = reference == out.ids && digest == out.digest;
                auto receipt = coordinator.receipt(slot);
                receipt.set("stream",std::uint64_t(slot));
                receipt.set("round",std::uint64_t(round)); receipt.set("mode",automatic ? "auto" : "off");
                receipt.set("raw_after_stop",true); receipt.set("exact",equal); receipt.set("request_wall_ms",wall);
                receipt.set("decode_tokens_per_second",1000*double(out.ids.size())/out.decode_ms);
                auto ids = Json::array(); for (auto id : out.ids) ids.push(std::uint64_t(id)); receipt.set("ids",ids);
                std::cout << receipt.dump() << std::endl;
                if (!equal) throw std::runtime_error("auto/OFF token or committed-state mismatch");
                if (automatic && options.streams > 1 && !out.multistream_auto_cycles)
                    throw std::runtime_error("multi-stream auto cycles missing");
                if (automatic && options.streams > 1 && !multi.enabled && out.multistream_auto_proposals)
                    throw std::runtime_error("multi-stream auto must remain k=0 with STRIX_G4_DRAFT_MULTI=0");
                if (automatic && options.streams > 1 && out.multistream_auto_max_window > multi.max_k)
                    throw std::runtime_error("multi-stream auto window above the STRIX_G4_DRAFT_MULTI_K cap");
            }
        }
        return 0;
    } catch (const std::exception& error) { std::cerr << "gemma4 gate: " << error.what() << '\n'; return 1; }
}
