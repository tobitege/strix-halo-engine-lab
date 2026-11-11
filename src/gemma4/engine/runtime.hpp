#pragma once

#include "host_backend.hpp"
#include "api.hpp"
#include "strix/profiles/gemma4.hpp"
#include "gemma4/eagle/session.hpp"
#include "gemma4/eagle/mtp/session.hpp"

namespace strix::gemma4::engine {

struct Options {
    std::filesystem::path model, rope, tokenizer, draft, draft_rope, draft_eagle, draft_mtp, chat, ids;
    std::filesystem::path exl3_arm; // explicit changed-weight identity; model supplies BF16 exemptions
    std::vector<std::filesystem::path> fixtures;
#ifdef STRIX_GEMMA4_DEVICE_BACKEND
    static constexpr const char* default_backend = "device"; // GPU builds run on the GPU unless --backend host
#else
    static constexpr const char* default_backend = "host";
#endif
    std::string prompt, output = "json", backend = default_backend, bind = "127.0.0.1";
    std::uint32_t threads = 16, streams = 1;
    std::string deploy;
    bool deploy_check = false;
    std::uint32_t deployment_context = 0; // zero preserves the native model limit
    std::uint32_t deploy_node = 0;
    std::uint16_t port = 8080;
    Request request;
    profile::Resolution profile; // --profile default|fast|quality, resolved and exported by parse_options
    bool inspect = false, help = false;
};
Options parse_options(int argc, char** argv);
std::string read_text(const std::filesystem::path& path);
class Runtime {
public:
    explicit Runtime(const Options& options);
    ModelManifest manifest;
    OwnedRopeTables rope;
    std::unique_ptr<dflash::DraftFile> draft;
    dflash::RopeFile draft_rope;
    std::unique_ptr<eagle::Weights> eagle_draft; // --draft-eagle, or --draft with STRIX_G4_DRAFT_KIND=eagle
    std::unique_ptr<eagle::mtp::Weights> mtp_draft; // --draft-mtp DIR (HF assistant layout), or --draft DIR with KIND=mtp
    std::shared_ptr<const gemma4::serve::Tokenizer> tokenizer;
    std::unique_ptr<Backend> backend;
    DraftFactory drafts;
};
// Optional device library implements this factory and engine::Backend/StreamState.
// Model/tables outlive it; begin() returns its own KvTxn recognized by forward_rows.
#ifdef STRIX_GEMMA4_DEVICE_BACKEND
std::unique_ptr<Backend> make_device_backend(const ModelManifest& model, RopeTables tables, bool inspect = false);
DraftFactory device_drafts(const dflash::Weights& weights, dflash::RopeView rope, LinearView embedding,
    Backend* backend = nullptr, bool prepare = true, LinearView head = {});
#endif

} // namespace strix::gemma4::engine
