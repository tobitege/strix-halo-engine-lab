#pragma once
#include "strix/serve/run.hpp"
#include "api.hpp"
namespace strix::gemma4::engine {
struct Options;
struct SessionAssets {
    std::shared_ptr<const void> owner;
    Backend* backend = nullptr;
    std::shared_ptr<const gemma4::serve::Tokenizer> tokenizer;
    DraftFactory drafts;
};
// A loader seam permits deterministic host construction without official
// assets. Empty selects the production Runtime, with the same owned lifetime.
using SessionAssetLoader = std::function<SessionAssets(const Options&)>;
void install_session_service(strix::serve::ServiceFactories&, SessionAssetLoader);
void install_session_service(strix::serve::ServiceFactories&);
}
