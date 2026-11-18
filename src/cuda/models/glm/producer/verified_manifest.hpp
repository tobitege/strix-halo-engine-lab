#pragma once
#include "verified_io.hpp"
#include "strix/serve/json.hpp"
namespace strix::glm::rtxpf::producer {
// Shared CPU ingestion path for Runtime and the integrity tool. The parsed
// object has no pathname dependency after construction; retain the source lock.
class VerifiedManifest {
    VerifiedFile file_;
public:
    const strix::serve::Json parsed;
    VerifiedManifest(const std::filesystem::path& path,const std::string& expected):
        file_(path,std::filesystem::file_size(path)),
        parsed(strix::serve::parse_json(file_.verified_contents(expected))){}
};
}
