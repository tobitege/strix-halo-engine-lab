#pragma once

#include "strix/model_description.hpp"

#include <string>
#include <vector>

// models/registry.toml: one table per supported model (description file, engine family, profile tag,
// defaults tables, entry points). The shared serve shell reads it for selection;
// numerical behavior and production defaults remain in the compiled adapters.
namespace strix::model_registry {

struct Entry {
    std::string id;                      // table key
    std::string name;                    // equals the description's `name`
    std::string description;             // repo-relative path of models/<name>.toml
    std::string family;                  // engine family
    std::string cmake;                   // CMake option building the family; empty: always built
    std::string profile_tag;             // profile::Engine tag; empty: the engine has no pair profile table
    std::string allow_pending_env;
    std::vector<std::string> defaults;   // source files holding the engine's switch/defaults tables
    std::vector<std::string> apps;       // entry points
    std::vector<std::string> adapter;    // family adapter validating the description
};

[[nodiscard]] std::vector<Entry> parse(const model_description::ModelDescription& file);
[[nodiscard]] std::vector<Entry> load(const std::string& path);
// Every problem found checking `entries` against the tree rooted at `root` (empty: all consistent).
[[nodiscard]] std::vector<std::string> check(const std::vector<Entry>& entries, const std::string& root);

} // namespace strix::model_registry
