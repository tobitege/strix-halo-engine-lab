#pragma once

// Shared serve configuration file (--serve-config FILE, TOML). One expander for every backend and family: a binary
// declares the options its own CLI parser takes (key -> flag, kind) and the file becomes argv in front of the CLI,
// so the binary's parser and its single resolver see exactly what an equivalent command line gives them.
//  - validated before any weight is read: syntax and duplicate keys (with line/column), unknown keys and wrong types
//    (with line), inline secrets, prefill settings given both here and through --config, untrusted files (POSIX);
//  - the CLI wins: a flag given on the command line drops the file's value; a list given on the command line
//    replaces the file's whole list;
//  - relative paths become absolute against the directory of the file;
//  - -- ends expansion/precedence/inspection; the remaining family arguments stay verbatim;
//  - --print-serve-config prints the effective options (file + CLI, with their origin) and exits before loading.

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::serve::config_file {

enum class Kind : std::uint8_t {
    flag,    // bool: true -> bare flag, false -> nothing (the flag is off by default)
    on_off,  // bool -> "on" / "off"
    value,   // string, integer or float -> one value
    list,    // array -> one value joined by Option::join, or the flag repeated per item when join == 0
    path,    // string path, relative to the file's directory
    secret,  // an inline credential: refused in the file (use the *_file form); redacted when printed from the CLI
};

struct Option {
    std::string_view key;              // TOML key, dotted for a table member ("prefill.disable")
    std::string_view flag;             // the CLI flag the binary's parser takes
    Kind kind = Kind::value;
    char join = 0;
    std::string_view conflicts = {};   // a CLI flag that sets the same settings another way; both -> error
};

struct Expanded {
    bool used = false;                 // --serve-config was given
    bool print = false;                // --print-serve-config was given
    std::vector<std::string> argv;     // leading positionals, file options, then the CLI options
    std::string effective;             // TOML of every declared option set by the file or the CLI, with its origin
};

// lead: argv[0] plus the positionals before the first option stay in front. Throws std::invalid_argument
// ("FILE:LINE: ...") on any refusal; never returns a partially applied configuration.
[[nodiscard]] Expanded expand(int argc, const char* const* argv, std::span<const Option> table);
// The same with the file text given (host tests; file_path still anchors relative paths and messages).
[[nodiscard]] Expanded expand_text(std::string_view text, const std::string& file_path, const std::vector<std::string>& args,
    std::span<const Option> table);

}  // namespace strix::serve::config_file
