#pragma once
// Gate outputs are new artifacts, never replacements for a reference or a previous candidate.
#include <cstdio>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>

namespace vision_gate {

inline void check_dump_paths(const std::filesystem::path& reference, const std::filesystem::path& dump,
                             const std::string& name) {
    for (const char* suffix : {".rows.f32", ".hidden.f32"}) {
        const auto src = reference / (name + suffix);
        const auto dst = dump / (name + suffix);
        if (std::filesystem::weakly_canonical(src) == std::filesystem::weakly_canonical(dst) ||
            (std::filesystem::exists(src) && std::filesystem::exists(dst) && std::filesystem::equivalent(src, dst)))
            throw std::runtime_error("dump aliases reference: " + dst.string());
        // symlink_status also rejects dangling symlinks; exists alone would follow them.
        if (std::filesystem::symlink_status(dst).type() != std::filesystem::file_type::not_found)
            throw std::runtime_error("dump already exists: " + dst.string());
    }
}

inline void write_dump(const std::filesystem::path& path, std::span<const float> values) {
    // Exclusive creation closes the check/open race and refuses hard links and symlinks as well as ordinary files.
    std::FILE* file = nullptr;
#ifdef _MSC_VER
    (void)_wfopen_s(&file, path.c_str(), L"wbx");
#else
    file = std::fopen(path.c_str(), "wbx");
#endif
    if (!file) throw std::runtime_error("cannot create new dump: " + path.string());
    const bool wrote = std::fwrite(values.data(), sizeof(float), values.size(), file) == values.size();
    const bool closed = std::fclose(file) == 0;
    if (!wrote || !closed) throw std::runtime_error("cannot write dump: " + path.string());
}

}  // namespace vision_gate
