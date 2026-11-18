#pragma once
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

struct TeacherRowsDirectory {
    std::filesystem::path path;
    TeacherRowsDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned attempt = 0; attempt < 1000U; ++attempt) {
            auto candidate = std::filesystem::current_path() /
                ("strix-teacher-rows-" + std::to_string(stamp) + "-" + std::to_string(attempt));
            if (std::filesystem::create_directory(candidate)) { path = std::move(candidate); return; }
        }
        throw std::runtime_error("cannot create unique teacher rows directory");
    }
    TeacherRowsDirectory(const TeacherRowsDirectory&) = delete;
    TeacherRowsDirectory& operator=(const TeacherRowsDirectory&) = delete;
    ~TeacherRowsDirectory() {
        std::error_code ignored;
        std::filesystem::remove(path / "rows.txt", ignored);
        std::filesystem::remove(path, ignored);
    }
};
