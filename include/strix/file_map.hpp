#pragma once
// Read-only whole-file memory mapping (no hashing, no copy): the host side of tiered expert residency reads the stored
// expert bytes straight from the page cache. Windows CreateFileMapping/MapViewOfFile, POSIX mmap.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix {

class FileMap {
public:
    FileMap() = default;
    explicit FileMap(const std::string& path) { open(path); }
    FileMap(const FileMap&) = delete;
    FileMap& operator=(const FileMap&) = delete;
    FileMap(FileMap&& o) noexcept { swap(o); }
    FileMap& operator=(FileMap&& o) noexcept {
        if (this != &o) { close(); swap(o); }
        return *this;
    }
    ~FileMap() { close(); }

    [[nodiscard]] const std::uint8_t* data() const noexcept { return data_; }
    [[nodiscard]] std::uint64_t size() const noexcept { return size_; }
    [[nodiscard]] bool valid() const noexcept { return data_ != nullptr; }

    void open(const std::string& path) {
        close();
#if defined(_WIN32)
        HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                                  FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
        LARGE_INTEGER bytes{};
        if (!GetFileSizeEx(file, &bytes) || bytes.QuadPart <= 0) { CloseHandle(file); throw std::runtime_error("empty file " + path); }
        HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        CloseHandle(file);
        if (!mapping) throw std::runtime_error("cannot map " + path);
        void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(mapping);  // the view keeps the section alive
        if (!view) throw std::runtime_error("cannot map view of " + path);
        data_ = static_cast<const std::uint8_t*>(view);
        size_ = static_cast<std::uint64_t>(bytes.QuadPart);
#else
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        struct stat st{};
        if (::fstat(fd, &st) != 0 || st.st_size <= 0) { ::close(fd); throw std::runtime_error("empty file " + path); }
        void* view = ::mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (view == MAP_FAILED) throw std::runtime_error("cannot map " + path);
        data_ = static_cast<const std::uint8_t*>(view);
        size_ = static_cast<std::uint64_t>(st.st_size);
#endif
    }
    void close() noexcept {
        if (!data_) return;
#if defined(_WIN32)
        UnmapViewOfFile(data_);
#else
        ::munmap(const_cast<std::uint8_t*>(data_), static_cast<std::size_t>(size_));
#endif
        data_ = nullptr;
        size_ = 0;
    }

private:
    void swap(FileMap& o) noexcept { std::swap(data_, o.data_); std::swap(size_, o.size_); }
    const std::uint8_t* data_ = nullptr;
    std::uint64_t size_ = 0;
};

}  // namespace strix
