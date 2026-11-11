#include "strix/mapped_model_file.hpp"
#include "strix/model_content_hash.hpp"
#include <span>

#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
#if !defined(__linux__) && !defined(_WIN32)
    std::cout
        << "mapped model file tests skipped on this platform\n";
    return 0;
#else
    const auto path =
        std::filesystem::temp_directory_path() /
        "strix-mapped-model-test.gguf";

    std::vector<std::uint8_t> bytes(
        64U * 1024U);
    for (std::size_t index = 0;
         index < bytes.size();
         ++index) {
        bytes[index] =
            static_cast<std::uint8_t>(
                index & 0xffU);
    }

    {
        std::ofstream output(
            path,
            std::ios::binary);
        output.write(
            reinterpret_cast<const char*>(
                bytes.data()),
            static_cast<std::streamsize>(
                bytes.size()));
    }

    strix::ModelManifest manifest;
    manifest.architecture = "fixture";

    strix::ManifestTensor first;
    first.name = "first.weight";
    first.dimensions = {32, 32};
    first.ggml_type = 1;
    first.absolute_offset = 4096;
    first.byte_size = 2048;
    first.role =
        strix::TensorRole::dense_projection;
    manifest.tensors.push_back(first);

    strix::ManifestTensor second;
    second.name = "second.weight";
    second.dimensions = {32, 32};
    second.ggml_type = 1;
    second.absolute_offset = 8192;
    second.byte_size = 2048;
    second.role =
        strix::TensorRole::dense_projection;
    manifest.tensors.push_back(second);

    manifest.total_tensor_bytes =
        first.byte_size +
        second.byte_size;
    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    strix::AllocationGenerationSource generations;
    auto mapped =
        strix::MappedModelFile::open_read_only(
            path,
            "model.gguf",
            generations);

    expect(mapped.valid() &&
               mapped.bytes() ==
                   bytes.size() &&
               mapped.content_identity().valid() &&
               mapped.allocation_generation() == 1,
           "mapped model owns stable read-only file mapping and content identity");

    mapped.validate_unchanged();
    const auto expected_raw=strix::sha256_bytes(std::as_bytes(std::span(bytes)));
    expect(mapped.file_sha256()==expected_raw,"mapped raw file SHA comes from the original full-file pass, not framed content alias");
    std::array<std::byte, 16> direct_read{};
    mapped.read_at(
        123,
        direct_read);
    bool direct_read_matches = true;
    for (std::size_t index = 0;
         index < direct_read.size();
         ++index) {
        direct_read_matches =
            direct_read_matches &&
            std::to_integer<std::uint8_t>(
                direct_read[index]) ==
                bytes[123 + index];
    }
    expect(
        direct_read_matches,
        "mapped model direct read uses the same opened file bytes");

    bool direct_read_range_rejected = false;
    try {
        mapped.read_at(
            mapped.bytes() - 4,
            direct_read);
    } catch (const std::out_of_range&) {
        direct_read_range_rejected = true;
    }
    expect(
        direct_read_range_rejected,
        "mapped model direct read rejects ranges beyond EOF");

    const auto binding =
        mapped.build_runtime_binding(
            manifest);
    expect(binding.tensors.size() == 2 &&
               binding.tensors[0].storage ==
                   strix::RuntimeTensorStorage::file_mapped &&
               binding.tensors[0].address ==
                   mapped.base_address() +
                       first.absolute_offset,
           "manifest tensor resolves directly into mapped file");

    const auto* first_data =
        reinterpret_cast<const std::uint8_t*>(
            binding.tensors[0].address);
    expect(first_data[0] ==
               bytes[first.absolute_offset] &&
               first_data[17] ==
                   bytes[first.absolute_offset + 17],
           "mapped tensor address reads original file bytes");

    auto invalid_manifest = manifest;
    invalid_manifest.tensors[1].absolute_offset =
        mapped.bytes() - 1024;
    invalid_manifest.tensors[1].byte_size =
        2048;
    invalid_manifest.layout_identity =
        strix::model_manifest_layout_identity(
            invalid_manifest);

    bool range_rejected = false;
    try {
        static_cast<void>(
            mapped.build_runtime_binding(
                invalid_manifest));
    } catch (const std::invalid_argument&) {
        range_rejected = true;
    }
    expect(range_rejected,
           "mapped model rejects tensor range outside file");

    auto moved = std::move(mapped);
    expect(!mapped.valid() &&
               moved.valid(),
           "mapped model ownership moves without duplicate unmap");

    expect(!mapped.file_sha256().valid() && moved.file_sha256()==expected_raw,
        "raw file identity transfers with its actual mapping owner");
    moved.validate_unchanged();
    {
        std::fstream changed(path,std::ios::in|std::ios::out|std::ios::binary);
        changed.seekp(0);changed.put('x');changed.flush();
        if(!changed) throw std::runtime_error("cannot mutate mapped identity fixture");
    }
    bool mutation_rejected=false;
    try{moved.validate_unchanged();}catch(const std::runtime_error&){mutation_rejected=true;}
    expect(mutation_rejected,"FD/stat setup witness rejects file mutation after the full-file hash");
    moved = strix::MappedModelFile{}; // Windows does not remove a file that is still mapped
    expect(!moved.valid(),"assigning an empty owner releases the mapping");
    std::error_code error;
    std::filesystem::remove(
        path,
        error);

    if (failures == 0) {
        std::cout << "mapped model file tests passed\n";
    }
    return failures == 0 ? 0 : 1;
#endif
}
