#include "strix/device/hip_gate.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/mapped_model_file.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/runtime_allocator.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

const char* storage_name(
    strix::RuntimeTensorStorage storage) {
    using strix::RuntimeTensorStorage;
    switch (storage) {
    case RuntimeTensorStorage::resident:
        return "resident";
    case RuntimeTensorStorage::file_mapped:
        return "file_mapped";
    case RuntimeTensorStorage::registered_file_mapped:
        return "registered_file_mapped";
    case RuntimeTensorStorage::repacked_cache:
        return "repacked_cache";
    case RuntimeTensorStorage::remote:
        return "remote";
    }
    return "unknown";
}

}  // namespace

int main(int argc, char** argv) try {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    if (argc < 3) {
        throw std::invalid_argument(
            "usage: strix-hip-model-residency MODEL.gguf register|copy "
            "[--promote-file-backed] [--coalesce-gap BYTES]");
    }

    strix::HipModelResidencyPolicy policy;
    const std::string_view mode = argv[2];
    if (mode == "register") {
        policy.mode =
            strix::HipModelResidencyMode::
                registered_file_mapping;
    } else if (mode == "copy") {
        policy.mode =
            strix::HipModelResidencyMode::
                resident_copy;
    } else {
        throw std::invalid_argument(
            "residency mode must be register or copy");
    }

    for (int index = 3; index < argc; ++index) {
        const std::string_view arg = argv[index];
        if (arg == "--promote-file-backed") {
            policy.promote_file_backed_candidates = true;
        } else if (arg == "--coalesce-gap") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "--coalesce-gap requires a byte count");
            }
            policy.coalesce_gap_bytes =
                std::stoull(argv[++index]);
        } else {
            throw std::invalid_argument(
                "unknown residency argument: " +
                std::string(arg));
        }
    }

    strix::AllocationGenerationSource generations;
    const auto info =
        strix::gguf::read_file(argv[1]);
    const auto manifest =
        strix::build_model_manifest(info);
    auto mapped =
        strix::MappedModelFile::open_read_only(
            argv[1],
            "model",
            generations);
    auto residency =
        strix::HipModelResidency::create(
            mapped,
            manifest,
            generations,
            policy);

    std::uint64_t gpu_visible = 0;
    std::uint64_t file_backed = 0;
    std::uint64_t remote = 0;
    std::uint64_t bytes_gpu_visible = 0;
    std::uint64_t bytes_file_backed = 0;

    for (std::size_t index = 0;
         index < residency.binding().tensors.size();
         ++index) {
        const auto& tensor =
            residency.binding().tensors[index];
        const auto bytes =
            manifest.tensors[index].byte_size;
        if (strix::runtime_tensor_gpu_visible(
                tensor.storage)) {
            ++gpu_visible;
            bytes_gpu_visible += bytes;
        }
        if (strix::runtime_tensor_file_backed(
                tensor.storage)) {
            ++file_backed;
            bytes_file_backed += bytes;
        }
        if (tensor.storage ==
            strix::RuntimeTensorStorage::remote) {
            ++remote;
        }
    }

    std::cout
        << '{' << '\n'
        << "  \"mode\": \""
        << (policy.mode ==
                    strix::HipModelResidencyMode::
                        registered_file_mapping
                ? "registered_file_mapping"
                : "resident_copy")
        << "\"," << '\n'
        << "  \"manifest_tensors\": "
        << manifest.tensors.size() << ',' << '\n'
        << "  \"mapped_file_bytes\": "
        << mapped.bytes() << ',' << '\n'
        << "  \"registered_bytes\": "
        << residency.registered_bytes() << ',' << '\n'
        << "  \"copied_bytes\": "
        << residency.copied_bytes() << ',' << '\n'
        << "  \"gpu_visible_tensors\": "
        << gpu_visible << ',' << '\n'
        << "  \"gpu_visible_tensor_bytes\": "
        << bytes_gpu_visible << ',' << '\n'
        << "  \"file_backed_tensors\": "
        << file_backed << ',' << '\n'
        << "  \"file_backed_tensor_bytes\": "
        << bytes_file_backed << ',' << '\n'
        << "  \"remote_tensors\": "
        << remote << ',' << '\n'
        << "  \"promote_file_backed_candidates\": "
        << (policy.promote_file_backed_candidates
                ? "true"
                : "false")
        << ',' << '\n'
        << "  \"coalesce_gap_bytes\": "
        << policy.coalesce_gap_bytes << ',' << '\n'
        << "  \"storages\": {";

    bool first = true;
    for (const auto storage : {
             strix::RuntimeTensorStorage::resident,
             strix::RuntimeTensorStorage::file_mapped,
             strix::RuntimeTensorStorage::registered_file_mapped,
             strix::RuntimeTensorStorage::repacked_cache,
             strix::RuntimeTensorStorage::remote}) {
        std::uint64_t count = 0;
        for (const auto& tensor :
             residency.binding().tensors) {
            if (tensor.storage == storage) {
                ++count;
            }
        }
        if (!first) {
            std::cout << ',';
        }
        first = false;
        std::cout << '\n' << "    \""
                  << storage_name(storage)
                  << "\": "
                  << count;
    }
    std::cout
        << '\n' << "  }" << '\n'
        << '}' << '\n';

    return 0;
} catch (const std::exception& error) {
    std::cerr
        << "strix-hip-model-residency: "
        << error.what() << '\n';
    return 1;
}
