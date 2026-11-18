#include "strix/asset_manifest.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_content_hash.hpp"

#include <chrono>
#include <fstream>
#include <map>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

std::string json_escape(const std::string& value) {
    std::string result;
    for (const unsigned char character : value) {
        if (character == '"' || character == '\\') {
            result.push_back('\\');
            result.push_back(static_cast<char>(character));
        } else if (character >= 0x20U && character < 0x80U) {
            result.push_back(static_cast<char>(character));
        } else {
            std::ostringstream escaped;
            escaped << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned int>(character);
            result += escaped.str();
        }
    }
    return result;
}

void print_dimensions(const std::vector<std::uint64_t>& dimensions) {
    std::cout << '[';
    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << dimensions[index];
    }
    std::cout << ']';
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc == 3 && std::string(argv[1]) == "--warm-content-hash") {
        const auto begin = std::chrono::steady_clock::now();
        const auto identity = strix::hash_single_model_content_file({"model", argv[2]});
        const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        std::cout << "{\"file_sha256\":\"" << strix::model_content_identity_hex(identity.file_sha256)
                  << "\",\"content_identity\":\"" << strix::model_content_identity_hex(identity.content_identity)
                  << "\",\"bytes\":" << identity.file_bytes << ",\"ms\":" << elapsed << "}\n";
        return 0;
    }
    if (argc > 1 && (std::string(argv[1]) == "bundle-create" || std::string(argv[1]) == "bundle-verify")) {
        try {
            const bool create = std::string(argv[1]) == "bundle-create";
            std::map<std::string, std::string> options;
            std::vector<std::string> roles;
            for (int i = 2; i < argc; i += 2) {
                if (i + 1 >= argc) throw std::invalid_argument("missing bundle option value");
                const std::string key = argv[i];
                if (!create && key == "--role") roles.emplace_back(argv[i + 1]);
                else if (key == "--root" || (create && (key == "--spec" || key == "--output")) || (!create && key == "--manifest")) {
                    if (!options.emplace(key, argv[i + 1]).second) throw std::invalid_argument("duplicate bundle option");
                } else throw std::invalid_argument("unknown bundle option");
            }
            if (options.size() != (create ? 3U : 2U)) throw std::invalid_argument("missing bundle options");
            const auto input = strix::read_asset_json(options.at(create ? "--spec" : "--manifest"));
            if (create) {
                const auto result = strix::create_asset_manifest(input, options.at("--root"));
                std::ofstream out(options.at("--output"), std::ios::binary);
                out << result << '\n';
                out.close();
                if (!out) throw std::runtime_error("asset output failed");
            } else std::cout << strix::verify_asset_manifest(input, options.at("--root"), roles) << '\n';
            return 0;
        } catch (const strix::AssetMismatch& e) {
            std::cout << '{';
            if (!e.role.empty()) {
                std::cout << "\"index\":" << e.index << ',';
                if (!e.key.empty()) std::cout << "\"key\":\"" << json_escape(e.key) << "\",";
                else std::cout << "\"key_digest\":\"" << e.key_digest << "\",";
                std::cout << "\"role\":\"" << e.role << "\",";
            }
            std::cout << "\"reason\":\"" << json_escape(e.what()) << "\",\"verification\":\"failed\"}\n";
            return 2;
        } catch (const std::exception&) {
            std::cerr << "error: invalid bundle input or I/O failure\n";
            return 1;
        }
    }
    if (argc != 2) {
        throw std::invalid_argument("usage: strix-model-manifest MODEL.gguf");
    }

    const auto info = strix::gguf::read_file(argv[1]);
    const auto manifest = strix::build_model_manifest(info);

    std::cout << "{\n"
              << "  \"architecture\": \"" << json_escape(manifest.architecture) << "\",\n"
              << "  \"layout_identity\": " << manifest.layout_identity << ",\n"
              << "  \"total_tensor_bytes\": " << manifest.total_tensor_bytes << ",\n"
              << "  \"file_backed_candidate_bytes\": " << manifest.file_backed_candidate_bytes << ",\n"
              << "  \"bounded_cache_candidate_bytes\": " << manifest.bounded_cache_candidate_bytes << ",\n"
              << "  \"unknown_tensor_count\": " << manifest.unknown_tensor_count << ",\n"
              << "  \"tensors\": [\n";

    for (std::size_t index = 0; index < manifest.tensors.size(); ++index) {
        const auto& tensor = manifest.tensors[index];
        std::cout << "    {\"name\":\"" << json_escape(tensor.name)
                  << "\",\"role\":\"" << strix::to_string(tensor.role)
                  << "\",\"residency\":\"" << strix::to_string(tensor.residency)
                  << "\",\"placement\":{\"kind\":\"" << strix::to_string(tensor.placement.kind)
                  << "\",\"node\":" << tensor.placement.node << "},\"layer\":";
        if (tensor.layer.has_value()) {
            std::cout << *tensor.layer;
        } else {
            std::cout << "null";
        }
        std::cout << ",\"ggml_type\":" << tensor.ggml_type
                  << ",\"byte_size\":" << tensor.byte_size
                  << ",\"absolute_offset\":" << tensor.absolute_offset
                  << ",\"dimensions\":";
        print_dimensions(tensor.dimensions);
        std::cout << '}' << (index + 1 == manifest.tensors.size() ? '\n' : ',') << '\n';
    }

    std::cout << "  ]\n}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}
