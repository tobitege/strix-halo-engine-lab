#include "gemma4/model/model.hpp"
#include <array>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <iostream>
int main() {
    try {
        const auto path = std::filesystem::current_path()/"gemma4-shared-reader.safetensors";
        const std::string header = R"({"n":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
        {
            std::ofstream f(path,std::ios::binary);
            const auto size = static_cast<std::uint64_t>(header.size()); const float values[2] = {1,2};
            f.write(reinterpret_cast<const char*>(&size),8); f.write(header.data(),static_cast<std::streamsize>(header.size()));
            f.write(reinterpret_cast<const char*>(values),sizeof(values));
        }
        auto file = strix::gemma4::read_safetensors(path);
        if (file.tensors.size() != 1 || file.tensors[0].bytes != 8 || file.tensors[0].shape != std::vector<std::uint64_t>{2}) return 1;
        if (file.index.tensors.size() != 1 || !file.index.owner) return 1;
        const auto& raw = file.index.tensors[0];
        if (raw.name != file.tensors[0].name || raw.wire_dtype != "F32" ||
            raw.payload.range.offset != file.tensors[0].offset || raw.payload.range.bytes != 8) return 1;
        auto index = file.index;
        const auto& t = file.tensors[0];
        const strix::weights::Tensor view{file.storage,t.shape,t.dtype,0,t.offset,t.bytes};
        file = {};
        if (strix::weights::decode_row(view,0) != std::vector<float>{1,2}) return 1;
        std::array<std::byte, 8> payload{};
        if (!index.tensors[0].payload.source->read(index.tensors[0].payload.range, payload)) return 1;
        if (std::memcmp(payload.data(), view.bytes().data(), payload.size()) != 0) return 1;
        index = {};
        // A second file avoids modifying a live mapping on Windows.
        const auto invalid = path.string() + ".invalid";
        std::filesystem::copy_file(path, invalid, std::filesystem::copy_options::overwrite_existing);
        { std::ofstream f(invalid, std::ios::binary | std::ios::app); f.put(0); }
        bool refused = false;
        try { (void)strix::gemma4::read_safetensors(invalid); }
        catch (const std::runtime_error&) { refused = true; }
        if (!refused) return 1;
        std::cout << "Gemma shared reader and view ownership passed\n"; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
