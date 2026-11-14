#include "strix/model_input.hpp"
#include "strix/gguf_reader.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace strix::gguf {
namespace {

class Reader {
public:
    Reader(std::istream& input, std::uint64_t file_size, const Limits& limits)
        : input_(input), file_size_(file_size), limits_(limits) {}

    FileInfo run() {
        std::array<char, 4> magic{};
        read_bytes(magic.data(), magic.size());
        if (magic != std::array<char, 4>{'G', 'G', 'U', 'F'}) {
            fail("invalid GGUF magic");
        }

        FileInfo info;
        info.file_size = file_size_;
        info.version = read_unsigned<std::uint32_t>();
        if (info.version != 3) {
            fail("only GGUF v3 little-endian is supported");
        }
        const auto tensor_count = read_unsigned<std::uint64_t>();
        const auto metadata_count = read_unsigned<std::uint64_t>();
        if (tensor_count > limits_.max_tensors) {
            fail("tensor count exceeds configured limit");
        }
        if (metadata_count > limits_.max_metadata_entries) {
            fail("metadata count exceeds configured limit");
        }

        info.metadata.reserve(to_size(metadata_count));
        for (std::uint64_t index = 0; index < metadata_count; ++index) {
            auto entry = read_metadata();
            if (!metadata_keys_.insert(entry.key).second) {
                fail("duplicate GGUF metadata key");
            }
            info.metadata.push_back(std::move(entry));
        }
        apply_alignment(info);

        info.tensors.reserve(to_size(tensor_count));
        for (std::uint64_t index = 0; index < tensor_count; ++index) {
            auto tensor = read_tensor();
            if (!tensor_names_.insert(tensor.name).second) {
                fail("duplicate GGUF tensor name");
            }
            info.tensors.push_back(std::move(tensor));
        }

        info.data_offset = align_up(position_, info.alignment);
        if (info.data_offset > file_size_) {
            fail("aligned tensor-data offset exceeds file size");
        }
        while (position_ < info.data_offset) {
            if (read_unsigned<std::uint8_t>() != 0) {
                fail("GGUF padding must contain zero bytes");
            }
        }
        for (auto& tensor : info.tensors) {
            if (tensor.relative_offset % info.alignment != 0) {
                fail("tensor offset is not aligned");
            }
            tensor.absolute_offset = checked_add(info.data_offset, tensor.relative_offset);
            if (tensor.absolute_offset > file_size_) {
                fail("tensor offset exceeds file size");
            }
            tensor.byte_size = tensor_size(tensor);
            tensor.absolute_end = checked_add(tensor.absolute_offset, tensor.byte_size);
            if (tensor.absolute_end > file_size_) {
                fail("tensor data range exceeds file size");
            }
        }
        validate_non_overlapping(info.tensors);
        return info;
    }

private:
    template <typename T>
    T read_unsigned() {
        static_assert(std::is_unsigned_v<T>);
        std::array<unsigned char, sizeof(T)> bytes{};
        read_bytes(reinterpret_cast<char*>(bytes.data()), bytes.size());
        std::uint64_t value = 0;
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
        }
        return static_cast<T>(value);
    }

    template <typename Signed, typename Unsigned>
    Signed read_signed() {
        return std::bit_cast<Signed>(read_unsigned<Unsigned>());
    }

    std::string read_string(std::uint64_t maximum) {
        const auto length = read_unsigned<std::uint64_t>();
        if (length > maximum || length > limits_.max_string_bytes) {
            fail("GGUF string exceeds configured limit");
        }
        std::string value(to_size(length), '\0');
        read_bytes(value.data(), value.size());
        return value;
    }

    ValueType read_value_type() {
        const auto raw = read_unsigned<std::uint32_t>();
        if (raw > static_cast<std::uint32_t>(ValueType::float64)) {
            fail("unknown GGUF metadata value type");
        }
        return static_cast<ValueType>(raw);
    }

    MetadataEntry read_metadata() {
        MetadataEntry entry;
        entry.key = read_string(65'535);
        if (!valid_key(entry.key, limits_.allow_hyphen_in_keys)) {
            fail("invalid GGUF metadata key");
        }
        entry.type = read_value_type();
        read_value(entry.type, &entry, 0);
        return entry;
    }

    void read_value(ValueType type, MetadataEntry* output, std::uint32_t depth) {
        switch (type) {
        case ValueType::uint8:
            store_unsigned(output, read_unsigned<std::uint8_t>());
            break;
        case ValueType::int8:
            store_signed(output, read_signed<std::int8_t, std::uint8_t>());
            break;
        case ValueType::uint16:
            store_unsigned(output, read_unsigned<std::uint16_t>());
            break;
        case ValueType::int16:
            store_signed(output, read_signed<std::int16_t, std::uint16_t>());
            break;
        case ValueType::uint32:
            store_unsigned(output, read_unsigned<std::uint32_t>());
            break;
        case ValueType::int32:
            store_signed(output, read_signed<std::int32_t, std::uint32_t>());
            break;
        case ValueType::float32: {
            const auto bits = read_unsigned<std::uint32_t>();
            if (output != nullptr) {
                output->floating_value = static_cast<double>(std::bit_cast<float>(bits));
            }
            break;
        }
        case ValueType::boolean: {
            const auto value = read_unsigned<std::uint8_t>();
            if (value > 1) {
                fail("GGUF boolean must be zero or one");
            }
            if (output != nullptr) {
                output->boolean_value = value == 1;
            }
            break;
        }
        case ValueType::string: {
            auto value = read_string(limits_.max_string_bytes);
            if (output != nullptr) {
                output->string_value = std::move(value);
            }
            break;
        }
        case ValueType::array:
            read_array(output, depth);
            break;
        case ValueType::uint64:
            store_unsigned(output, read_unsigned<std::uint64_t>());
            break;
        case ValueType::int64:
            store_signed(output, read_signed<std::int64_t, std::uint64_t>());
            break;
        case ValueType::float64: {
            const auto bits = read_unsigned<std::uint64_t>();
            if (output != nullptr) {
                output->floating_value = std::bit_cast<double>(bits);
            }
            break;
        }
        }
    }

    void read_array(MetadataEntry* output, std::uint32_t depth) {
        if (depth >= limits_.max_array_depth) {
            fail("GGUF array nesting exceeds configured limit");
        }
        const auto element_type = read_value_type();
        const auto length = read_unsigned<std::uint64_t>();
        total_array_elements_ = checked_add(total_array_elements_, length);
        if (total_array_elements_ > limits_.max_total_array_elements) {
            fail("GGUF arrays exceed configured element limit");
        }
        if (output != nullptr) {
            output->array_element_type = element_type;
            output->array_length = length;
        }
        const bool retain = output != nullptr && depth == 0 && length <= limits_.max_retained_array_elements &&
                            element_type != ValueType::string && element_type != ValueType::array;
        if (retain) {
            retain_array(element_type, length, *output);
            return;
        }
        for (std::uint64_t index = 0; index < length; ++index) {
            read_value(element_type, nullptr, depth + 1);
        }
    }

    void retain_array(ValueType element_type, std::uint64_t length, MetadataEntry& output) {
        for (std::uint64_t index = 0; index < length; ++index) {
            MetadataEntry value;
            read_value(element_type, &value, 1);
            if (value.unsigned_value.has_value()) {
                output.unsigned_array.push_back(*value.unsigned_value);
            } else if (value.signed_value.has_value()) {
                output.signed_array.push_back(*value.signed_value);
            } else if (value.floating_value.has_value()) {
                output.floating_array.push_back(*value.floating_value);
            } else {
                output.unsigned_array.push_back(value.boolean_value.value_or(false) ? 1U : 0U);
            }
        }
    }

    TensorInfo read_tensor() {
        TensorInfo tensor;
        tensor.name = read_string(64);
        if (tensor.name.empty()) {
            fail("tensor name cannot be empty");
        }
        const auto dimensions = read_unsigned<std::uint32_t>();
        if (dimensions == 0 || dimensions > limits_.max_tensor_dimensions) {
            fail("tensor dimension count is invalid");
        }
        tensor.dimensions.reserve(dimensions);
        for (std::uint32_t index = 0; index < dimensions; ++index) {
            const auto extent = read_unsigned<std::uint64_t>();
            if (extent == 0) {
                fail("tensor dimensions must be non-zero");
            }
            tensor.dimensions.push_back(extent);
        }
        tensor.ggml_type = read_unsigned<std::uint32_t>();
        if (!valid_ggml_type(tensor.ggml_type)) {
            fail("unsupported or removed GGML tensor type");
        }
        tensor.relative_offset = read_unsigned<std::uint64_t>();
        return tensor;
    }

    void apply_alignment(FileInfo& info) {
        const auto* entry = info.find_metadata("general.alignment");
        if (entry != nullptr) {
            if (!entry->unsigned_value.has_value()) {
                fail("general.alignment must be an unsigned integer");
            }
            info.alignment = *entry->unsigned_value;
        }
        if (info.alignment == 0 || info.alignment % 8 != 0) {
            fail("general.alignment must be a non-zero multiple of 8");
        }
    }

    void read_bytes(char* destination, std::size_t length) {
        if (length > file_size_ - position_) {
            fail("unexpected end of GGUF file");
        }
        if (length != 0) {
            input_.read(destination, static_cast<std::streamsize>(length));
            if (!input_) {
                fail("failed to read GGUF bytes");
            }
        }
        position_ += static_cast<std::uint64_t>(length);
    }

    static bool valid_key(const std::string& key, bool allow_hyphen) {
        if (key.empty() || key.front() == '.' || key.back() == '.') {
            return false;
        }
        bool previous_dot = false;
        for (const char symbol : key) {
            const auto character = static_cast<unsigned char>(symbol);
            const bool valid = (character >= 'a' && character <= 'z') ||
                               (character >= '0' && character <= '9') ||
                               character == '_' || character == '.' || (allow_hyphen && character == '-');
            if (!valid || (character == '.' && previous_dot)) {
                return false;
            }
            previous_dot = character == '.';
        }
        return true;
    }

    static bool valid_ggml_type(std::uint32_t type) {
        if (type == 42) {  // Q2_0: block 64, fp16 d + 2-bit codes (GSQ-RCO builds)
            return true;
        }
        if (type > 39) {
            return false;
        }
        switch (type) {
        case 4:
        case 5:
        case 31:
        case 32:
        case 33:
        case 36:
        case 37:
        case 38:
            return false;
        default:
            return true;
        }
    }

    static std::uint64_t tensor_size(const TensorInfo& tensor) {
        const auto geometry = tensor_type_geometry(tensor.ggml_type);
        if (!geometry.has_value()) {
            fail("GGML tensor type has no pinned storage geometry");
        }
        if (tensor.dimensions[0] % geometry->block_elements != 0) {
            fail("tensor row is not divisible by its quantization block size");
        }
        auto bytes = checked_mul(tensor.dimensions[0] / geometry->block_elements,
                                 geometry->block_bytes);
        for (std::size_t dimension = 1; dimension < tensor.dimensions.size(); ++dimension) {
            bytes = checked_mul(bytes, tensor.dimensions[dimension]);
        }
        return bytes;
    }

    static void validate_non_overlapping(const std::vector<TensorInfo>& tensors) {
        std::vector<const TensorInfo*> ordered;
        ordered.reserve(tensors.size());
        for (const auto& tensor : tensors) {
            ordered.push_back(&tensor);
        }
        std::sort(ordered.begin(), ordered.end(), [](const auto* left, const auto* right) {
            return left->absolute_offset < right->absolute_offset;
        });
        for (std::size_t index = 1; index < ordered.size(); ++index) {
            if (ordered[index]->absolute_offset < ordered[index - 1]->absolute_end) {
                fail("GGUF tensor data ranges overlap");
            }
        }
    }

    static std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
        const auto remainder = value % alignment;
        return remainder == 0 ? value : checked_add(value, alignment - remainder);
    }

    static std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
        if (right > std::numeric_limits<std::uint64_t>::max() - left) {
            throw std::overflow_error("GGUF offset overflow");
        }
        return left + right;
    }

    static std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right) {
        if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
            throw std::overflow_error("GGUF tensor-size overflow");
        }
        return left * right;
    }

    static std::size_t to_size(std::uint64_t value) {
        if (value > std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("GGUF size does not fit this host");
        }
        return static_cast<std::size_t>(value);
    }

    template <typename T>
    static void store_unsigned(MetadataEntry* output, T value) {
        if (output != nullptr) {
            output->unsigned_value = static_cast<std::uint64_t>(value);
        }
    }

    template <typename T>
    static void store_signed(MetadataEntry* output, T value) {
        if (output != nullptr) {
            output->signed_value = static_cast<std::int64_t>(value);
        }
    }

    [[noreturn]] static void fail(const char* message) {
        throw std::runtime_error(message);
    }

    std::istream& input_;
    std::uint64_t file_size_;
    const Limits& limits_;
    std::uint64_t position_ = 0;
    std::uint64_t total_array_elements_ = 0;
    std::unordered_set<std::string> metadata_keys_;
    std::unordered_set<std::string> tensor_names_;
};

}  // namespace


std::optional<TensorTypeGeometry> tensor_type_geometry(
    std::uint32_t type) noexcept {
    switch (type) {
    case 0: return TensorTypeGeometry{1, 4};
    case 1: return TensorTypeGeometry{1, 2};
    case 2: return TensorTypeGeometry{32, 18};
    case 3: return TensorTypeGeometry{32, 20};
    case 6: return TensorTypeGeometry{32, 22};
    case 7: return TensorTypeGeometry{32, 24};
    case 8: return TensorTypeGeometry{32, 34};
    case 9: return TensorTypeGeometry{32, 36};
    case 10: return TensorTypeGeometry{256, 84};
    case 11: return TensorTypeGeometry{256, 110};
    case 12: return TensorTypeGeometry{256, 144};
    case 13: return TensorTypeGeometry{256, 176};
    case 14: return TensorTypeGeometry{256, 210};
    case 15: return TensorTypeGeometry{256, 292};
    case 16: return TensorTypeGeometry{256, 66};
    case 17: return TensorTypeGeometry{256, 74};
    case 18: return TensorTypeGeometry{256, 98};
    case 19: return TensorTypeGeometry{256, 50};
    case 20: return TensorTypeGeometry{32, 18};
    case 21: return TensorTypeGeometry{256, 110};
    case 22: return TensorTypeGeometry{256, 82};
    case 23: return TensorTypeGeometry{256, 136};
    case 24: return TensorTypeGeometry{1, 1};
    case 25: return TensorTypeGeometry{1, 2};
    case 26: return TensorTypeGeometry{1, 4};
    case 27: return TensorTypeGeometry{1, 8};
    case 28: return TensorTypeGeometry{1, 8};
    case 29: return TensorTypeGeometry{256, 56};
    case 30: return TensorTypeGeometry{1, 2};
    case 34: return TensorTypeGeometry{256, 54};
    case 35: return TensorTypeGeometry{256, 66};
    case 39: return TensorTypeGeometry{32, 17};
    case 42: return TensorTypeGeometry{64, 18};
    default: return std::nullopt;
    }
}

const MetadataEntry* FileInfo::find_metadata(const std::string& key) const {
    for (const auto& entry : metadata) {
        if (entry.key == key) {
            return &entry;
        }
    }
    return nullptr;
}

FileInfo read(std::istream& input, std::uint64_t file_size, const Limits& limits) {
    return Reader(input, file_size, limits).run();
}

FileInfo read_file(const std::filesystem::path& path, const Limits& limits) {
    const auto size = std::filesystem::file_size(path);
    strix::ModelInput input(path);
    if (!input) {
        throw std::runtime_error("cannot open GGUF file");
    }
    return read(input, size, limits);
}

}  // namespace strix::gguf
