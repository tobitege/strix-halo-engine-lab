#include "strix/serve/prompt_scorer.hpp"

#include "strix/gguf_reader.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace strix::serve::pflash {
namespace {

// Tensor data is read as stored: little-endian IEEE float32.
static_assert(std::endian::native == std::endian::little, "prompt scorer weight files need a little-endian host");

constexpr std::uint32_t kGgmlF32 = 0;
constexpr std::string_view kBaseModel = "Qwen/Qwen3.5-0.8B";
// Bound on any one tensor of these small files; a larger one is a wrong or damaged file.
constexpr std::uint64_t kMaxTensorBytes = 256ULL * 1024ULL * 1024ULL;

class WeightFile {
public:
    WeightFile(const std::filesystem::path& path, std::string_view prefix, std::string_view architecture)
        : path_(path), prefix_(prefix), info_(gguf::read_file(path)) {
        if (text("general.architecture") != architecture) {
            fail("general.architecture is not " + std::string(architecture));
        }
    }

    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error("pflash weights " + path_.string() + ": " + what);
    }

    [[nodiscard]] std::string key(std::string_view name) const {
        return prefix_ + "." + std::string(name);
    }

    [[nodiscard]] const gguf::MetadataEntry* find(const std::string& full_key) const {
        return info_.find_metadata(full_key);
    }

    [[nodiscard]] std::string text(const std::string& full_key) const {
        const gguf::MetadataEntry* entry = find(full_key);
        if (entry == nullptr || entry->type != gguf::ValueType::string || !entry->string_value) {
            fail("missing string " + full_key);
        }
        return *entry->string_value;
    }

    [[nodiscard]] std::string optional_text(const std::string& full_key) const {
        return find(full_key) == nullptr ? std::string() : text(full_key);
    }

    [[nodiscard]] float f32(const std::string& full_key) const {
        const gguf::MetadataEntry* entry = find(full_key);
        if (entry == nullptr || entry->type != gguf::ValueType::float32 || !entry->floating_value) {
            fail("missing float32 " + full_key);
        }
        return static_cast<float>(*entry->floating_value);
    }

    [[nodiscard]] std::uint32_t u32(const std::string& full_key) const {
        const gguf::MetadataEntry* entry = find(full_key);
        if (entry == nullptr || entry->type != gguf::ValueType::uint32 || !entry->unsigned_value) {
            fail("missing uint32 " + full_key);
        }
        return static_cast<std::uint32_t>(*entry->unsigned_value);
    }

    // The checks every weight file shares: schema, base model, drafter hash, feature tap.
    [[nodiscard]] std::string identity(std::initializer_list<std::string_view> schemas,
                                       std::string_view runtime_sha256) const {
        const std::string schema = text(key("schema"));
        bool known = false;
        for (const std::string_view candidate : schemas) {
            known = known || schema == candidate;
        }
        if (!known) {
            fail("unknown schema " + schema);
        }
        if (text(key("base_model")) != kBaseModel) {
            fail("unexpected base model");
        }
        if (runtime_sha256.empty() || text(key("runtime_gguf_sha256")) != runtime_sha256) {
            fail("trained against different drafter weights (runtime_gguf_sha256)");
        }
        if (text(key("feature_tap")) != kFeatureTap) {
            fail("unexpected feature tap");
        }
        return schema;
    }

    [[nodiscard]] const gguf::TensorInfo& tensor(std::string_view name) const {
        const std::string full = key(name);
        for (const gguf::TensorInfo& candidate : info_.tensors) {
            if (candidate.name == full) {
                if (candidate.ggml_type != kGgmlF32) {
                    fail(full + " is not F32");
                }
                return candidate;
            }
        }
        fail("missing tensor " + full);
    }

    // Reads an F32 tensor whose dimensions (GGUF order, fastest first) equal `shape`. Trailing unit
    // dimensions may be absent in the file. A zero in `shape` accepts any size and reports it back.
    [[nodiscard]] std::vector<float> floats(std::string_view name, std::vector<std::uint64_t>& shape) const {
        const gguf::TensorInfo& info = tensor(name);
        std::uint64_t elements = 1;
        for (std::size_t d = 0; d < shape.size() || d < info.dimensions.size(); ++d) {
            const std::uint64_t have = d < info.dimensions.size() ? info.dimensions[d] : 1;
            if (d >= shape.size()) {
                if (have != 1) {
                    fail(key(name) + " has too many dimensions");
                }
                continue;
            }
            if (shape[d] == 0) {
                shape[d] = have;
            }
            if (have != shape[d] || have == 0) {
                fail(key(name) + " has an unexpected shape");
            }
            if (elements > kMaxTensorBytes / have) {
                fail(key(name) + " is too large");
            }
            elements *= have;
        }
        if (elements * sizeof(float) > kMaxTensorBytes || info.byte_size != elements * sizeof(float) ||
            info.absolute_end > info_.file_size) {
            fail(key(name) + " has an unexpected size");
        }
        std::vector<float> values(static_cast<std::size_t>(elements));
        std::ifstream input(path_, std::ios::binary);
        input.seekg(static_cast<std::streamoff>(info.absolute_offset));
        input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(info.byte_size));
        if (!input || static_cast<std::uint64_t>(input.gcount()) != info.byte_size) {
            fail(key(name) + " could not be read");
        }
        for (const float value : values) {
            if (!std::isfinite(value)) {
                fail(key(name) + " holds a non-finite value");
            }
        }
        return values;
    }

    [[nodiscard]] std::vector<float> floats(std::string_view name, std::initializer_list<std::uint64_t> shape) const {
        std::vector<std::uint64_t> wanted(shape);
        return floats(name, wanted);
    }

private:
    std::filesystem::path path_;
    std::string prefix_;
    gguf::FileInfo info_;
};

std::uint32_t narrow(const WeightFile& file, std::uint64_t value) {
    if (value == 0 || value > std::numeric_limits<std::uint32_t>::max()) {
        file.fail("dimension out of range");
    }
    return static_cast<std::uint32_t>(value);
}

// norm, fc1, fc2 of a probe; hidden and width come from fc1.
ProbeMlp load_mlp(const WeightFile& file) {
    ProbeMlp mlp;
    std::vector<std::uint64_t> fc1_shape{0, 0};  // [hidden, width] in GGUF order: rows are [width][hidden]
    mlp.fc1 = file.floats("fc1.weight", fc1_shape);
    mlp.hidden = narrow(file, fc1_shape[0]);
    mlp.width = narrow(file, fc1_shape[1]);
    mlp.norm_weight = file.floats("norm.weight", {mlp.hidden});
    mlp.norm_bias = file.floats("norm.bias", {mlp.hidden});
    mlp.fc1_bias = file.floats("fc1.bias", {mlp.width});
    mlp.fc2 = file.floats("fc2.weight", {mlp.width});
    mlp.fc2_bias = file.floats("fc2.bias", {1}).front();
    mlp.norm_eps = file.f32(file.key("norm_eps"));
    if (!(mlp.norm_eps > 0.0f) || !std::isfinite(mlp.norm_eps)) {
        file.fail("norm_eps must be positive");
    }
    return mlp;
}

}  // namespace

ScoringHeadFile load_scoring_head(const std::filesystem::path& path, std::string_view runtime_sha256) {
    const WeightFile file(path, "scoringhead", "pflash_scoring_head");
    ScoringHeadFile head;
    head.schema = file.identity({kScoringHeadSchema}, runtime_sha256);
    head.base_model = std::string(kBaseModel);
    std::vector<std::uint64_t> q_shape{0, 0};
    head.wq = file.floats("attn_q.weight", q_shape);
    head.hidden = narrow(file, q_shape[0]);
    head.q_rows = narrow(file, q_shape[1]);
    std::vector<std::uint64_t> k_shape{head.hidden, 0};
    head.wk = file.floats("attn_k.weight", k_shape);
    head.k_rows = narrow(file, k_shape[1]);
    if (head.q_rows % head.k_rows != 0) {
        file.fail("attn_q rows are not a multiple of attn_k rows");
    }
    return head;
}

SegmentProbe load_segment_probe(const std::filesystem::path& path, std::string_view runtime_sha256) {
    const WeightFile file(path, "segmentprobe", "segmentprobe");
    SegmentProbe probe;
    probe.schema = file.identity({kSegmentProbeSchemaV1, kSegmentProbeSchemaV2}, runtime_sha256);
    probe.mlp = load_mlp(file);
    std::vector<std::uint64_t> conv_shape{0};
    probe.conv = file.floats("conv.weight", conv_shape);
    if (probe.conv.size() % 2 == 0) {
        file.fail("conv length must be odd");
    }
    probe.conv_bias = file.floats("conv.bias", {1}).front();
    probe.threshold = file.f32(file.key("threshold"));
    if (!(probe.threshold > 0.0f && probe.threshold < 1.0f)) {
        file.fail("threshold must be inside (0, 1)");
    }
    probe.min_segment = file.u32(file.key("min_segment"));
    probe.max_segment = file.u32(file.key("max_segment"));
    if (probe.min_segment < 1 || probe.max_segment < probe.min_segment) {
        file.fail("min_segment / max_segment out of order");
    }
    probe.oversize_split = file.optional_text(file.key("oversize_split"));
    if (!probe.oversize_split.empty() && probe.oversize_split != "line_aligned" &&
        probe.oversize_split != "best_score") {
        file.fail("unknown oversize_split");
    }
    return probe;
}

RequestProbe load_request_probe(const std::filesystem::path& path, std::string_view runtime_sha256) {
    const WeightFile file(path, "requestprobe", "requestprobe");
    RequestProbe probe;
    probe.schema = file.identity({kRequestProbeSchema}, runtime_sha256);
    if (file.text(file.key("extraction_version")) != kRequestExtraction) {
        file.fail("unknown extraction_version");
    }
    probe.mlp = load_mlp(file);
    probe.tau = file.f32(file.key("tau"));
    if (std::isnan(probe.tau)) {
        file.fail("tau is NaN");
    }
    probe.candidate_rule = file.optional_text(file.key("candidate_rule"));
    return probe;
}

}  // namespace strix::serve::pflash
