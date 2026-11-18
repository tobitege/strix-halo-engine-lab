#include "strix/gguf_reader.hpp"
#include "strix/memory_plan.hpp"
#include "strix/oracles.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Function>
void expect_overflow(Function&& function, const char* message) {
    try {
        function();
        expect(false, message);
    } catch (const std::overflow_error&) {
    }
}

template <typename Function>
void expect_runtime_error(Function&& function, const char* message) {
    try {
        function();
        expect(false, message);
    } catch (const std::runtime_error&) {
    }
}

template <typename T>
void append_le(std::string& bytes, T value) {
    static_assert(std::is_unsigned_v<T>);
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        bytes.push_back(static_cast<char>((value >> (index * 8U)) & 0xffU));
    }
}

void append_string(std::string& bytes, const std::string& value) {
    append_le<std::uint64_t>(bytes, value.size());
    bytes.append(value);
}

std::string valid_gguf(std::uint64_t tensor_offset = 0,
                       std::uint32_t tensor_type = 1) {
    std::string bytes = "GGUF";
    append_le<std::uint32_t>(bytes, 3);
    append_le<std::uint64_t>(bytes, 1);
    append_le<std::uint64_t>(bytes, 2);

    append_string(bytes, "general.architecture");
    append_le<std::uint32_t>(bytes, 8);
    append_string(bytes, "qwen4exp");
    append_string(bytes, "general.alignment");
    append_le<std::uint32_t>(bytes, 4);
    append_le<std::uint32_t>(bytes, 32);

    append_string(bytes, "blk.0.weight");
    append_le<std::uint32_t>(bytes, 2);
    append_le<std::uint64_t>(bytes, 16);
    append_le<std::uint64_t>(bytes, 32);
    append_le<std::uint32_t>(bytes, tensor_type);
    append_le<std::uint64_t>(bytes, tensor_offset);
    while (bytes.size() % 32 != 0) {
        bytes.push_back('\0');
    }
    bytes.append(1024, '\0');
    return bytes;
}

std::string array_gguf() {
    std::string bytes = "GGUF";
    append_le<std::uint32_t>(bytes, 3);
    append_le<std::uint64_t>(bytes, 0);
    append_le<std::uint64_t>(bytes, 1);
    append_string(bytes, "test.values");
    append_le<std::uint32_t>(bytes, 9);
    append_le<std::uint32_t>(bytes, 4);
    append_le<std::uint64_t>(bytes, 3);
    append_le<std::uint32_t>(bytes, 10);
    append_le<std::uint32_t>(bytes, 20);
    append_le<std::uint32_t>(bytes, 30);
    while (bytes.size() % 32 != 0) {
        bytes.push_back('\0');
    }
    return bytes;
}

std::string two_tensor_gguf(std::uint64_t second_offset) {
    std::string bytes = "GGUF";
    append_le<std::uint32_t>(bytes, 3);
    append_le<std::uint64_t>(bytes, 2);
    append_le<std::uint64_t>(bytes, 0);
    const auto append_tensor = [&](const std::string& name, std::uint64_t offset) {
        append_string(bytes, name);
        append_le<std::uint32_t>(bytes, 2);
        append_le<std::uint64_t>(bytes, 16);
        append_le<std::uint64_t>(bytes, 32);
        append_le<std::uint32_t>(bytes, 1);
        append_le<std::uint64_t>(bytes, offset);
    };
    append_tensor("first.weight", 0);
    append_tensor("second.weight", second_offset);
    while (bytes.size() % 32 != 0) {
        bytes.push_back('\0');
    }
    bytes.append(2048, '\0');
    return bytes;
}

strix::gguf::FileInfo parse_gguf(const std::string& bytes,
                                 const strix::gguf::Limits& limits = {}) {
    std::istringstream input(bytes, std::ios::in | std::ios::binary);
    return strix::gguf::read(input, bytes.size(), limits);
}

}  // namespace

int main() {
    const auto model = strix::ModelContract::qwen38_flash_next();
    expect(model.validate().empty(), "official model shape contract validates");
    expect(strix::base_kv_bytes_per_token(model) == 24ULL * 1024ULL,
           "base KV is 24 KiB/token/session");
    expect(strix::mtp_kv_bytes_per_token(model) == 2ULL * 1024ULL,
           "MTP KV is 2 KiB/token/session");
    {
        auto int8 = model;
        int8.attention_kv_storage = strix::AttentionKvStorage::int8_token;
        expect(int8.validate().empty(), "kv-int8-tok-v1 contract validates");
        expect(strix::base_kv_bytes_per_token(int8) == 12ULL * (2ULL * 512ULL + 16ULL),
               "kv-int8-tok-v1 KV is 12 x (1024 code + 16 scale) bytes/token/session");
        expect(strix::mtp_kv_bytes_per_token(int8) == 2ULL * 1024ULL,
               "the MTP head KV stays BF16");
        auto invalid = model;
        invalid.attention_kv_storage = static_cast<strix::AttentionKvStorage>(7U);
        expect(!invalid.validate().empty(), "unknown attention KV storage is rejected");
    }
    expect(strix::indexer_bytes_per_token(model) == 3ULL * 1024ULL,
           "exact QSA raw-key cache is 3 KiB/token/session");
    expect(strix::provisional_indexer_bytes_per_token(model) ==
               strix::indexer_bytes_per_token(model),
           "legacy indexer sizing alias resolves to exact raw-key storage");
    expect(model.vocab_size == 248320,
           "official vocabulary size is explicit");
    expect(model.full_attention_interval == 4 &&
               model.ple_layer_1based == 2 &&
               model.hc_count == 4,
           "official layer pattern and hyper-connection geometry are explicit");
    expect(model.max_position_embeddings == 262144,
           "official maximum context is explicit");
    expect(model.hc_count == 4 &&
               model.hc_lowrank == 320,
           "official hyper-connection geometry is explicit");
    expect(model.ple_head_count() == 16 &&
               model.ple_head_dim() == 160 &&
               model.ple_row_bytes_bf16() == 320 &&
               model.ngram_vocab_size_base == 20'000'000 &&
               model.ngram_vocab_divisor == 128 &&
               model.ngram_hash_seed == 1234 &&
               model.ple_eos_token_id == 248044,
           "canonical PLE hash geometry is explicit");

    expect(model.rms_norm_epsilon == 1e-6F &&
               model.norm_topk_probability,
           "official RMS/router policy is explicit");
    expect(model.indexer_n_heads == 4 &&
               model.indexer_kv_heads == 1 &&
               model.indexer_head_dim == 128 &&
               model.indexer_budget == 2048 &&
               model.indexer_compress_ratio == 4 &&
               model.partial_rotary_factor == 0.25F &&
               model.rotary_dim() == 64 &&
               model.rope_theta == 10'000'000.0F,
           "official QSA geometry and partial RoPE are explicit");
    expect(model.gdn_output_gate_sigmoid,
           "official GDN output gate is sigmoid");
    expect(model.hidden_size == 2560 &&
               model.moe_intermediate_size == 640 &&
               model.shared_expert_intermediate_size == 640 &&
               model.num_experts == 512 &&
               model.num_experts_per_token == 10,
           "official MoE geometry is explicit");
    expect(model.gdn_key_heads == 16 &&
               model.gdn_key_head_dim == 128 &&
               model.gdn_value_heads == 48 &&
               model.gdn_value_head_dim == 128 &&
               model.gdn_conv_kernel == 4,
           "official GDN geometry is explicit");
    expect(strix::gdn_matrix_state_bytes_per_layer(model) ==
               3ULL * 1024ULL * 1024ULL,
           "one GDN matrix layer is 3 MiB");
    {
        auto f16_model = model;
        f16_model.recurrent_state = strix::RecurrentStateType::f16;
        expect(strix::gdn_matrix_state_bytes_per_layer(f16_model) * 2 == strix::gdn_matrix_state_bytes_per_layer(model) &&
                   strix::gdn_conv_state_bytes_per_layer(f16_model) == strix::gdn_conv_state_bytes_per_layer(model),
               "f16 recurrent matrix is half of f32, convolution unchanged");
    }
    expect(strix::gdn_conv_state_bytes_per_layer(model) ==
               80ULL * 1024ULL,
           "one GDN convolution layer is 80 KiB");
    expect(strix::gdn_matrix_state_bytes_per_session(model) ==
               108ULL * 1024ULL * 1024ULL,
           "GDN FP32 matrix state is 108 MiB/session");
    expect(strix::gdn_conv_state_bytes_per_session(model) ==
               2880ULL * 1024ULL,
           "GDN convolution history is 2880 KiB/session");
    expect(strix::gdn_state_bytes_per_session(model) ==
               110ULL * 1024ULL * 1024ULL + 832ULL * 1024ULL,
           "complete GDN state includes recurrent matrix and convolution history");
    expect(strix::ple_conv_state_bytes_per_session(model) ==
               180ULL * 1024ULL,
           "PLE short convolution state is 180 KiB");
    expect(strix::ngram_history_bytes_per_session(model) == 8,
           "PLE resume history keeps two uint32 token ids");
    expect(strix::position_history_bytes_per_session(model) == 8,
           "text serial position history is one uint64");

    strix::MemoryPlanRequest request;
    request.context_tokens = 262'144;
    request.sessions = 1;
    request.include_indexer = false;
    const auto one_session = strix::build_memory_plan(model, request);
    expect(one_session.ple_conv_state == 180ULL * 1024ULL,
           "memory plan includes PLE convolution state");
    expect(one_session.ngram_history == 8 &&
               one_session.position_history == 8,
           "memory plan includes host resume metadata");
    expect(one_session.base_kv == strix::gib(6), "262K base KV is 6 GiB");
    expect(one_session.mtp_kv == strix::gib(0) + 512ULL * 1024ULL * 1024ULL,
           "262K MTP KV is 512 MiB");

    request.sessions = 4;
    request.rollback = strix::RollbackPolicy::full_gdn_snapshots;
    request.draft_tokens = 7;
    const auto c4 = strix::build_memory_plan(model, request);
    expect(c4.base_kv == strix::gib(24), "C4 base KV is 24 GiB");
    expect(c4.rollback ==
               3ULL * strix::gib(1) +
                   30ULL * 1024ULL * 1024ULL +
                   768ULL * 1024ULL,
           "C4 seven-snapshot rollback includes convolution history");

    const strix::HostBudget host{strix::gib(128), strix::gib(2), strix::gib(18)};
    auto small = strix::MemoryPlanRequest{};
    small.resident_model_bytes = strix::gib(68);
    const auto small_plan = strix::build_memory_plan(model, small);
    expect(strix::decide_admission(host, small_plan).admitted,
           "default 32K C1 plan fits host budget");

    request.resident_model_bytes = strix::gib(93);
    request.bounded_cache_bytes = strix::gib(4);
    const auto large_plan = strix::build_memory_plan(model, request);
    expect(!strix::decide_admission(host, large_plan).admitted,
           "large C4 full-rollback plan is rejected");

    expect_overflow(
        [&model] {
            auto overflow = strix::MemoryPlanRequest{};
            overflow.context_tokens = std::numeric_limits<std::uint64_t>::max();
            overflow.sessions = 2;
            static_cast<void>(strix::build_memory_plan(model, overflow));
        },
        "overflow is rejected");

    const float tied_scores[] = {4.0F, 9.0F, 9.0F, -1.0F, 8.0F};
    const auto top = strix::oracle::top_k_lower_index(tied_scores, 3);
    expect(top == std::vector<std::size_t>({1, 2, 4}),
           "candidate top-k keeps ties and orders them by lower index");
    try {
        const float invalid_scores[] = {1.0F, std::nanf("")};
        static_cast<void>(strix::oracle::top_k_lower_index(invalid_scores, 1));
        expect(false, "candidate top-k rejects NaN");
    } catch (const std::invalid_argument&) {
    }

    const auto gguf_bytes = valid_gguf();
    const auto gguf = parse_gguf(gguf_bytes);
    expect(gguf.version == 3, "GGUF v3 fixture parses");
    expect(gguf.alignment == 32, "GGUF alignment metadata is applied");
    expect(gguf.metadata.size() == 2, "GGUF metadata count is preserved");
    expect(gguf.tensors.size() == 1, "GGUF tensor directory parses");
    expect(gguf.tensors[0].absolute_offset == gguf.data_offset,
           "GGUF relative tensor offset resolves from data section");
    expect(gguf.tensors[0].byte_size == 1024 &&
               gguf.tensors[0].absolute_end == gguf_bytes.size(),
           "GGUF F16 tensor range is computed exactly");
    const auto* architecture = gguf.find_metadata("general.architecture");
    expect(architecture != nullptr && architecture->string_value == "qwen4exp",
           "GGUF architecture metadata is available");

    expect_runtime_error(
        [] {
            auto bytes = valid_gguf();
            bytes[0] = 'B';
            static_cast<void>(parse_gguf(bytes));
        },
        "GGUF rejects invalid magic");
    expect_runtime_error(
        [] { static_cast<void>(parse_gguf(valid_gguf(1))); },
        "GGUF rejects unaligned tensor offsets");
    expect_runtime_error(
        [] { static_cast<void>(parse_gguf(valid_gguf(0, 31))); },
        "GGUF rejects removed tensor types");
    expect_runtime_error(
        [] { static_cast<void>(parse_gguf(valid_gguf(0, 2))); },
        "GGUF rejects rows not divisible by quantization block size");
    expect_runtime_error(
        [] {
            auto bytes = valid_gguf();
            bytes.pop_back();
            static_cast<void>(parse_gguf(bytes));
        },
        "GGUF rejects tensor ranges extending beyond the file");
    const auto two_tensors = parse_gguf(two_tensor_gguf(1024));
    expect(two_tensors.tensors[1].absolute_offset ==
               two_tensors.tensors[0].absolute_end,
           "GGUF permits adjacent non-overlapping tensor ranges");
    expect_runtime_error(
        [] { static_cast<void>(parse_gguf(two_tensor_gguf(0))); },
        "GGUF rejects overlapping tensor data ranges");
    expect_runtime_error(
        [] {
            auto bytes = valid_gguf();
            strix::gguf::Limits limits;
            limits.max_string_bytes = 4;
            static_cast<void>(parse_gguf(bytes, limits));
        },
        "GGUF enforces string limits before allocation");

    expect_runtime_error(
        [] {
            std::string bytes = "GGUF";
            append_le<std::uint32_t>(bytes, 3);
            append_le<std::uint64_t>(bytes, 0);
            append_le<std::uint64_t>(bytes, 1);
            append_string(bytes, "test.flag");
            append_le<std::uint32_t>(bytes, 7);
            append_le<std::uint8_t>(bytes, 2);
            while (bytes.size() % 32 != 0) {
                bytes.push_back('\0');
            }
            static_cast<void>(parse_gguf(bytes));
        },
        "GGUF rejects non-canonical booleans");

    const auto array_info = parse_gguf(array_gguf());
    expect(array_info.metadata[0].array_element_type == strix::gguf::ValueType::uint32 &&
               array_info.metadata[0].array_length == 3,
           "GGUF array summary parses without retaining its payload");
    expect_runtime_error(
        [] {
            strix::gguf::Limits limits;
            limits.max_total_array_elements = 2;
            static_cast<void>(parse_gguf(array_gguf(), limits));
        },
        "GGUF enforces aggregate array element limit");

    if (failures == 0) {
        std::cout << "all core tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
