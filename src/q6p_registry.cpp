#include "strix/env_flags.hpp"
#include "strix/q6p_registry.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace strix::q6p {
namespace {
struct Registry {
    std::shared_mutex mutex;
    std::unordered_map<const void*, Tensor> tensors;
    std::atomic<std::uint64_t> hits{0};
};
Registry& registry() {
    static Registry instance;
    return instance;
}

// Byte offset of block b (of cols / 32) inside a row: chunk start and the block's
// qs, qh and d offsets within that chunk.
struct BlockOffsets {
    std::uint64_t qs, qh, d;
};
BlockOffsets block_offsets(std::uint64_t cols, std::uint64_t b) {
    const std::uint64_t chunk = b / 8U, in = b % 8U, n = cols - 256U * chunk < 256U ? cols - 256U * chunk : 256U;
    const std::uint64_t at = 208U * chunk;
    return {at + 16U * in, at + n / 2U + 8U * in, at + n / 2U + n / 4U + 2U * in};
}
}  // namespace

void register_tensor(const void* original_address, const Tensor& tensor) {
    auto& r = registry();
    const std::unique_lock lock(r.mutex);
    r.tensors[original_address] = tensor;
}

void unregister_tensor(const void* original_address, const unsigned char* copy_address) noexcept {
    auto& r = registry();
    const std::unique_lock lock(r.mutex);
    const auto it = r.tensors.find(original_address);
    if (it != r.tensors.end() && it->second.base == copy_address) r.tensors.erase(it);
}

void clear() noexcept {
    auto& r = registry();
    const std::unique_lock lock(r.mutex);
    r.tensors.clear();
}

const Tensor* find(const void* original_address) noexcept {
    if (!kernels_enabled()) return nullptr;
    auto& r = registry();
    const std::shared_lock lock(r.mutex);
    const auto it = r.tensors.find(original_address);
    if (it == r.tensors.end()) return nullptr;
    r.hits.fetch_add(1U, std::memory_order_relaxed);
    return &it->second;
}

bool registered(const void* original_address) noexcept {
    if (!kernels_enabled()) return false;
    auto& r = registry();
    const std::shared_lock lock(r.mutex);
    return r.tensors.contains(original_address);
}

std::uint64_t hits() noexcept { return registry().hits.load(std::memory_order_relaxed); }

bool kernels_enabled() noexcept {
    static const bool enabled = strix::envcfg::flag("STRIX_Q6P", true);
    return enabled;
}

std::uint64_t registered_count() noexcept {
    auto& r = registry();
    const std::shared_lock lock(r.mutex);
    return r.tensors.size();
}

namespace {
std::atomic<bool> derived_admitted{false};
unsigned char derived_identity[32]{};
}  // namespace
void admit_derived_content(const unsigned char (&identity)[32]) noexcept {
    std::memcpy(derived_identity, identity, 32U);
    derived_admitted.store(true, std::memory_order_release);
}
bool derived_content_admitted(const unsigned char* identity) noexcept {
    return derived_admitted.load(std::memory_order_acquire) && std::memcmp(derived_identity, identity, 32U) == 0;
}

bool to_q6p32(const unsigned char* q8, std::uint64_t rows, std::uint64_t cols, unsigned char* out) {
    const std::uint64_t blocks = cols / 32U, row_bytes = blocks * 26U;
    for (std::uint64_t r = 0; r < rows; ++r)
        for (std::uint64_t b = 0; b < blocks; ++b) {
            const unsigned char* block = q8 + (r * blocks + b) * 34U;
            unsigned char c[32];
            for (unsigned e = 0; e < 32U; ++e) {
                const int q = static_cast<signed char>(block[2U + e]);
                if (q < -32 || q > 31) return false;
                c[e] = static_cast<unsigned char>(q + 32);
            }
            const auto o = block_offsets(cols, b);
            unsigned char* row = out + r * row_bytes;
            for (unsigned k = 0; k < 4U; ++k)
                for (unsigned i = 0; i < 4U; ++i)
                    row[o.qs + 4U * k + i] = static_cast<unsigned char>((c[8U * k + i] & 0x0fU) | (c[8U * k + 4U + i] & 0x0fU) << 4U);
            for (unsigned j = 0; j < 2U; ++j)
                for (unsigned i = 0; i < 4U; ++i) {
                    unsigned v = 0;
                    for (unsigned f = 0; f < 4U; ++f) v |= ((c[16U * j + 4U * f + i] >> 4U) & 3U) << (2U * f);
                    row[o.qh + 4U * j + i] = static_cast<unsigned char>(v);
                }
            row[o.d] = block[0];
            row[o.d + 1U] = block[1];
        }
    return true;
}

void to_q8_0(const unsigned char* q6, std::uint64_t rows, std::uint64_t cols, unsigned char* out) {
    const std::uint64_t blocks = cols / 32U, row_bytes = blocks * 26U;
    for (std::uint64_t r = 0; r < rows; ++r)
        for (std::uint64_t b = 0; b < blocks; ++b) {
            const auto o = block_offsets(cols, b);
            const unsigned char* row = q6 + r * row_bytes;
            unsigned char* block = out + (r * blocks + b) * 34U;
            block[0] = row[o.d];
            block[1] = row[o.d + 1U];
            for (unsigned e = 0; e < 32U; ++e) {
                const unsigned k = e / 8U, i = e % 4U, upper = (e % 8U) >= 4U ? 4U : 0U;
                const unsigned lo = (row[o.qs + 4U * k + i] >> upper) & 0x0fU;
                const unsigned hi = (row[o.qh + 4U * (e / 16U) + i] >> (2U * ((e % 16U) / 4U))) & 3U;
                block[2U + e] = static_cast<unsigned char>(static_cast<signed char>(static_cast<int>(lo | hi << 4U) - 32));
            }
        }
}

}  // namespace strix::q6p
