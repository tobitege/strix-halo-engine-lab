#include "strix/vision/preprocess.hpp"

#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>

namespace strix::vision {
namespace {

[[noreturn]] void bad(const char* what) { throw ImageError(ImageErrorCode::invalid_argument, std::string("vision: ") + what); }

}  // namespace

std::vector<float> normalize(const Image& im, std::array<float, 3> mean, std::array<float, 3> std) {
    if (im.width <= 0 || im.height <= 0 || im.rgb.size() != im.pixels() * 3) bad("image size and buffer disagree");
    std::vector<float> out(im.rgb.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        const std::size_t c = i % 3;
        out[i] = (float(im.rgb[i]) * (1.0F / 255.0F) - mean[c]) / std[c];
    }
    return out;
}

std::vector<float> normalize_div255(const Image& im, std::array<float, 3> mean, std::array<float, 3> std, bool round_bf16) {
    if (im.width <= 0 || im.height <= 0 || im.rgb.size() != im.pixels() * 3) bad("image size and buffer disagree");
    std::array<std::array<float, 256>, 3> table{};
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t level = 0; level < 256; ++level) {
            float v = (float(level) / 255.0F - mean[c]) / std[c];
            if (round_bf16) {  // nearest BF16, ties to even (finite values)
                std::uint32_t u = std::bit_cast<std::uint32_t>(v);
                u = (u + 0x7FFFU + ((u >> 16) & 1U)) & 0xFFFF0000U;
                v = std::bit_cast<float>(u);
            }
            table[c][level] = v;
        }
    std::vector<float> out(im.rgb.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = table[i % 3][im.rgb[i]];
    return out;
}

void normalize_inplace(std::span<float> rgb, std::array<float, 3> mean, std::array<float, 3> std) {
    if (rgb.size() % 3 != 0) bad("normalize needs [n][3] samples");
    for (std::size_t i = 0; i < rgb.size(); ++i) {
        const std::size_t c = i % 3;
        rgb[i] = (rgb[i] - mean[c]) / std[c];
    }
}

Patches patchify(std::span<const float* const> frames, int height, int width, const PatchSpec& spec) {
    const int cell = spec.patch * spec.merge;
    if (spec.patch <= 0 || spec.temporal <= 0 || spec.merge <= 0) bad("patch spec");
    if (frames.size() != 1 && frames.size() != std::size_t(spec.temporal)) bad("patchify needs 1 or `temporal` frames");
    for (const float* f : frames)
        if (f == nullptr) bad("patchify frame");
    if (height <= 0 || width <= 0 || height % cell != 0 || width % cell != 0)
        bad("patchify size is not a multiple of patch x merge");
    Patches p;
    p.grid_h = height / spec.patch;
    p.grid_w = width / spec.patch;
    p.merge = spec.merge;
    p.feature = 3 * spec.temporal * spec.patch * spec.patch;
    p.data.resize(p.rows() * std::size_t(p.feature));
    float* dst = p.data.data();
    for (int bh = 0; bh < p.grid_h / spec.merge; ++bh)
        for (int bw = 0; bw < p.grid_w / spec.merge; ++bw)
            for (int mh = 0; mh < spec.merge; ++mh)
                for (int mw = 0; mw < spec.merge; ++mw) {
                    const int y0 = (bh * spec.merge + mh) * spec.patch, x0 = (bw * spec.merge + mw) * spec.patch;
                    for (int c = 0; c < 3; ++c)
                        for (int t = 0; t < spec.temporal; ++t) {
                            const float* fr = frames[frames.size() == 1 ? 0 : std::size_t(t)];
                            for (int py = 0; py < spec.patch; ++py) {
                                const float* src =
                                    fr + (std::size_t(y0 + py) * std::size_t(width) + std::size_t(x0)) * 3 + std::size_t(c);
                                for (int px = 0; px < spec.patch; ++px) *dst++ = src[std::size_t(px) * 3];
                            }
                        }
                }
    return p;
}

std::string pixel_digest(std::string_view adapter_identity, const Image& resized) {
    if (resized.width <= 0 || resized.height <= 0 || resized.rgb.size() != resized.pixels() * 3)
        bad("image size and buffer disagree");
    // Length-prefixed fields, little-endian, so no two (identity, size, pixels) triples share a byte stream.
    std::array<std::byte, 16> head{};
    const auto put = [&](std::size_t at, std::uint64_t v, int bytes) {
        for (int i = 0; i < bytes; ++i) head[at + std::size_t(i)] = std::byte((v >> (8 * i)) & 0xFF);
    };
    put(0, adapter_identity.size(), 8);
    put(8, std::uint64_t(resized.width), 4);
    put(12, std::uint64_t(resized.height), 4);
    Sha256Fast sha;
    sha.update(std::span<const std::byte>(head.data(), 8));
    sha.update(std::as_bytes(std::span<const char>(adapter_identity.data(), adapter_identity.size())));
    sha.update(std::span<const std::byte>(head.data() + 8, 8));
    sha.update(std::as_bytes(std::span<const std::uint8_t>(resized.rgb)));
    return model_content_identity_hex(sha.finalize());
}

void parallel_for(std::size_t count, unsigned max_threads, const std::function<void(std::size_t)>& fn) {
    if (count == 0) return;
    unsigned threads = max_threads != 0 ? max_threads : std::max(1U, std::thread::hardware_concurrency());
    threads = unsigned(std::min<std::size_t>(threads, count));
    std::atomic<std::size_t> next{0};
    std::exception_ptr error;
    std::mutex error_mutex;
    const auto work = [&] {
        for (std::size_t i = next.fetch_add(1); i < count; i = next.fetch_add(1)) {
            try {
                fn(i);
            } catch (...) {
                const std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
                next.store(count);
            }
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    for (unsigned t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
    if (error) std::rethrow_exception(error);
}

}  // namespace strix::vision
