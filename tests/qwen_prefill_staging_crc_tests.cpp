#include <hip/hip_runtime.h>
namespace strix { inline hipStream_t hip_execution_stream() { return nullptr; } }
#include "../src/hip_qwen_prefill_pair_staging.hpp"
#include <iostream>

// Allocation probe only: these tests never run a device CRC kernel.
namespace strix::xport {
unsigned crc_allocations = 0;
unsigned crc_executions = 0;
DeviceCrc32c::~DeviceCrc32c() { delete[] tables_; }
void DeviceCrc32c::allocate(std::size_t) { tables_ = new std::uint32_t[1]; ++crc_allocations; }
void DeviceCrc32c::enqueue(hipStream_t, const void*, std::size_t, std::size_t) {
    ++crc_executions;
}
std::uint32_t DeviceCrc32c::value(std::size_t, std::size_t, std::span<const std::byte>) const {
    ++crc_executions; return 0;
}
}

namespace {
void require(bool condition) {
    if (!condition) throw std::runtime_error("PP seam publication assertion failed");
}
void crc_allocation_checks() {
    using namespace strix::qwen_pp;
    const auto before = strix::xport::crc_allocations;
    for (const auto capacity : {std::size_t{65536}, seam_piece_bytes}) {
        auto staging = make_hip_pipe_staging(capacity);
        staging->route_device({});
        // Host-only and CRC OFF adapters install a nonempty but always-false rule.
        staging->route_device([](std::size_t) { return false; });
        require(strix::xport::crc_allocations == before);
        staging->route_device([](std::size_t bytes) { return bytes > 65536U; });
        const auto expected = before + (capacity > 65536U ? 1U : 0U);
        require(strix::xport::crc_allocations == expected);
        staging->route_device([](std::size_t bytes) { return bytes > 65536U; });
        require(strix::xport::crc_allocations == expected);
    }
}
std::size_t stress(bool device_only) {
    using namespace strix::qwen_pp;
    seam_visibility::device_only = device_only;
    std::mt19937 random(0x505033U);
    std::size_t mismatches = 0;
    for (unsigned trial = 0; trial < 24U; ++trial) {
        // Random short pieces, full pieces and partial final pieces. Two live
        // staging slots model immutable snapshots while source compute resumes.
        const auto size = static_cast<std::size_t>(random() % 65535U) + 1U +
            (trial % 3U == 0U ? seam_piece_bytes : 0U);
        std::array<std::shared_ptr<PipeStaging>, 2> slots;
        std::vector<std::byte> source(size);
        for (auto& byte : source) byte = static_cast<std::byte>(random() & 255U);
        const auto expected = source;
        for (auto& slot : slots) {
            slot = make_hip_pipe_staging(size);
            auto host = slot->resize(size);
            slot->copy(true, 0U, source.data(), size);
            slot->seal(size);
            // The last partial copy is allowed to complete before earlier
            // pieces; completion still does not publish coarse host memory.
            seam_visibility::complete_shuffled(seam_visibility::last_stream, random);
            require(std::all_of(seam_visibility::last_stream->copies.begin(),
                seam_visibility::last_stream->copies.end(), [](const auto& copy) { return copy.complete; }));
            std::fill(source.begin(), source.end(), std::byte{0x5a});
            std::vector<std::size_t> offsets;
            for (std::size_t offset = 0; offset < size; offset += seam_piece_bytes) offsets.push_back(offset);
            std::shuffle(offsets.begin(), offsets.end(), random);
            for (const auto offset : offsets) {
                slot->await_piece(offset);
                const auto length = std::min(seam_piece_bytes, size - offset);
                const auto piece = host.subspan(offset, length);
                SeamCrc actual_crc, expected_crc;
                actual_crc.update(piece);
                expected_crc.update(std::span(expected).subspan(offset, length));
                if (actual_crc.value() != expected_crc.value() ||
                    !std::equal(piece.begin(), piece.end(), expected.begin() + static_cast<std::ptrdiff_t>(offset))) ++mismatches;
            }
            source = expected;
        }
    }
    return mismatches;
}
// Strided pack/scatter: three components gathered from strided rows equal the per-row packing, both ways.
void strided_rows() {
    using namespace strix::qwen_pp;
    seam_visibility::device_only = false;
    constexpr std::size_t rows = 37U, row_bytes = 40U, stride = 96U;
    const std::array<std::size_t, 3> lengths{24U, 12U, 4U};
    std::vector<std::byte> device(rows * stride * 3U);
    for (std::size_t index = 0; index < device.size(); ++index) device[index] = static_cast<std::byte>((index * 131U + 7U) & 255U);
    auto slot = make_hip_pipe_staging(rows * row_bytes);
    auto host = slot->resize(rows * row_bytes);
    std::size_t part_offset = 0;
    for (std::size_t part = 0; part < lengths.size(); ++part) {
        slot->copy_rows(true, part_offset, row_bytes, device.data() + part * rows * stride, stride, lengths[part], rows);
        part_offset += lengths[part];
    }
    slot->seal(rows * row_bytes);
    std::mt19937 order(5U);
    seam_visibility::complete_shuffled(seam_visibility::last_stream, order);
    slot->await_piece(0U);
    for (std::size_t row = 0; row < rows; ++row) {
        std::size_t offset = row * row_bytes;
        for (std::size_t part = 0; part < lengths.size(); ++part) {
            const auto* source = device.data() + part * rows * stride + row * stride;
            require(std::equal(source, source + lengths[part], host.begin() + static_cast<std::ptrdiff_t>(offset)));
            offset += lengths[part];
        }
    }
    std::vector<std::byte> scattered(device.size(), std::byte{0});
    part_offset = 0;
    for (std::size_t part = 0; part < lengths.size(); ++part) {
        slot->copy_rows(false, part_offset, row_bytes, scattered.data() + part * rows * stride, stride, lengths[part], rows);
        part_offset += lengths[part];
    }
    for (std::size_t part = 0; part < lengths.size(); ++part)
        for (std::size_t row = 0; row < rows; ++row) {
            const auto at = part * rows * stride + row * stride;
            require(std::equal(device.begin() + static_cast<std::ptrdiff_t>(at), device.begin() + static_cast<std::ptrdiff_t>(at + lengths[part]),
                scattered.begin() + static_cast<std::ptrdiff_t>(at)));
        }
}
}
int main() {
    try {
        crc_allocation_checks();
        const auto before = stress(true);
        const auto after = stress(false);
        require(before > 0U && after == 0U);
        // Direct receive installs the immutable snapshot first. CPU CRC must
        // see the received bytes through the system-release readback event.
        for (const auto size : {std::size_t{65537}, strix::qwen_pp::seam_piece_bytes + 17U}) {
            auto slot = strix::qwen_pp::make_hip_pipe_staging(size);
            auto host = slot->resize(size);
            std::memset(slot->device_piece(0), 0x37, size);
            for (std::size_t offset = 0; offset < size; offset += strix::qwen_pp::seam_piece_bytes)
                slot->read_device_piece(offset, std::min(strix::qwen_pp::seam_piece_bytes, size - offset));
            require(std::all_of(host.begin(), host.end(), [](auto byte) { return byte == std::byte{0x37}; }));
        }
        strided_rows();
        require(strix::xport::crc_executions == 0U);
        std::cout << "PASS: 24 seeded sizes x 2 slots; shuffled copy/fence order, source reuse and partial tails; device-only mismatches="
            << before << ", system-release mismatches=" << after << " (host model, not device qualification)\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
