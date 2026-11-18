#pragma once
#include "strix/cuda/core/nvfp4_a8_cuda.hpp"

inline bool check_device(const std::array<std::uint8_t, 34>& q8, bool invalid) {
    namespace a8 = strix::cuda::core::nvfp4_a8;
    struct Storage {
        void* p = nullptr;
        ~Storage() { if (p) cudaFree(p); }
    } storage;
    auto ck = [](cudaError_t e) {
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    };
    try {
        ck(cudaMalloc(&storage.p, 128));
        auto* bytes = static_cast<std::uint8_t*>(storage.p);
        auto* scratch = reinterpret_cast<float*>(bytes + 64);
        auto* flag = reinterpret_cast<int*>(bytes + 68);
        ck(cudaMemcpy(bytes, q8.data(), q8.size(), cudaMemcpyHostToDevice));
        a8::Image image;
        image.codes = bytes + 80;
        image.scales = bytes + 96;
        // The amax scan is the only nonfinite check (it reads every value the quantize pass reads).
        {
            ck(cudaMemset(scratch, 0, sizeof(float)));
            ck(cudaMemset(flag, 0, sizeof(int)));
            ck(a8::amax_q8_0_enqueue(bytes, 1, 32, scratch, flag, nullptr));
            int detected = 0;
            ck(cudaMemcpy(&detected, flag, sizeof detected, cudaMemcpyDeviceToHost));
            if (detected != (invalid ? 1 : 0)) return false;
        }
        try {
            ck(a8::convert_q8_0(bytes, 1, 32, image, scratch, nullptr));
        } catch (const std::invalid_argument& e) {
            return invalid && std::strcmp(e.what(), "nonfinite q8_0 weight") == 0;
        }
        if (invalid) return false;
        std::array<std::uint8_t, 16> expected_codes{}, actual_codes{};
        std::array<std::uint8_t, 2> expected_scales{}, actual_scales{};
        float expected_ts = 0;
        a8::reference_convert_q8_0(q8.data(), 1, 32, expected_codes.data(), expected_scales.data(), &expected_ts);
        ck(cudaMemcpy(actual_codes.data(), image.codes, actual_codes.size(), cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(actual_scales.data(), image.scales, actual_scales.size(), cudaMemcpyDeviceToHost));
        return expected_codes == actual_codes && expected_scales == actual_scales &&
               std::memcmp(&expected_ts, &image.tensor_scale, sizeof(float)) == 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return false;
    }
}
