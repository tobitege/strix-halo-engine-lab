// Included inside namespace strix by the launcher and standalone fixture.
// Prefill key / value projections of 64 strided rows through the wide q8_rows kernels, with the
// arithmetic of project_wave8_q8_bf16_to_fp32_rows_kernel per row: coefficient scale * code, chain
// acc += x * c over features 8L + l + 256k, shuffle-down lane tree then the leaf tree.
__device__ __forceinline__ float ple_rows_tree_add(float lower, float upper) {
    float result;
    asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(upper), "v"(lower));
    return result;
}
template <unsigned Rows> struct PleRowsSite {
    using Input = std::uint16_t;
    static constexpr unsigned rows = Rows;
    static constexpr bool fused_fma = true;  // Site::fma compiles to v_fmac_f32 (wide VOPD rows)
    static constexpr bool tree_upper_src0 = true;  // add(lower, upper) reads upper as src0 (lane VOPD tree)
    q8_rows::Segments segments;
    std::uint32_t width, outputs, in_stride, out_stride;
    const std::uint16_t* base;
    float* out;
    __device__ const std::uint16_t* row_input(unsigned r) const { return base + static_cast<std::uint64_t>(r) * in_stride; }
    __device__ static float coefficient(unsigned scale, std::int8_t code) {
        return __half2float(__ushort_as_half(static_cast<unsigned short>(scale))) * static_cast<float>(code);
    }
    __device__ static void fma(float& acc, float c, float x) { acc += x * c; }
    __device__ static float add(float lower, float upper) { return ple_rows_tree_add(lower, upper); }
    __device__ void store(std::uint32_t o, unsigned r, float value) const { out[static_cast<std::uint64_t>(r) * out_stride + o] = value; }
};
constexpr std::uint32_t kPleWideRows = 64U;

// Grouped sites preserve the 64-row kernel body, including blockIdx.x's output/wave mapping.
// Only row addresses and the packed FP32 slab acquire a blockIdx.y offset.
template <unsigned Rows> struct PleRowsGroupSite : PleRowsSite<Rows> {
    static constexpr bool grouped_rows = true;
    __device__ const std::uint16_t* row_input(unsigned r) const {
        return PleRowsSite<Rows>::row_input(blockIdx.y * Rows + r);
    }
    __device__ void store(std::uint32_t o, unsigned r, float value) const {
        PleRowsSite<Rows>::store(o, blockIdx.y * Rows + r, value);
    }
};

// Batch only the default lane recipe. Other dispatch/diagnostic settings retain their old path.
inline bool ple_batch_admits(const PleRowsSite<64>& site, unsigned groups) {
    static const bool recipe = strix::env_number<unsigned>("STRIX_Q8_ROWS_LANE_WAVES", 16U) == 16U &&
        strix::env_number<unsigned>("STRIX_Q8_ROWS_LANE_STAGE", 8U) == 8U &&
        strix::env_number<unsigned>("STRIX_Q8_ROWS_LANE_PREFETCH", 1U) == 1U &&
        strix::env_number<unsigned>("STRIX_Q8_ROWS_LANE_EXP", 0U) == 0U;
    if (!recipe || !groups || groups > 65535U || !site.width ||
        q8_rows::wide_kernel() != 2U || q8_rows::wide_stage() || !q8_rows::lane_admits(site) ||
        site.width > UINT32_MAX / 64U ||
        (reinterpret_cast<std::uintptr_t>(site.base) | (static_cast<std::uint64_t>(site.in_stride) * 2U)) % 16U)
        return false;
    for (unsigned m = 0; m < site.segments.count; ++m)
        if (reinterpret_cast<std::uintptr_t>(site.segments.weights[m]) % 16U) return false;
    return true;
}

// Unlike reserve(), a failed allocation leaves the existing buffer usable by the fallback.
// The old allocation may still be read by the preceding window: join before releasing it.
inline bool ple_batch_reserve(q8_rows::LaneBuffer& rows, std::uint64_t need) {
    if (need <= rows.bytes) return true;
    void* next = nullptr;
    if (hipMalloc(&next, need) != hipSuccess) { (void)hipGetLastError(); return false; }
    if (rows.base && (hipDeviceSynchronize() != hipSuccess || hipFree(rows.base) != hipSuccess)) {
        (void)hipFree(next);
        throw std::runtime_error("PLE batch scratch release");
    }
    rows.base = next; rows.bytes = need;
    return true;
}

inline void ple_batch_pack(const PleRowsSite<64>& original, unsigned groups, float* packed, hipStream_t stream) {
    PleRowsGroupSite<64> site{};
    static_cast<PleRowsSite<64>&>(site) = original;
    hipLaunchKernelGGL(q8_rows::lane_pack_rows_kernel<PleRowsGroupSite<64>>,
        dim3((64U * site.width + 255U) / 256U, groups), dim3(256U), 0, stream, site, packed);
}

inline void ple_batch_project(const PleRowsSite<64>& original, q8_rows::LaneWeights weights,
    unsigned groups, const float* packed, hipStream_t stream) {
    PleRowsGroupSite<64> site{};
    static_cast<PleRowsSite<64>&>(site) = original;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_rows::project_wide_rows_lane_kernel<PleRowsGroupSite<64>, 16U, 8U, 2U>),
        dim3((site.outputs + q8_rows::kLaneOutputs - 1U) / q8_rows::kLaneOutputs, groups), dim3(512U),
        0, stream, site, weights.codes, weights.scales, packed);
}

inline void ple_batch_launch(const PleRowsSite<64>& site, q8_rows::LaneWeights weights,
    unsigned groups, float* packed, hipStream_t stream) {
    // Keep every AQL barrier bit: a later pack cannot overwrite earlier consumers.
    ple_batch_pack(site, groups, packed, stream);
    ple_batch_project(site, weights, groups, packed, stream);
}

inline bool ple_tile_requested() {
    static const bool on = ple_tile::requested(std::getenv("STRIX_QPF2K_PLE_TILE"));
    return on;
}

inline bool ple_tile_admits(bool enabled, const PleRowsSite<64>& key, const PleRowsSite<64>& value, unsigned groups) {
    return key.base == value.base && key.width == value.width && key.in_stride == value.in_stride &&
        ple_tile::admits(enabled, {reinterpret_cast<std::uintptr_t>(key.base), key.width, key.in_stride},
            {reinterpret_cast<std::uintptr_t>(key.out), key.outputs, key.out_stride},
            {reinterpret_cast<std::uintptr_t>(value.out), value.outputs, value.out_stride}, groups);
}

// Both sites and both weight packs are exactly those of the original transaction.
// Only reuse the identical, read-only input panel across the two output tilings.
inline void ple_tile_launch(const PleRowsSite<64>& key, const PleRowsSite<64>& value,
    unsigned groups, float* packed, hipStream_t stream) {
    const auto kw = q8_rows::lane_weights(key, stream);
    ple_batch_pack(key, groups, packed, stream);
    ple_batch_project(key, kw, groups, packed, stream);
    const auto vw = q8_rows::lane_weights(value, stream);
    ple_batch_project(value, vw, groups, packed, stream);
}
