// Included inside strix::q8_rows after the dense A8 types and quantizer.
// Q8_0 only. Keep GSQ and Q6P32 adapters outside this component.
inline bool dense_a8_small_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRIX_DENSE_A8_WMMA_SMALL");
        return v && std::strcmp(v, "1") == 0;
    }();
    return enabled && !known_window_cohort_feature_off("densesmall"); // in-process A/B: STRIX_COHORT_AB_OFF=densesmall
}
// One line on the first small-row launch, so a serve log shows that the lever is engaged.
inline void dense_a8_small_note_active(unsigned rows) {
    static const bool once = [rows] {
        std::fprintf(stderr, "lever active: STRIX_DENSE_A8_WMMA_SMALL (first launch: %u rows)\n", rows);
        return true;
    }();
    static_cast<void>(once);
}

template <class Site>
__global__ __launch_bounds__(128) void project_rows_dense_a8_small_kernel(
    Site site, const std::int8_t* codes, const float* deltas) {
#pragma clang fp contract(off)
    constexpr unsigned Rows = Site::rows;
    static_assert(Rows >= 2U && Rows <= 8U);
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U, r = lane & 15U;
    const unsigned width = site.width, pairs = width / 64U;
    const unsigned first = (blockIdx.x * 4U + wave) * 16U;
    constexpr int magic_i = 0x4b400000;
    const dense_a8_v8i magic{magic_i, magic_i, magic_i, magic_i, magic_i, magic_i, magic_i, magic_i};
    // Persistent output tiles preserve the launcher's limited residency contract.
#pragma clang loop unroll(disable)
    for (unsigned n0 = first; n0 < site.outputs; n0 += gridDim.x * 64U) {
        unsigned segment; std::uint32_t row;
        const auto* w = segment_row(site.segments,
            min(n0 + dense_a8_small::weight_row(lane), site.outputs - 1U), width, segment, row);
        float acc[8]{};
        // Runtime K loop must stay rolled: never expand the weight scan into code.
#pragma clang loop unroll(disable)
        for (unsigned p = 0; p < pairs; ++p) {
            const auto* src = reinterpret_cast<const DenseA8Dword4*>(w + 68U * p);
            const auto v0 = src[0], v1 = src[1], v2 = src[2], v3 = src[3];
            const unsigned q[17]{v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w,
                v2.x, v2.y, v2.z, v2.w, v3.x, v3.y, v3.z, v3.w,
                reinterpret_cast<const unsigned*>(w + 68U * p)[16]};
#pragma unroll
            for (unsigned s = 0; s < 2U; ++s) {
                dense_a8_v4i a0, a1, b0{}, b1{};
#pragma unroll
                for (unsigned k = 0; k < 4U; ++k) {
                    a0[k] = static_cast<int>(dense_a8_small::code_word(q, s, 0U, k));
                    a1[k] = static_cast<int>(dense_a8_small::code_word(q, s, 1U, k));
                }
                float dw = __half2float(__ushort_as_half(
                    static_cast<unsigned short>(dense_a8_small::scale_bits(q, s))));
                asm volatile("" : "+v"(dw));
                float da = 0.0F;
                // Only real rows are read. Loads execute uniformly, before register
                // selection, rather than under a partial EXEC mask. Padding is B=0.
#pragma unroll
                for (unsigned j = 0; j < Rows; ++j) {
                    const auto* x = reinterpret_cast<const DenseA8Dword4*>(
                        codes + static_cast<std::uint64_t>(j) * width + 32U * (2U * p + s));
                    auto lo = x[0], hi = x[1];
                    float d = deltas[(2U * p + s) * kDenseA8Rows + j];
                    asm volatile("" : "+v"(lo.x), "+v"(lo.y), "+v"(lo.z), "+v"(lo.w),
                        "+v"(hi.x), "+v"(hi.y), "+v"(hi.z), "+v"(hi.w), "+v"(d));
                    const dense_a8_v4i low{static_cast<int>(lo.x), static_cast<int>(lo.y), static_cast<int>(lo.z), static_cast<int>(lo.w)};
                    const dense_a8_v4i high{static_cast<int>(hi.x), static_cast<int>(hi.y), static_cast<int>(hi.z), static_cast<int>(hi.w)};
#pragma unroll
                    for (unsigned k = 0; k < 4U; ++k) {
                        b0[k] = r == j ? low[k] : b0[k];
                        b1[k] = r == j ? high[k] : b1[k];
                    }
                    da = r == j ? d : da;
                }
                auto c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b0, magic, false);
                c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, c, false);
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) {
                    float scale = __shfl(dw, static_cast<int>(dense_a8_small::scale_lane(lane, e)), 32);
                    // All source lanes participate before masking the epilogue.
                    asm volatile("" : "+v"(scale));
                    if (r < Rows) acc[e] = __builtin_fmaf(scale * da, __int_as_float(c[e]) - 12582912.0F, acc[e]);
                }
            }
        }
        if (r < Rows) {
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned n = n0 + dense_a8_small::output_row(lane, e);
                if (n < site.outputs) site.store(n, r, dense_a8_canonical(acc[e]));
            }
        }
    }
}

template <class Site>
void launch_dense_a8_small(const Site& site, const DenseA8Slot& slot, hipStream_t stream) {
    constexpr auto kernel = project_rows_dense_a8_small_kernel<Site>;
    if (!site.outputs) return;
    dense_a8_small_note_active(Site::rows);
    const unsigned blocks = std::min((site.outputs + 63U) / 64U, resident_blocks<kernel, 128U>(0U));
    hipLaunchKernelGGL(kernel, dim3(blocks), dim3(128U), 0, stream, site, slot.codes, slot.deltas);
}
