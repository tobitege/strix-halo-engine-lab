// MC certificate: |w| <= 1 and every stored raw dot |d| < 2^120 imply all ten FP32 FMA
// intermediates < 2^124. Admitted calls cannot set merge bit 8. Unproven calls execute the
// original merge_z body and publish only after its final header is known. No public admission bits.
// A row status implies header[0] != 0 before merge (production CSR/projection invariant).
// Prune (moe_core/route_prune.hpp): a zero-weight slot was left out of the route plan, its raw dots are stale and are
// replaced by 0 before the chain (the chain keeps its ten steps; 0 * 0 adds nothing).
template <bool Acc, bool RouteBF16 = false, bool RouteI8 = false, bool Prune = false>
__device__ __forceinline__ void moe_prefill_merge_commit_element(std::uint32_t* header, std::uint32_t* status,
    const std::uint32_t* routes, const float* rawdots, const float* route_weights, float* accumulation, std::uint16_t* output,
    std::uint32_t out, std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k) {
    if (header[0] || *status) { if (Acc && out < hidden_size) accumulation[out] = 0.0F; return; }
    if (out >= hidden_size) return;
    for (std::uint32_t slot = 0; slot < top_k; ++slot) if (routes[slot] >= experts) {
        if (Acc) accumulation[out] = 0.0F;
        output[out] = 0U;
        return;
    }
    float dots[10];
#pragma unroll
    for (std::uint32_t slot = 0; slot < 10U; ++slot) if (slot < top_k) dots[slot] = moe_route_load<RouteBF16, RouteI8>(rawdots, static_cast<std::uint64_t>(slot) * hidden_size + out, hidden_size, top_k);
    float current = 0.0F;
#pragma unroll
    for (std::uint32_t slot = 0; slot < 10U; ++slot) if (slot < top_k) {
        const float weight = route_weights[slot];
        if constexpr (Prune) if (weight == 0.0F) dots[slot] = 0.0F;
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current) : "s"(weight), "v"(dots[slot]));
    }
    if (Acc) accumulation[out] = current;
    const auto bits = __float_as_uint(current);
    if ((bits & 0x7fffffffU) >= 0x7f7f8000U) { atomicOr(status, 8U); atomicOr(header, 8U); return; }
    output[out] = static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
// The base commit's block (0, 0) without its element stores: one 256-thread block after the merge. RowStatus(r) returns
// row r's status word.
template <class RowStatus>
__device__ __forceinline__ void moe_prefill_commit_record(std::uint32_t* header, std::uint32_t row_count,
    const RowStatus& row_status, DeferredMoeFirstError* record) {
    const std::uint32_t failure = header[0];
    __shared__ std::uint32_t first_bad;
    if (threadIdx.x == 0U) first_bad = row_count;
    __syncthreads();
    for (std::uint32_t r = threadIdx.x; r < row_count; r += blockDim.x)
        if (*row_status(r)) { atomicMin(&first_bad, r); break; }
    __syncthreads();
    if (threadIdx.x == 0U) {
        if (failure & 7U) for (const auto word : {1U, 2U, 4U, 5U}) header[word] = 0U;
        if (failure & 7U) for (std::uint32_t word = 8U; word < 24U; ++word) header[word] = 0U;
        if (first_bad < row_count)
            merge_deferred_moe_first_error(*record, *row_status(first_bad), DeferredMoeDomain::q4k_a8);
    }
}

template <bool Fused = false, bool Acc = false, bool RouteBF16 = false, bool RouteI8 = false, bool Prune = false, class View, class RowAt>
__device__ __forceinline__ void moe_prefill_merge_z_body(View view, std::uint32_t* header,
    std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k, const RowAt& row_at, const unsigned* unsafe = nullptr) {
    const auto row = blockIdx.y;
    auto* status = row_at(view.status, row);
    const auto out = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (header[0] || *status) { if (out < hidden_size) row_at(view.accumulation, row)[out] = 0.0F; return; }
    // The private word is read after the original header/status guards, before any changed stores.
    if constexpr (Fused) if (*unsafe == 0U) {
        moe_prefill_merge_commit_element<Acc, RouteBF16, RouteI8, Prune>(header, status, row_at(view.routes, row), row_at(view.rawdots, row),
            row_at(view.route_weights, row), row_at(view.accumulation, row), row_at(view.output, row),
            out, hidden_size, experts, top_k);
        return;
    }
    if (out >= hidden_size) return;
    const auto* routes = row_at(view.routes, row);
    for (std::uint32_t slot = 0; slot < top_k; ++slot) if (routes[slot] >= experts) { row_at(view.accumulation, row)[out] = 0.0F; return; }
    const auto* rawdots = row_at(view.rawdots, row);
    const auto* route_weights = row_at(view.route_weights, row);
    float dots[10];
#pragma unroll
    for (std::uint32_t slot = 0; slot < 10U; ++slot) if (slot < top_k) dots[slot] = moe_route_load<RouteBF16, RouteI8>(rawdots, static_cast<std::uint64_t>(slot) * hidden_size + out, hidden_size, top_k);
    float current = 0.0F;
#pragma unroll
    for (std::uint32_t slot = 0; slot < 10U; ++slot) if (slot < top_k) {
        const float weight = route_weights[slot];
        if constexpr (Prune) if (weight == 0.0F) dots[slot] = 0.0F;
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current) : "s"(weight), "v"(dots[slot]));
    }
    row_at(view.accumulation, row)[out] = current;
    if ((__float_as_uint(current) & 0x7fffffffU) >= 0x7f7f8000U) { atomicOr(status, 8U); atomicOr(header, 8U); }
}

// Vector MC merge (STRIX_PREFILL_MOE_MERGE_VEC, default on): moe_prefill_merge_z_body<true, Acc> with four adjacent outputs per
// thread; each output keeps its own guards and its slot 0..top_k-1 v_fmac chain, only loads/stores are widened. Admitted
// calls (*unsafe == 0) cannot set merge bit 8 (MC certificate): header/status are fixed during the launch and every stored
// word equals the scalar body's. Unproven calls: identical words while no sum is non-finite. When one is, bit 8 fails the
// call; outputs whose guard already sees it keep a zero accumulation in both bodies (which ones is schedule-dependent in
// the scalar body as well); every row the merge flags gets status exactly 8 (header and status were 0 before the merge),
// so the finish record ({q4k_a8, 8}: the bits of the first flagged row, not its index) and the thrown completion error are
// the same, and the failed call's accumulation is never read. Host requires hidden % 4 == 0, 16-byte rawdot/accumulation rows and 8-byte output rows.
template <bool Acc, bool RouteBF16 = false, bool RouteI8 = false, bool Prune = false, class View, class RowAt>
__device__ __forceinline__ void moe_prefill_merge_z_body_vec4(View view, std::uint32_t* header,
    std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k, const RowAt& row_at, const unsigned* unsafe) {
    const auto row = blockIdx.y;
    auto* status = row_at(view.status, row);
    const auto out = (static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4U;
    if (out >= hidden_size) return;
    auto* accumulation = row_at(view.accumulation, row) + out;
    auto* output = row_at(view.output, row) + out;
    const auto store4 = [&](float a, float b, float c, float d) { *reinterpret_cast<float4*>(accumulation) = make_float4(a, b, c, d); };
    if (header[0] || *status) { store4(0.0F, 0.0F, 0.0F, 0.0F); return; }
    const bool admitted = *unsafe == 0U;
    const auto* routes = row_at(view.routes, row);
    for (std::uint32_t slot = 0; slot < top_k; ++slot) if (routes[slot] >= experts) {
        if (!admitted || Acc) store4(0.0F, 0.0F, 0.0F, 0.0F);
        if (admitted) *reinterpret_cast<uint2*>(output) = make_uint2(0U, 0U);
        return;
    }
    const auto* rawdots = row_at(view.rawdots, row);
    const auto* route_weights = row_at(view.route_weights, row);
    float4 dots[10];
#pragma unroll
    for (std::uint32_t slot = 0; slot < 10U; ++slot) if (slot < top_k) {
        if constexpr (RouteI8) {
            const auto index = static_cast<std::uint64_t>(slot) * hidden_size + out;
            dots[slot] = moe_route_i8_load4(rawdots,index,hidden_size,top_k);
        } else if constexpr (RouteBF16) {
            const auto index = static_cast<std::uint64_t>(slot) * hidden_size + out;
            const auto packed = *reinterpret_cast<const uint2*>(reinterpret_cast<const std::uint16_t*>(rawdots) + index);
            dots[slot] = make_float4(__uint_as_float((packed.x & 0xffffU) << 16U), __uint_as_float(packed.x & 0xffff0000U),
                __uint_as_float((packed.y & 0xffffU) << 16U), __uint_as_float(packed.y & 0xffff0000U));
        } else
            dots[slot] = *reinterpret_cast<const float4*>(rawdots + static_cast<std::uint64_t>(slot) * hidden_size + out);
    }
    float current[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
    for (std::uint32_t slot = 0; slot < 10U; ++slot) if (slot < top_k) {
        const float weight = route_weights[slot];
        if constexpr (Prune) if (weight == 0.0F) dots[slot] = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current[0]) : "s"(weight), "v"(dots[slot].x));
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current[1]) : "s"(weight), "v"(dots[slot].y));
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current[2]) : "s"(weight), "v"(dots[slot].z));
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current[3]) : "s"(weight), "v"(dots[slot].w));
    }
    if (!admitted || Acc) store4(current[0], current[1], current[2], current[3]);
    bool finite = true;
    std::uint32_t words[4];
#pragma unroll
    for (std::uint32_t j = 0; j < 4U; ++j) {
        const auto bits = __float_as_uint(current[j]);
        finite &= (bits & 0x7fffffffU) < 0x7f7f8000U;
        words[j] = (bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U;
    }
    if (finite) {
        if (admitted) *reinterpret_cast<uint2*>(output) = make_uint2(words[0] | words[1] << 16U, words[2] | words[3] << 16U);
        return;
    }
#pragma unroll
    for (std::uint32_t j = 0; j < 4U; ++j) {
        if ((__float_as_uint(current[j]) & 0x7fffffffU) >= 0x7f7f8000U) { atomicOr(status, 8U); atomicOr(header, 8U); continue; }
        if (admitted) output[j] = static_cast<std::uint16_t>(words[j]);
    }
}

template <class View, class RowAt, class Store>
__device__ __forceinline__ void moe_prefill_commit_body(View view, std::uint32_t* header,
    std::uint32_t row_count, std::uint32_t hidden_size, std::uint32_t* record, unsigned logical_x, unsigned logical_y, const RowAt& row_at, const Store& store) {
    const std::uint32_t failure = header[0];
    if (logical_x == 0U && logical_y == 0U) {
        // The serial merge over rows 0..T-1 records only the first nonzero row status (a merge of
        // a zero status changes nothing): the block finds that row in parallel, thread 0 merges it.
        __shared__ std::uint32_t first_bad;
        if (threadIdx.x == 0U) first_bad = row_count;
        __syncthreads();
        for (std::uint32_t r = threadIdx.x; r < row_count; r += blockDim.x)
            if (*row_at(view.status, r)) { atomicMin(&first_bad, r); break; }
        __syncthreads();
        if (threadIdx.x == 0U) {
            if (failure & 7U) for (const auto word : {1U, 2U, 4U, 5U}) header[word] = 0U;
            if (failure & 7U) for (std::uint32_t word = 8U; word < 24U; ++word) header[word] = 0U;
            if (first_bad < row_count)
                merge_deferred_moe_first_error(*reinterpret_cast<DeferredMoeFirstError*>(record),
                    *row_at(view.status, first_bad), DeferredMoeDomain::q4k_a8);
        }
    }
    if (failure) return;
    const auto out = logical_x * blockDim.x + threadIdx.x;
    if (out < hidden_size) store(row_at(view.output, logical_y) + out,
        row_at(view.accumulation, logical_y)[out]);
}

template <unsigned CTAs, class View, class RowAt, class Store>
__device__ __forceinline__ void moe_prefill_finish_body(View view, unsigned* header, unsigned row_count,
    unsigned hidden, unsigned* record, const unsigned* unsafe, const RowAt& row_at, const Store& store) {
    static_assert(CTAs > 0U);
    if (header[0] || *unsafe == 0U) {
        if (blockIdx.x == 0U)
            moe_prefill_commit_record(header, row_count, [&](unsigned r) { return row_at(view.status, r); },
                reinterpret_cast<DeferredMoeFirstError*>(record));
        return;
    }
    const unsigned tiles=(hidden+blockDim.x-1U)/blockDim.x;
    for (unsigned logical=blockIdx.x; logical<tiles*row_count; logical+=CTAs)
        moe_prefill_commit_body(view, header, row_count, hidden, record, logical%tiles, logical/tiles, row_at, store);
}
