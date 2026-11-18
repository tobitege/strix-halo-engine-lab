// Included inside glm_pf_exact_moe.hip's anonymous namespace (Args, Tile, XRows, kSharedRows): the FAST
// glm-pf-wmma-x routed + shared linears (STRIX_GLM_PF_WMMA=1, replaces V11's X3 kernels; plan, transforms,
// activation and combine use the optional auxiliary epilogue policies). Part: 0 = the whole span on 64-row tiles (shared slices, ROWS=64),
// 1 = a routed group's full 128-row tiles, 2 = its remainder on 64-row storage (the tail kernel does not inherit the
// 128-row kernel's LDS / VGPR allocation). tail128: a remainder of 65..127 rows runs as one more 128-row tile in part 1
// (two K traversals for a typical 228-row group instead of three); part 2 then keeps remainders of up to 64 rows.
// split S (part 1 only): grid.y = E x S, CTA slot s runs the group's 128-row tiles s, s + S, ... (hot experts' tiles in
// parallel CTAs). Shared with Part 1 (P3 of GLMPF2O, STRIX_GLM_PF_WMMA_SHARED128, default on with 128-row tiles): the shared
// expert's rows are sliced by 128 instead of kSharedRows = 64, so each weight tile is decoded once per 128 rows (the shared span
// runs the 128-row core, its last partial slice the smallest covering tile). Every row takes the same arithmetic in all forms
// (same bits).
template<int Bits, bool Down, bool Shared, unsigned Part = 0, unsigned CP = 1>
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(6))) void linear_kernel_w(Args a, unsigned tail128, unsigned split = 1U, unsigned all_frag = 0U) {
    __shared__ wmma_x::StorageR<Part == 1U ? 128U : 64U> lds;
    const unsigned expert = Shared ? E : (Part == 1U ? blockIdx.y / split : blockIdx.y);
    const Tile group = a.groups[expert];
    constexpr unsigned kSlice = Shared && Part == 1U ? 128U : kSharedRows;
    const unsigned r0 = Shared ? blockIdx.y * kSlice : 0U;
    if (r0 >= group.count) return;  // uniform, before any barrier
    const unsigned n = Shared ? (group.count - r0 < kSlice ? group.count - r0 : kSlice) : group.count;
    const unsigned full = !Part || Shared ? 0U : (tail128 && n % 128U > 64U ? n : n / 128U * 128U);
    if (!Shared && ((Part == 1U && full == 0) || (Part == 2U && full == n))) return;  // uniform
    const unsigned first_m = Down ? 2U : blockIdx.z;
    const unsigned end_m = !Down && a.aux_act ? 2U : first_m + 1U;
    for (unsigned m = first_m; m < end_m; ++m) {
        const auto w = *d::uniform((&a.experts[expert].gate) + m);
        if constexpr (Part == 1U && !Shared) {  // every tile here has more than 64 rows: the 128-row core
            for (unsigned t = blockIdx.y % split * 128U; t < full; t += split * 128U) {
                const unsigned k = full - t < 128U ? full - t : 128U;
                wmma_x::core<Bits, 8, CP>(w, XRows<Down>{a, Tile{expert, group.first + t, k}, m}, k, lds, MoeAuxEpilogue<Down>{}, {}, all_frag != 0U);
            }
        } else {
        const unsigned b = Part == 2U ? full : 0U;
        wmma_x::span<Bits, CP>(w, [&](unsigned r, unsigned k) { return XRows<Down>{a, Tile{expert, group.first + r0 + b + r, k}, m}; },
                           n - b, lds, MoeAuxEpilogue<Down>{}, {}, all_frag != 0U);
        }
    }
}
// Routed = false: the shared expert only (FAST_I4 runs the routed experts on the IU4 core).
template<bool Down, bool Routed = true>
void launch_linear_w(Args a, unsigned shared_bits, hipStream_t stream) {
    constexpr unsigned N = Down ? H : I;
    const bool shared128 = wmma_x::tile_rows() == 128U && wmma_x::shared128();
    const unsigned slice = shared128 ? 128U : kSharedRows;
    const dim3 routed(N / 128U, E, Down || a.aux_act ? 1U : 2U), shared(N / 128U, (a.rows + slice - 1U) / slice, Down || a.aux_act ? 1U : 2U);
    const unsigned af = wmma_x::all_fragments();
    // Shared slices: 128 rows on the 128-row core (shared128), else kSharedRows = 64 rows.
    if constexpr (!Routed) {
    } else if (wmma_x::tile_rows() == 128U) {
        const unsigned t128 = wmma_x::tail128(), split = wmma_x::split();
        const dim3 full_grid(N / 128U, E * split, Down || a.aux_act ? 1U : 2U);
        if (wmma_x::cols_per_wave() == 2U) linear_kernel_w<4, Down, false, 1, 2><<<full_grid, 256, 0, stream>>>(a, t128, split, af);
        else linear_kernel_w<4, Down, false, 1><<<full_grid, 256, 0, stream>>>(a, t128, split, af);
        linear_kernel_w<4, Down, false, 2><<<routed, 256, 0, stream>>>(a, t128, 1U, af);
    } else {
        linear_kernel_w<4, Down, false><<<routed, 256, 0, stream>>>(a, 0U, 1U, af);
    }
    if constexpr (Routed) ok(hipGetLastError(), "WMMA routed linear");
    if (shared128 && shared_bits == 6U) linear_kernel_w<6, Down, true, 1><<<shared, 256, 0, stream>>>(a, 0U, 1U, af);
    else if (shared128) linear_kernel_w<4, Down, true, 1><<<shared, 256, 0, stream>>>(a, 0U, 1U, af);
    else if (shared_bits == 6U) linear_kernel_w<6, Down, true><<<shared, 256, 0, stream>>>(a, 0U, 1U, af);
    else linear_kernel_w<4, Down, true><<<shared, 256, 0, stream>>>(a, 0U, 1U, af);
    ok(hipGetLastError(), "WMMA shared linear");
}
