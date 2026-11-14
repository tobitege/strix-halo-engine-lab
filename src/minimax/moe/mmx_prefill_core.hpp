#pragma once

#include "minimax/engine/route_identity.hpp"
#include "minimax/moe/mmx_moe_traits.hpp"
#include "../../moe_core/grouped_core.hpp"

namespace strix::minimax::moe::opt::pfk {
using strix::moe_core::Half16;
using strix::moe_core::Float8;
using strix::moe_core::Item;
template <int Bits> using Window = typename WeightDecode<Bits>::Window;
template <int Bits> using Norm = typename WeightDecode<Bits>::Norm;
template <int MT> using Accum = strix::moe_core::Accum<MT>;
using Core = strix::moe_core::GroupedCore<WeightDecode, ExpertLayout, InputTransform, Epilogue>;
template <bool Packed> using PipeCore = strix::moe_core::GroupedCore<WeightDecode, ExpertLayout, InputTransform, Epilogue,
                                                                  MinimaxPrefillTraits<Packed>>;

__device__ __forceinline__ float codebook_mul1(std::uint32_t state) {
    return WeightDecode<3>::codebook_value(state);
}

template <int Bits>
__device__ __forceinline__ void load_window(const std::uint32_t* tile, int w0, std::uint32_t (&w)[Window<Bits>::kNW]) {
    return Core::load_window<Bits>(tile, w0, w);
}

template <int Bits>
__device__ __forceinline__ void decode_fragment(const std::uint32_t (&w)[Window<Bits>::kNW], bool h, std::uint32_t (&frag)[8]) {
    return Core::decode_fragment<Bits>(w, h, frag);
}

__device__ __forceinline__ Half16 as_half16(const std::uint32_t (&f)[8]) {
    return Core::as_half16(f);
}

template <int Bits, int MT>
__device__ __forceinline__ void item_core(const Item& it, unsigned bx, unsigned by, unsigned bz) {
    return Core::item_core<Bits, MT>(it, bx, by, bz);
}

template <int Bits, int MT, int Mode, bool Tiled = false, int Core8 = 0, bool Paired = false>
__device__ __forceinline__ Accum<MT> item_acc_v3(const Item& it, unsigned bx, unsigned by, unsigned bz, Half16* exchange = nullptr) {
    return Core::item_acc_v3<Bits, MT, Mode, Tiled, Core8, Paired>(it, bx, by, bz, exchange);
}

template <int Bits, int MT, int Mode, bool Tiled = false, int Core8 = 0>
__device__ __forceinline__ void item_core_v3(const Item& it, unsigned bx, unsigned by, unsigned bz, Half16* exchange = nullptr) {
    return Core::item_core_v3<Bits, MT, Mode, Tiled, Core8>(it, bx, by, bz, exchange);
}

__device__ __forceinline__ std::uint32_t pipe_product(std::uint32_t state) {
    return Core::pipe_product(state);
}

__device__ __forceinline__ float pipe_codebook(std::uint32_t state) {
    return Core::pipe_codebook(state);
}

__device__ __forceinline__ Half16 pipe_load_a(const unsigned char* base, std::uint32_t offset) {
    return Core::pipe_load_a(base, offset);
}

template <int Bits, int MT, int G, bool FullRows = true, bool Simplify = false>
__device__ __forceinline__ void item_pipe(const Item& it, unsigned bx, unsigned by, unsigned bz) {
    return Core::item_pipe<Bits, MT, G, FullRows, Simplify>(it, bx, by, bz);
}

template <int Bits, int MT, int G, bool Simplify, bool Packed = false>
__device__ __forceinline__ void pipe_or_tail(const Item& it, unsigned bx, unsigned by, unsigned bz) {
    return PipeCore<Packed>::template pipe_or_tail<Bits, MT, G, Simplify>(it, bx, by, bz);
}

template <int Bits, bool Simplify>
__device__ __forceinline__ void item_pipe_pair(const Item& it, unsigned bx, unsigned by, unsigned bz, Half16* exchange) {
    return Core::item_pipe_pair<Bits, Simplify>(it, bx, by, bz, exchange);
}

template <int Bits, bool Simplify, bool Packed = false>
__device__ __forceinline__ void pipe_pair_or_tail(const Item& it, unsigned bx, unsigned by, unsigned bz, Half16* exchange) {
    return PipeCore<Packed>::template pipe_pair_or_tail<Bits, Simplify>(it, bx, by, bz, exchange);
}

template <int Bits, int W, bool Simplify, bool Packed = false>
__device__ __forceinline__ void item_pipe_band(const Item& it, unsigned bx, unsigned by, unsigned bz, Half16* exchange) {
    return PipeCore<Packed>::template item_pipe_band<Bits, W, Simplify>(it, bx, by, bz, exchange);
}

template <int Bits, int Core8>
__device__ __forceinline__ void item_core8(const Item& it, unsigned bx, unsigned by, unsigned bz, Half16* exchange) {
    return Core::item_core8<Bits, Core8>(it, bx, by, bz, exchange);
}

template <int MT = 8, int G = 4, bool Simplify = false, bool Packed = false>
__global__ __launch_bounds__(128) void grouped_pipe_kernel(const Item* items, unsigned count) {
    unsigned lo = 0, hi = count;
    while (hi - lo > 1) {
        const unsigned mid = (lo + hi) / 2;
        if (items[mid].cta_begin <= blockIdx.x) lo = mid;
        else hi = mid;
    }
    const Item it = items[lo];
    const unsigned local = blockIdx.x - it.cta_begin;
    const unsigned bx = local % it.gx, by = (local / it.gx) % it.gy, bz = local / (it.gx * it.gy);
    switch (it.bits) {
        case 3: pipe_or_tail<3, MT, G, Simplify, Packed>(it, bx, by, bz); break;
        case 4: pipe_or_tail<4, MT, G, Simplify, Packed>(it, bx, by, bz); break;
        case 5: pipe_or_tail<5, MT, G, Simplify, Packed>(it, bx, by, bz); break;
        case 6: pipe_or_tail<6, MT, G, Simplify, Packed>(it, bx, by, bz); break;
        default: pipe_or_tail<8, MT, G, Simplify, Packed>(it, bx, by, bz); break;
    }
}

template <bool Simplify = false, bool Packed = false>
__global__ __launch_bounds__(128) void grouped_pipe_pair_kernel(const Item* items, unsigned count) {
    __shared__ Half16 exchange[128];  // 2 buffers x 4 decoded tiles x 16 columns
    unsigned lo = 0, hi = count;
    while (hi - lo > 1) {
        const unsigned mid = (lo + hi) / 2;
        if (items[mid].cta_begin <= blockIdx.x) lo = mid;
        else hi = mid;
    }
    const Item it = items[lo];
    const unsigned local = blockIdx.x - it.cta_begin;
    const unsigned bx = local % it.gx, by = (local / it.gx) % it.gy, bz = local / (it.gx * it.gy);
    switch (it.bits) {
        case 3: pipe_pair_or_tail<3, Simplify, Packed>(it, bx, by, bz, exchange); break;
        case 4: pipe_pair_or_tail<4, Simplify, Packed>(it, bx, by, bz, exchange); break;
        case 5: pipe_pair_or_tail<5, Simplify, Packed>(it, bx, by, bz, exchange); break;
        case 6: pipe_pair_or_tail<6, Simplify, Packed>(it, bx, by, bz, exchange); break;
        default: pipe_pair_or_tail<8, Simplify, Packed>(it, bx, by, bz, exchange); break;
    }
}

// Tall bands: grid x = column pairs (32 outputs), grid y = bands of W full 128-row tiles.
template <int W = 4, bool Simplify = false, bool Packed = false>
__global__ __launch_bounds__(W * 32) void grouped_pipe_band_kernel(const Item* items, unsigned count) {
    __shared__ Half16 exchange[64];  // 2 buffers x 2 decoded tiles x 16 columns
    unsigned lo = 0, hi = count;
    while (hi - lo > 1) {
        const unsigned mid = (lo + hi) / 2;
        if (items[mid].cta_begin <= blockIdx.x) lo = mid;
        else hi = mid;
    }
    const Item it = items[lo];
    const unsigned local = blockIdx.x - it.cta_begin;
    const unsigned bx = local % it.gx, by = (local / it.gx) % it.gy, bz = local / (it.gx * it.gy);
    switch (it.bits) {
        case 3: item_pipe_band<3, W, Simplify, Packed>(it, bx, by, bz, exchange); break;
        case 4: item_pipe_band<4, W, Simplify, Packed>(it, bx, by, bz, exchange); break;
        case 5: item_pipe_band<5, W, Simplify, Packed>(it, bx, by, bz, exchange); break;
        case 6: item_pipe_band<6, W, Simplify, Packed>(it, bx, by, bz, exchange); break;
        default: item_pipe_band<8, W, Simplify, Packed>(it, bx, by, bz, exchange); break;
    }
}

template <int MT, int V>
__global__ __launch_bounds__(128) void grouped_kernel(const Item* items, unsigned count) {
    unsigned lo = 0, hi = count;  // last item with cta_begin <= blockIdx.x
    while (hi - lo > 1) {
        const unsigned mid = (lo + hi) / 2;
        if (items[mid].cta_begin <= blockIdx.x) lo = mid;
        else hi = mid;
    }
    const Item it = items[lo];
    const unsigned local = blockIdx.x - it.cta_begin;
    const unsigned bx = local % it.gx, by = (local / it.gx) % it.gy, bz = local / (it.gx * it.gy);
    if constexpr (V == 2) {
        switch (it.bits) {  // as wmma_grouped_kernel
            case 3: item_core<3, MT>(it, bx, by, bz); break;
            case 4: item_core<4, MT>(it, bx, by, bz); break;
            case 5: item_core<5, MT>(it, bx, by, bz); break;
            case 6: item_core<6, MT>(it, bx, by, bz); break;
            default: item_core<8, MT>(it, bx, by, bz); break;
        }
    } else {
        constexpr int Mode = V == 3 || V == 4 ? 0 : V - 30;
        constexpr bool Tiled = V == 4;
        switch (it.bits) {
            case 3: item_core_v3<3, MT, Mode, Tiled>(it, bx, by, bz); break;
            case 4: item_core_v3<4, MT, Mode, Tiled>(it, bx, by, bz); break;
            case 5: item_core_v3<5, MT, Mode, Tiled>(it, bx, by, bz); break;
            case 6: item_core_v3<6, MT, Mode, Tiled>(it, bx, by, bz); break;
            default: item_core_v3<8, MT, Mode, Tiled>(it, bx, by, bz); break;
        }
    }
}

template <int Core8>
__global__ __launch_bounds__(128) void grouped_core8_kernel(const Item* items, unsigned count) {
    // Variant 2 uses 4 KB for B, reusing the full 8 KB allocation for its A fallback.
    __shared__ Half16 exchange[256];
    unsigned lo = 0, hi = count;
    while (hi - lo > 1) {
        const unsigned mid = (lo + hi) / 2;
        if (items[mid].cta_begin <= blockIdx.x) lo = mid;
        else hi = mid;
    }
    const Item it = items[lo];
    const unsigned local = blockIdx.x - it.cta_begin;
    const unsigned bx = local % it.gx, by = (local / it.gx) % it.gy, bz = local / (it.gx * it.gy);
    switch (it.bits) {
        case 3: item_core8<3, Core8>(it, bx, by, bz, exchange); break;
        case 4: item_core8<4, Core8>(it, bx, by, bz, exchange); break;
        case 5: item_core8<5, Core8>(it, bx, by, bz, exchange); break;
        case 6: item_core8<6, Core8>(it, bx, by, bz, exchange); break;
        default: item_core8<8, Core8>(it, bx, by, bz, exchange); break;
    }
}

// Row-tile class of an item with `rows` rows: MT in {1, 2, 4, 8} (index 0..3).
inline unsigned mt_class(unsigned rows) { return route::mt_class(rows); }
inline unsigned mt_of_class(unsigned cls) { return route::mt_of_class(cls); }

}  // namespace strix::minimax::moe::opt::pfk
