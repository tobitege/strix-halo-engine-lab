// Chunk-staged IU8 plan. Recovered from the project's split-v2 implementation;
// routing, GGUF decoding and epilogues remain operand/adapter policies.
static_assert(AD && !Fused && Probe == 0 && BM == 16 && BN == 64);
static_assert(!Operand::fast_i4 && !Contract::relaxed);
// One tile per CTA: the body returns instead of looping back to next(), so the
// staged row tables stay LDS pointers and nothing is carried around the tile loop.
static_assert(Adapter::kSingleTile);
using Plan = typename Operand::Plan;
using Ops = typename Operand::Ops;
using V4 = typename Operand::Int4;
using V8 = typename Operand::Int8;
using Raw = typename Operand::Raw;
static_assert(Plan::class_waves == 8 && Plan::chunk_groups == 32 && Plan::half_exchange);
constexpr bool Q2 = Operand::affine, Direct = Operand::direct_weights || !Plan::weight_lds;
constexpr unsigned CodeBytes = 16384, DeltaBytes = 2048;
constexpr unsigned StageBytes = Operand::activation_bytes + Operand::sum_bytes;
constexpr unsigned RowWords = Operand::chunk_row_bytes / 4;
constexpr unsigned RunWords = 32 * RowWords, Loads = (RunWords + 511) / 512;
// The IQ2_XXS grid is staged for every codec that decodes through it, also without the LDS weight tile.
__shared__ unsigned long long grid[!Q2 && !Operand::direct_weights ? 256 : 1];
__shared__ unsigned sources[16], slots[Adapter::kFinish ? 16 : 1];
__shared__ float values[Adapter::kFinish ? BM * BN : 1];
if constexpr (!Q2 && !Operand::direct_weights)
    for (unsigned i = unsigned(thread); i < 256; i += G::threads) grid[i] = ad.grid()[i];
if (thread < rows) {
    sources[thread] = ad.source(at, thread);
    if constexpr (Adapter::kFinish) slots[thread] = ad.slot(at, thread);
}
// Staged row tables (LDS): source row of tile row r, and its destination slot.
const unsigned* const dst_rows = Adapter::kFinish ? slots : sources;
const unsigned pair = unsigned(wave) & 1, q = unsigned(wave) >> 1, lo = unsigned(lane) & 15, h = unsigned(lane) >> 4;
const unsigned groups = unsigned(inputs) / 32, chunks = (groups + 31) / 32;
const auto sync = [] {
    if constexpr (Plan::lds_only_barriers) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    } else __syncthreads();
};
const auto other = [](int v) {
    int old;
    asm volatile("" : "=v"(old));
    return int(__builtin_amdgcn_permlanex16(unsigned(old), unsigned(v), 0x76543210U, 0xfedcba98U, false, false));
};
const auto other4 = [&](V4 v) { return V4{other(v[0]), other(v[1]), other(v[2]), other(v[3])}; };
struct TileRegs { unsigned w[2][Loads]; };
const auto load_tile = [&](unsigned chunk) {
    TileRegs r{};
    const unsigned first = chunk * 32;
    if (first >= groups) return r;
    const unsigned left = groups - first;
    const unsigned used = ((left + 7) / 8 * Operand::block_bytes + 3) / 4;
    const auto* b0 = ad.weight_row(at, first_col) + std::uint64_t(chunk) * Operand::chunk_row_bytes;
    const auto* b1 = ad.weight_row(at, first_col + 32) + std::uint64_t(chunk) * Operand::chunk_row_bytes;
    const unsigned rb = ad.row_bytes();
    // Keep two uniform bases and 32-bit offsets, as in the replaced launch.
    if (!Plan::partial_blocks || left >= 32) {
        #pragma unroll
        for (unsigned i = 0; i < Loads; ++i) {
            const unsigned word = unsigned(thread) + 512 * i;
            if (word < RunWords) {
                const unsigned off = integer_weight_offset(word, RowWords, rb);
                r.w[0][i] = *reinterpret_cast<const unsigned*>(b0 + off);
                r.w[1][i] = *reinterpret_cast<const unsigned*>(b1 + off);
            }
        }
    } else {
        #pragma unroll
        for (unsigned i = 0; i < Loads; ++i) {
            const unsigned word = unsigned(thread) + 512 * i;
            if (word < RunWords && word % RowWords < used) {
                const unsigned off = integer_weight_offset(word, RowWords, rb);
                r.w[0][i] = *reinterpret_cast<const unsigned*>(b0 + off);
                r.w[1][i] = *reinterpret_cast<const unsigned*>(b1 + off);
            }
        }
    }
    return r;
};
auto* tile = lds + StageBytes;
const auto* own_row = ad.weight_row(at, first_col + int(pair * 32) + lane);
Raw ahead{};
TileRegs tile_next{};
// A wave whose first group is past a narrow row (inputs < 256) has no group at all: it must not read past the row.
if constexpr (Direct) { if (q < groups) ahead = Operand::load_raw(own_row, q); }
else tile_next = load_tile(0);
V8 magic{Ops::dot_magic, Ops::dot_magic, Ops::dot_magic, Ops::dot_magic,
         Ops::dot_magic, Ops::dot_magic, Ops::dot_magic, Ops::dot_magic};
if constexpr (Plan::opaque_magic) asm volatile("" : "+v"(magic));
float leaf[4][2][8] = {};
#pragma unroll 1
for (unsigned chunk = 0; chunk < chunks; ++chunk) {
    sync();
    // Both global activation reads issue before either LDS store.
    // Stage registers. Plan::split_stage_registers: each one is zeroed on its own, so both items' loads are issued
    // before any result is consumed. Otherwise one initialiser per array: the two deltas then share a 64-bit register
    // pair and item 0's loads are awaited before item 1 is addressed (one more load-counter wait per chunk), but the
    // gate/up instantiations get the denser instruction schedule with it. Same values either way.
    V4 av[2];
    float dv[2];
    if constexpr (!Plan::split_stage_registers) {
        __builtin_memset(av, 0, sizeof(av));
        __builtin_memset(dv, 0, sizeof(dv));
    }
    #pragma unroll
    for (unsigned i = 0; i < 2; ++i) {
        const auto item = integer_stage_item(unsigned(thread), i);
        const unsigned r = item.row, kh = item.half, gg = item.group;
        const unsigned g = chunk * 32 + gg;
        if constexpr (Plan::split_stage_registers) {
            av[i] = V4{0, 0, 0, 0};
            dv[i] = 0.0f;
        }
        if (g < groups && r < unsigned(rows)) {
            const unsigned src = sources[r];
            av[i] = *reinterpret_cast<const V4*>(ad.code_row_of(src) + g * 32 + kh * 16);
            if (!kh) dv[i] = ad.delta_of(src, g);
        }
    }
    if constexpr (!Direct) {
        #pragma unroll
        for (unsigned i = 0; i < Loads; ++i) {
            const unsigned word = unsigned(thread) + 512 * i;
            if (word < RunWords) {
                reinterpret_cast<unsigned*>(tile)[word] = tile_next.w[0][i];
                reinterpret_cast<unsigned*>(tile)[RunWords + word] = tile_next.w[1][i];
            }
        }
        if (chunk + 1 < chunks) tile_next = load_tile(chunk + 1);
    }
    #pragma unroll
    for (unsigned i = 0; i < 2; ++i) {
        const auto item = integer_stage_item(unsigned(thread), i);
        const unsigned r = item.row, kh = item.half, gg = item.group;
        if (chunk * 32 + gg >= groups) continue;
        reinterpret_cast<V4*>(lds)[(2 * gg + kh) * 16 + r] = av[i];
        const unsigned at_row = integer_row_position(r);
        if (!kh) reinterpret_cast<float*>(lds + CodeBytes)[gg * 16 + at_row] = dv[i];
        if constexpr (Q2) {
            int sum = 0;
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j)
                sum = __builtin_amdgcn_sudot4(false, 0x01010101, true, av[i][j], sum, false);
            if constexpr (Plan::float_half_sums)
                reinterpret_cast<float*>(lds + CodeBytes + DeltaBytes)[kh * 512 + gg * 16 + at_row] = float(sum);
            else reinterpret_cast<short*>(lds + CodeBytes + DeltaBytes)[2 * (gg * 16 + at_row) + kh] = short(sum);
        }
    }
    sync();
    #pragma unroll
    for (unsigned k = 0; k < 4; ++k) {
        const unsigned gg = q + 8 * k;
        if (chunk * 32 + gg >= groups) break;
        Raw raw;
        if constexpr (Direct) {
            raw = ahead;
            const unsigned next = k < 3 ? chunk * 32 + gg + 8 : (chunk + 1) * 32 + q;
            if (next < groups) ahead = Operand::load_raw(own_row, next);
        } else raw = Operand::load_raw(tile + (pair * 32 + unsigned(lane)) * Operand::chunk_row_bytes, gg);
        const auto own = [&] {
            if constexpr (Operand::direct_weights) return Ops::decode(raw, grid);
            else return Ops::decode_lean(raw, grid);
        }();
        const V4 x0 = other4(own.b0), x1 = other4(own.b1);
        const unsigned xdm = Q2 ? unsigned(other(int(own.dm_bits))) : 0;
        const unsigned xsc = Q2 ? unsigned(other(int(own.sc_bits))) : 0;
        const float xs = Q2 ? 0.0f : __int_as_float(other(__float_as_int(own.scale)));
        const V4 a0 = reinterpret_cast<const V4*>(lds)[2 * gg * 16 + lo];
        const V4 a1 = reinterpret_cast<const V4*>(lds)[(2 * gg + 1) * 16 + lo];
        const auto* dt = reinterpret_cast<const float*>(lds + CodeBytes) + gg * 16 + 8 * h;
        #pragma unroll
        for (unsigned t = 0; t < 2; ++t) {
            const bool mine = (t == 1) == (h != 0);
            const V4 b0 = mine ? own.b0 : x0, b1 = mine ? own.b1 : x1;
            V8 dot = integer_wmma<8>(a0, b0, magic);
            dot = integer_wmma<8>(a1, b1, dot);
            if constexpr (Q2) {
                const unsigned dm = mine ? own.dm_bits : xdm, sc = mine ? own.sc_bits : xsc;
                const float d = Ops::half_bits(static_cast<unsigned short>(dm));
                const float dmin = Ops::half_bits(static_cast<unsigned short>(dm >> 16));
                const float m0 = float((sc & 255) >> 4), m1 = float((sc >> 8) >> 4);
                const auto* s0 = reinterpret_cast<const float*>(lds + CodeBytes + DeltaBytes) + gg * 16 + 8 * h;
                #pragma unroll
                for (unsigned e = 0; e < 8; ++e) {
                    const float minimum = [&] {
                        if constexpr (Plan::float_half_sums)
                            return __builtin_fmaf(m1, s0[512 + e], __builtin_fmaf(m0, s0[e], 0.0f));
                        else {
                            const int packed = reinterpret_cast<const int*>(lds + CodeBytes + DeltaBytes)[gg * 16 + 8 * h + e];
                            return float(__mul24(int(m0), int(short(packed))) + __mul24(int(m1), packed >> 16));
                        }
                    }();
                    const float scaled = d * Ops::magic_float(dot[e]), offset = dmin * minimum;
                    leaf[k][t][e] = leaf[k][t][e] + (scaled - offset) * dt[e];
                }
            } else {
                const float scale = mine ? own.scale : xs;
                #pragma unroll
                for (unsigned e = 0; e < 8; ++e)
                    leaf[k][t][e] = leaf[k][t][e] + Ops::magic_float(dot[e]) * (scale * dt[e]);
            }
        }
    }
}
float v[2][8];
#pragma unroll
for (unsigned t = 0; t < 2; ++t)
    #pragma unroll
    for (unsigned e = 0; e < 8; ++e)
        v[t][e] = (leaf[0][t][e] + leaf[2][t][e]) + (leaf[1][t][e] + leaf[3][t][e]);
auto* exchange = reinterpret_cast<float*>(lds);
#pragma unroll
for (unsigned s = 4; s; s /= 2) {
    sync();
    if (q >= s && q < 2 * s) {
        auto* slot = exchange + (4 * pair + q - s) * 512;
        #pragma unroll
        for (unsigned t = 0; t < 2; ++t)
            #pragma unroll
            for (unsigned e = 0; e < 8; ++e) slot[(8 * t + e) * 32 + unsigned(lane)] = v[t][e];
    }
    sync();
    if (q < s) {
        const auto* slot = exchange + (4 * pair + q) * 512;
        #pragma unroll
        for (unsigned t = 0; t < 2; ++t)
            #pragma unroll
            for (unsigned e = 0; e < 8; ++e) v[t][e] = v[t][e] + slot[(8 * t + e) * 32 + unsigned(lane)];
    }
}
if (q == 0) {
    #pragma unroll
    for (unsigned t = 0; t < 2; ++t)
        #pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            const unsigned r = 2 * e + h, c = pair * 32 + t * 16 + lo;
            if (r < unsigned(rows)) {
                ad.store_raw_at(at, dst_rows[r], first_col + int(c), v[t][e]);
                if constexpr (Adapter::kFinish) values[r * BN + c] = v[t][e];
            }
        }
}
if constexpr (Adapter::kFinish) {
    __syncthreads();
    ad.finish_rows(at, values, slots, BM, BN);
}
return;
