// Shared small-row GEMV sweep, direct schedule (see gemv_rows.hpp). Textual body: include it inside a
// kernel. A textual include, like the integer main loops, because the same body written
// against callable policies compiled to different register allocation on gfx1151 (spills in
// the sweep, a longer epilogue), and decode sweeps are short enough for that to show.
// In scope at the include:
//   constants  Rows, T, Streams (1 = one matrix, 2 = two matrices sharing every activation read)
//   values     const float* x; unsigned inputs, outputs, first (first of this wave's T outputs), lane (inputs % 16 == 0)
//   macros     GEMV_ROWS_WEIGHT_ROW(stream, output)      a weight row handle (uint8 pointer)
//              GEMV_ROWS_CHUNK                           the type that decodes 16 weights, with at(e)
//              GEMV_ROWS_LOAD_CHUNK(row, start)          a GEMV_ROWS_CHUNK of that row at column `start`
//              GEMV_ROWS_LANE_SUM(partial)               the family's lane sum
//              GEMV_ROWS_STORE(row, output, sum0, sum1)  the family's epilogue (every lane; it stores once
//                                                        and only when output < outputs)
// Every (stream, output, row, lane) accumulator starts at +0 and visits its chunks and their
// elements in ascending order as acc = acc + weight * activation.
// Each wave loads its own activations and shares them between its T outputs: the activations
// are read once per T outputs.
{
    if (first < outputs) {
        const auto chunks = inputs / 16U;
        const std::uint8_t* wp[T];
        const std::uint8_t* uptr[T];
#pragma unroll
        for (unsigned t = 0; t < T; ++t) {
            // A tail wave reuses a valid row for its unused loads; it never stores it.
            const auto o = min(first + t, outputs - 1U);
            wp[t] = GEMV_ROWS_WEIGHT_ROW(0U, o);
            uptr[t] = Streams == 2U ? GEMV_ROWS_WEIGHT_ROW(1U, o) : wp[t];
        }
        float acc[T][Rows] = {}, uacc[T][Rows] = {};
        for (unsigned c = lane; c < chunks; c += 32U) {
            const auto start = c * 16U;
            GEMV_ROWS_CHUNK a[T], b[T];
#pragma unroll
            for (unsigned t = 0; t < T; ++t) {
                a[t] = GEMV_ROWS_LOAD_CHUNK(wp[t], start);
                if constexpr (Streams == 2U)
                    b[t] = GEMV_ROWS_LOAD_CHUNK(uptr[t], start);
            }
#pragma unroll
            for (unsigned g = 0; g < 4; ++g) {
                float4 input[Rows];
#pragma unroll
                for (unsigned r = 0; r < Rows; ++r)
                    input[r] = *reinterpret_cast<const float4*>(x + std::size_t(r) * inputs + start + 4U * g);
#pragma unroll
                for (unsigned k = 0; k < 4; ++k) {
                    const unsigned e = 4U * g + k;
#pragma unroll
                    for (unsigned t = 0; t < T; ++t) {
                        const float v = a[t].at(e);
                        float u = 0.0F;
                        if constexpr (Streams == 2U)
                            u = b[t].at(e);
#pragma unroll
                        for (unsigned r = 0; r < Rows; ++r) {
                            const float in = k == 0   ? input[r].x
                                             : k == 1 ? input[r].y
                                             : k == 2 ? input[r].z
                                                      : input[r].w;
                            acc[t][r] = acc[t][r] + v * in;
                            if constexpr (Streams == 2U)
                                uacc[t][r] = uacc[t][r] + u * in;
                        }
                    }
                }
            }
        }
#pragma unroll
        for (unsigned t = 0; t < T; ++t) {
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r) {
                const float sa = GEMV_ROWS_LANE_SUM(acc[t][r]);
                float sb = 0.0F;
                if constexpr (Streams == 2U)
                    sb = GEMV_ROWS_LANE_SUM(uacc[t][r]);
                GEMV_ROWS_STORE(r, first + t, sa, sb);
            }
        }
    }
}
