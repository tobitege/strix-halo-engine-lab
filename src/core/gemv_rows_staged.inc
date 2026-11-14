// Shared small-row GEMV sweep, staged schedule (see gemv_rows.hpp). Textual body: include it inside a
// kernel. A textual include, like the integer main loops, because the same body written
// against callable policies compiled to different register allocation on gfx1151 (spills in
// the sweep, a longer epilogue), and decode sweeps are short enough for that to show.
// In scope at the include:
//   constants  Rows, T, Waves (block = Waves * 32 threads), Buffers (1 or 2), Streams (1 = one matrix, 2 = two matrices sharing every activation read)
//   values     const float* x; unsigned inputs, outputs, first (first of this wave's T outputs), lane;
//              __shared__ float4 staged[Buffers][Rows * 4U * 32U] (x 16-byte aligned, inputs % 16 == 0)
//   macros     GEMV_ROWS_WEIGHT_ROW(stream, output)      a weight row handle (uint8 pointer)
//              GEMV_ROWS_CHUNK                           the type that decodes 16 weights, with at(e)
//              GEMV_ROWS_LOAD_CHUNK(row, start)          a GEMV_ROWS_CHUNK of that row at column `start`
//              GEMV_ROWS_LANE_SUM(partial)               the family's lane sum
//              GEMV_ROWS_STORE(row, output, sum0, sum1)  the family's epilogue (every lane; it stores once
//                                                        and only when output < outputs)
// Every (stream, output, row, lane) accumulator starts at +0 and visits its chunks and their
// elements in ascending order as acc = acc + weight * activation.
// The block stages the next 32 chunks of every row in LDS and its Waves * T outputs share that
// copy: the activations are read once per Waves * T outputs. Two buffers overlap the next stage
// with the products; one buffer halves the LDS at two barriers per step.
{
    static_assert((T == 1U || T == 2U) && (Buffers == 1U || Buffers == 2U) && (Streams == 1U || Streams == 2U));
    const bool valid = first < outputs;
    const auto chunks = inputs / 16U, steps = (chunks + 31U) / 32U;
    const auto stage = [&](unsigned step, unsigned buffer) {
        for (unsigned i = threadIdx.x; i < Rows * 128U; i += Waves * 32U) {
            const unsigned r = i / 128U, group = i % 128U, c = step * 32U + group / 4U;
            if (c < chunks)
                staged[buffer][(r * 4U + group % 4U) * 32U + group / 4U] =
                    *reinterpret_cast<const float4*>(x + std::size_t(r) * inputs + std::size_t(c) * 16U +
                                                     4U * (group % 4U));
        }
    };
    const std::uint8_t* wp[T];
    const std::uint8_t* uptr[T];
    if (valid) {
#pragma unroll
        for (unsigned t = 0; t < T; ++t) {
            // Reuse a valid weight row for unused tail loads; never store it.
            const auto o = min(first + t, outputs - 1U);
            wp[t] = GEMV_ROWS_WEIGHT_ROW(0U, o);
            uptr[t] = Streams == 2U ? GEMV_ROWS_WEIGHT_ROW(1U, o) : wp[t];
        }
    }
    float acc[T][Rows] = {}, uacc[T][Rows] = {};
    stage(0U, 0U);
    __syncthreads();
    for (unsigned step = 0; step < steps; ++step) {
        if (Buffers == 2U && step + 1U < steps)
            stage(step + 1U, (step + 1U) & 1U);
        const unsigned c = step * 32U + lane;
        if (valid && c < chunks) {
            const auto start = c * 16U;
            GEMV_ROWS_CHUNK a[T], b[T];
#pragma unroll
            for (unsigned t = 0; t < T; ++t) {
                a[t] = GEMV_ROWS_LOAD_CHUNK(wp[t], start);
                if constexpr (Streams == 2U)
                    b[t] = GEMV_ROWS_LOAD_CHUNK(uptr[t], start);
            }
            const float4* in = staged[Buffers == 2U ? step & 1U : 0U];
#pragma unroll
            for (unsigned g = 0; g < 4; ++g) {
                float4 input[Rows];
#pragma unroll
                for (unsigned r = 0; r < Rows; ++r)
                    input[r] = in[(r * 4U + g) * 32U + lane];
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
                            const float value = k == 0   ? input[r].x
                                                : k == 1 ? input[r].y
                                                : k == 2 ? input[r].z
                                                         : input[r].w;
                            acc[t][r] = acc[t][r] + v * value;
                            if constexpr (Streams == 2U)
                                uacc[t][r] = uacc[t][r] + u * value;
                        }
                    }
                }
            }
        }
        if (step + 1U < steps) {
            __syncthreads();
            if (Buffers == 1U) {
                // One buffer: the next window is staged after the products, with a second barrier.
                stage(step + 1U, 0U);
                __syncthreads();
            }
        }
    }
    if (valid) {
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
