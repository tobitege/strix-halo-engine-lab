#pragma once
// GLM post-exchange kernels (design notes section 3). The first consumer of every
// sublayer reduction: R17 y = p0 + p1 (rank 0's partial first, on both ranks), then the R7 expand of the mHC residual
//   X'_i[d] = post_i * y[d] + (((C_0i X_0[d] + C_1i X_1[d]) + C_2i X_2[d]) + C_3i X_3[d])
// with the C1 non-finite flag, i.e. exactly Tier A `dev::hc_expand(X, mix, y)` after `y = p0 + p1` (gate: bitwise).
//   * post_expand: one kernel instead of two; optionally waits for the peer's message itself (RxView, glmx_wait).
//   * hc_bracket + post_expand_bracket: the bracket depends only on X and the mix of THIS sublayer (known before its
//     attention / FFN runs), so it is computed before the exchange and only post_i * y + B is left after it. The same
//     two roundings in the same order as hc_expand (product, then sum with the bracket): bitwise by construction.
// Layout as Tier A: X [rows][4][4096], mix [rows][24] (pre[4] post[4] comb[16], comb[j * 4 + i]), y/p [rows][4096].
#include "glm_xfer.hpp"

#include <hip/hip_runtime.h>

namespace strix::glm::xfer {

// `peer` == nullptr: the peer partial is view.payload and every block waits in-kernel for it (recv_view()).
void post_expand(const float* X, const float* mix, const float* local, const float* peer, const RxView& view, unsigned rank,
                 float* Xout, unsigned* flag, unsigned rows, hipStream_t s, unsigned flag_stride = 0);
// Same, for a payload (local and peer / folded slot) in the split chunk-major layout, SplitLayout{rows, 4096, 2048}.
void post_expand_split(const float* X, const float* mix, const float* local, const float* peer, const RxView& view, unsigned rank,
                       float* Xout, unsigned* flag, unsigned rows, hipStream_t s, unsigned flag_stride = 0);
void hc_bracket(const float* X, const float* mix, float* B, unsigned rows, hipStream_t s);
void post_expand_bracket(const float* B, const float* mix, const float* local, const float* peer, const RxView& view,
                         unsigned rank, float* Xout, unsigned* flag, unsigned rows, hipStream_t s, unsigned flag_stride = 0);

// Tier A dev::hc_pre with the R6 mix (sigmoids, row softmax, 20 sinkhorn rounds) on 24 lanes of one wave instead of
// lane 0: lane t < 16 owns comb element t, lanes 16..23 own pre / post; row and column sums are gathered with
// shuffles and computed by every lane of the row / column in the declared order, so each value is the same sequence
// of IEEE operations as on lane 0 (bitwise, gated against dev::hc_pre in the selftest). Same arguments and layout.
void hc_pre_v2(const float* X, const float* fn, const float* base, const float* scale, float* F, float* mix, float* xc,
               unsigned rows, hipStream_t s);

}  // namespace strix::glm::xfer
