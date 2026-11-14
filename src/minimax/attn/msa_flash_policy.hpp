// MiniMax MSA prefill on the shared FAST key-split flash core (src/attn_core/sparse_flash_keysplit.hpp).
// Included by msa_fused.hip inside its unnamed namespace, after kernel_view / load4 / norm_rope_lane.
//
// Work-group = (row, KV group): the group's 16 query heads against the row's selected 128-key blocks (dense rows:
// every visible block), K / V read as stored in the BF16 caches. Same grid, selection and block lists as the exact
// in-block kernel (mode 10); only the arithmetic differs (FAST: BF16 queries and probabilities, WMMA sums, one
// softmax over the whole key list instead of per-block partials + combine).
struct MsaFlashArgs {
    PrefillView p;
    std::uint32_t dense;
};
struct MsaFlashPolicy {
    using Args = MsaFlashArgs;
    struct Row {
        const std::uint16_t* kc;
        const std::uint16_t* vc;
        const std::uint32_t* ids;
        const float* q;
        const float* q_w;
        const RopeRow* rope;
        float* o;
        std::uint64_t count;
        std::uint32_t cnt, kv_heads, g, all;
    };
    static constexpr unsigned kHeads = kGroup, kDim = kHeadDim, kSegKeys = kBlock;
    static constexpr strix::attn_core::tile::Operand kOp = strix::attn_core::tile::Operand::bf16;
    static constexpr float kScaleLog2 = kScale * 1.44269504088896341F;  // logits = dot x 128^-0.5 (natural log units)
    static __device__ __forceinline__ bool begin(const Args& a, unsigned wg, Row& x) {
        const PrefillView& in = a.p;
        const unsigned kvh = in.window.kv_heads;
        const unsigned row = in.first_window * kMaxRows + wg / kvh;
        const unsigned window = row / kMaxRows - in.first_window;
        const DecodeView v = kernel_view(in, window);
        const unsigned rg = (row % kMaxRows) * kvh + wg % kvh;
        if (rg >= v.rows_per_seq * kvh) return false;
        const unsigned r = rg / kvh, g = rg % kvh, s = r / v.rows_per_seq, q_heads = kvh * kGroup;
        x.count = v.pos[r] + 1;
        const std::uint32_t nb = static_cast<std::uint32_t>(block_count(x.count));
        x.all = (a.dense != 0U || nb <= kTopK) ? 1U : 0U;
        x.cnt = x.all != 0U ? nb : v.idx_count[rg];
        if (x.cnt == 0) return false;
        x.ids = v.idx + static_cast<std::size_t>(rg) * kTopK;
        x.kc = v.k_cache[s];
        x.vc = v.v_cache[s];
        x.q = v.q + (static_cast<std::size_t>(r) * q_heads + g * kGroup) * kHeadDim;
        x.o = v.o + (static_cast<std::size_t>(r) * q_heads + g * kGroup) * kHeadDim;
        x.q_w = v.q_w;
        x.rope = &v.rope[r];
        x.kv_heads = kvh;
        x.g = g;
        return true;
    }
    static __device__ __forceinline__ unsigned segments(const Args&, const Row& x) { return x.cnt; }
    static __device__ __forceinline__ std::uint64_t first(const Row& x, unsigned c) {
        return static_cast<std::uint64_t>(x.all != 0U ? c : x.ids[c]) * kBlock;
    }
    static __device__ __forceinline__ unsigned seg_keys(const Args&, const Row& x, unsigned c) {
        const std::uint64_t lo = first(x, c);
        return static_cast<unsigned>((x.count < lo + kBlock ? x.count : lo + kBlock) - lo);
    }
    static __device__ __forceinline__ const std::uint16_t* key_row(const Args&, const Row& x, unsigned c, unsigned k) {
        return x.kc + ((first(x, c) + k) * x.kv_heads + x.g) * kHeadDim;
    }
    static __device__ __forceinline__ const std::uint16_t* value_row(const Args&, const Row& x, unsigned c, unsigned k) {
        return x.vc + ((first(x, c) + k) * x.kv_heads + x.g) * kHeadDim;
    }
    static __device__ __forceinline__ void query(const Args&, const Row& x, unsigned head, unsigned lane, float (&q)[4]) {
        load4(q, x.q + static_cast<std::size_t>(head) * kHeadDim, lane);
        norm_rope_lane(q, x.q_w, *x.rope, lane);
    }
    static __device__ __forceinline__ float* out(const Args&, const Row& x) { return x.o; }
};
