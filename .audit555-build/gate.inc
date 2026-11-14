struct Args {
    std::string model, prompt_path, transport, loads_out, hashes_out, seed_out;
    std::uint32_t rank = 0, world = 1, layers = 40, rows = 0, chunk = 1024, window = 8, lanes = 4, moe_chunk = 4096, repeat = 1, warmup = 0,
                  split_min_rows = strix::deepseek41::PrefillOptions{}.split_min_rows,
                  joint_max_rows = strix::deepseek41::PrefillOptions{}.joint_max_rows;
    unsigned connect_timeout = 60;
    bool embed_identity = false;  // --embed-identity A:B
    std::uint32_t embed_a = 0, embed_b = 0;
    bool select_ab = false;  // --select-ab: run k with the exact dense selector off (k even) / on (k odd), one load
    std::uint32_t draft_rows = 0; // --draft-rows N, optional --seed-out PREFIX for raw seed gates
    std::uint32_t extend_check = 0;  // --extend-check S: after the runs, run() + extend() in S-row pieces == the whole run
    std::string select_pattern;  // --select-pattern OSSO...: run k off ('O') / on ('S') by the k-th letter (cyclic)
    bool timing = false, reference = false, cone = true, loads = false, tp = true, stream_experts = false, row_split = true, decode_reference = false,
         state_timing = false;
    bool ttft = false;  // --ttft-timing
    std::string task_levels;  // --task-levels
    std::vector<std::uint32_t> task_rows;  // --task-rows: prompt prefixes gated in turn (default: the whole prompt)
    std::uint32_t task_units = 1, interleave = 0, bg_rows = 16;
    std::uint32_t task_moe_chunk = strix::deepseek41::PrefillTaskOptions{}.moe_chunk_rows;  // --task-moe-chunk (0: the prefill's)
    std::uint32_t task_engram_rows = strix::deepseek41::PrefillTaskOptions{}.engram_rows;  // --task-engram-rows
    double task_budget = 0.0;
    double mem_guard_gib = 16.0;  // --mem-guard-gib (0: off)
    bool task_table = false;
    std::uint32_t ttft_warmup = 0, ttft_decode = 0;
};

void validate_embed_identity_args(const Args& a) {
    if (a.embed_identity && (a.ttft || a.world != 1U))
        throw std::invalid_argument("--embed-identity requires --world 1 without --ttft-timing");
}


strix::deepseek41::PrefillOptions options(const Args& a, std::size_t rows) {
    strix::deepseek41::PrefillOptions o;
    o.layers = a.layers;
    o.draft_rows = a.draft_rows;
    o.max_positions = static_cast<std::uint32_t>(rows);
    o.chunk_rows = a.chunk;
    o.window_rows = a.window;
    o.attention_lanes = a.lanes;
    o.moe_chunk_rows = a.moe_chunk;
    o.tp_attention = a.tp;
    o.row_split = a.row_split;
    o.stream_experts = a.stream_experts;
    o.stream_hashes = a.embed_identity || a.reference || !a.hashes_out.empty();
    o.verify_embed_extras = a.embed_identity;
    o.timing = a.timing;
    o.cone = a.cone;
    o.routing_loads = a.loads;
    o.warmup_rows = a.warmup;
    o.split_min_rows = a.split_min_rows;
    o.joint_max_rows = a.joint_max_rows;
    return o;
}


int embed_identity_gate(const Args& a, strix::deepseek41::DevicePrefill& pre, const std::vector<std::uint32_t>& prompt,
                        const strix::deepseek41::PrefillResult& ref) {
    namespace ss = strix::session;
    const std::uint32_t A = a.embed_a, B = a.embed_b;
    if (A >= B || B > prompt.size()) throw std::invalid_argument("--embed-identity: need A < B <= rows");
    if (a.layers == 0U || ref.stream_hashes.size() != a.layers)
        throw std::runtime_error("--embed-identity: missing reference stream hashes");
    const auto fnv = [](const auto& v) -> std::uint64_t {  // logits, or the labelled state hashes
        std::uint64_t h = 14695981039346656037ULL;
        const auto mix = [&h](std::uint64_t b, int bytes) {
            for (int k = 0; k < bytes; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
        };
        for (const auto& e : v) {
            if constexpr (requires { e.second; }) {
                mix(e.second, 8);
            } else {
                std::uint32_t b;
                std::memcpy(&b, &e, 4);
                mix(b, 4);
            }
        }
        return h;
    };
    const auto ref_state = pre.state_hashes();
    const std::span<const std::uint32_t> span_tokens(prompt.data() + A, B - A);
    const std::vector<std::uint16_t> rows = pre.embedding_rows(span_tokens);
    const std::uint32_t width = static_cast<std::uint32_t>(rows.size() / (B - A));
    struct Out {
        strix::deepseek41::PrefillResult r;
        std::vector<std::pair<std::string, std::uint64_t>> state;
    };
    const auto run = [&](const char* name, const std::vector<std::uint16_t>* data, ss::EmbedRowFormat format, bool extras) {
        if (data != nullptr) {
            ss::EmbedSplice sp;
            sp.width = width;
            sp.format = format;
            ss::EmbedSegment seg;
            seg.run_row = A;
            seg.rows = B - A;
            seg.data = data->data();
            seg.extras = extras;
            sp.segments.push_back(seg);
            pre.set_embed_rows(std::move(sp));
        }
        Out o;
        o.r = pre.run(prompt);
        if (o.r.stream_hashes.size() != a.layers)
            throw std::runtime_error("--embed-identity: missing arm stream hashes");
        o.state = pre.state_hashes();
        std::printf("embed-identity %-8s next %u logits %016llx state %016llx\n", name, o.r.next_token,
                    static_cast<unsigned long long>(fnv(o.r.logits)), static_cast<unsigned long long>(fnv(o.state)));
        return o;
    };
    const auto same = [](const Out& x, const strix::deepseek41::PrefillResult& r, const std::vector<std::pair<std::string, std::uint64_t>>& st) {
        return x.r.next_token == r.next_token && x.r.logits.size() == r.logits.size() &&
               std::memcmp(x.r.logits.data(), r.logits.data(), r.logits.size() * sizeof(float)) == 0 &&
               x.r.stream_hashes == r.stream_hashes && x.state == st;
    };
    int bad = 0;
    const auto verdict = [&](const char* name, bool ok) {
        std::printf("embed-identity %-8s %s\n", name, ok ? "PASS" : "FAIL");
        if (!ok) ++bad;
    };
    const Out identity = run("identity", &rows, ss::EmbedRowFormat::f16, false);
    verdict("identity", same(identity, ref, ref_state));

    std::vector<std::uint16_t> shifted(rows.size());  // row i <- the row of token i + 1 (cyclic)
    for (std::uint32_t i = 0; i < B - A; ++i)
        std::memcpy(shifted.data() + std::size_t{i} * width, rows.data() + std::size_t{(i + 1U) % (B - A)} * width, std::size_t{width} * 2U);
    const bool movable = shifted != rows;  // an unchanged control cannot establish substitution
    const Out control = run("control", &shifted, ss::EmbedRowFormat::f16, false);
    verdict("control", movable && control.r.logits.size() == ref.logits.size() &&
                       std::memcmp(control.r.logits.data(), ref.logits.data(), ref.logits.size() * sizeof(float)) != 0);
    if (!movable) std::printf("embed-identity control inconclusive: choose an interval with distinct embedding rows\n");

    std::vector<std::uint16_t> cut(rows.size()), cut_bf16(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        cut[i] = static_cast<std::uint16_t>(rows[i] & 0xFFF8U);  // 1 + 7 significant bits: exact in both formats
        const std::uint32_t sign = (cut[i] & 0x8000U) << 16U, e = (cut[i] >> 10U) & 31U;
        std::uint32_t mant = cut[i] & 0x3FFU, bits = 0;
        if (e == 31U) throw std::runtime_error("--embed-identity: the embedding table holds a non-finite value");
        if (e != 0U) {
            bits = ((e + 112U) << 23U) | (mant << 13U);
        } else if (mant != 0U) {
            std::uint32_t ex = 113U;
            while ((mant & 0x400U) == 0U) {
                mant <<= 1U;
                --ex;
            }
            bits = (ex << 23U) | ((mant & 0x3FFU) << 13U);
        }
        cut_bf16[i] = static_cast<std::uint16_t>((sign | bits) >> 16U);
    }
    const Out as_f16 = run("cut-f16", &cut, ss::EmbedRowFormat::f16, false);
    const Out as_bf16 = run("cut-bf16", &cut_bf16, ss::EmbedRowFormat::bf16, false);
    verdict("bf16", same(as_bf16, as_f16.r, as_f16.state));

    const Out image = run("image", &rows, ss::EmbedRowFormat::f16, true);
    bool finite = !image.r.logits.empty();
    for (const float v : image.r.logits) finite = finite && std::isfinite(v);
    verdict("image-router", image.r.embed_image_router_checked);
    verdict("image-dead", image.r.embed_dead_hash_checked);
    verdict("image", finite && !same(image, ref, ref_state));
    std::printf("embed-identity image    logits %s the plain run\n",
                fnv(image.r.logits) == fnv(ref.logits) ? "equal to" : "differ from");

    // Keep the same allocation: long image -> short text -> long text must restore the original result.
    if (prompt.size() > 1U) (void)pre.run(std::span<const std::uint32_t>(prompt.data(), prompt.size() / 2U));
    const Out text = run("text", nullptr, ss::EmbedRowFormat::bf16, false);
    verdict("text", same(text, ref, ref_state));
    (void)run("image-again", &rows, ss::EmbedRowFormat::f16, true);
    if (prompt.size() > 1U) {
        ss::EmbedSplice short_splice;
        short_splice.width = width;
        short_splice.format = ss::EmbedRowFormat::f16;
        short_splice.segments.push_back({0, 1, rows.data(), false});
        pre.set_embed_rows(std::move(short_splice));
        (void)pre.run(std::span<const std::uint32_t>(prompt.data(), prompt.size() / 2U));
    }
    verdict("text-after-short-splice", same(run("text", nullptr, ss::EmbedRowFormat::bf16, false), ref, ref_state));
    std::printf("embed-identity rows [%u, %u) of %zu, width %u, staging %.2f MiB: %s\n", A, B, prompt.size(), width,
                static_cast<double>(strix::deepseek41::DevicePrefill::embed_staging_bytes(B - A, width)) / 1048576.0,
                bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}

