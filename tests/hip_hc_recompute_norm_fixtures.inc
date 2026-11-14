// Included in hip_hc_wide_rows_tests.hip's fixture namespace.
// A transaction is the existing producer, weight pack, prefetched projection and
// output. No normalization surrogate and no separate inverse pass are timed.
struct RecomputeFixture {
    Fixture f;
    // STRIX_HC_RECOMPUTE_STREAM_STRIDE (bytes, >= 20480, multiple of 8): the stream row pitch of the timed fixture.
    static inline const std::uint64_t ss = [] {
        const char* v = std::getenv("STRIX_HC_RECOMPUTE_STREAM_STRIDE");
        const std::uint64_t n = v ? std::strtoull(v, nullptr, 0) : 0U;
        return n >= kWidth * 2U && n % 8U == 0U ? n : std::uint64_t{(kWidth + 64U) * 2U};
    }();
    static constexpr std::uint64_t hs = (kHidden + 32U) * 2U;
    Buffer hyper, streams, hidden, routed, combined, gates, norm, inverse;
    std::vector<std::uint16_t> hyper_host, hidden_host, routed_host;
    strix::HipHyperMixLauncherConfig c = config(true);
    unsigned kind; // 0 norm only, 1 bit-1 inject+norm, 2 bit-2 combine+inject+norm
    bool in_place;
    RecomputeFixture(unsigned rows, unsigned site, unsigned flavor, bool alias, float epsilon, bool output_bf16 = true, bool guard_inverse = false)
        : f(rows, output_bf16, 4U, 719U), hyper(rows * ss), streams(rows * ss), hidden(rows * hs), routed(rows * hs),
          combined(rows * hs), gates(rows * 16U), norm(kWidth * 4U), inverse(rows * 16U + (guard_inverse ? 64U : 0U)), kind(site), in_place(alias) {
        c.rms_epsilon = epsilon;
        c.fast_prefill_wmma = true;
        std::mt19937 rng(812U + flavor);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        const auto bf16 = [](float v) {
            const auto b = std::bit_cast<std::uint32_t>(v);
            return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U);
        };
        // Whole-stream cases make subnormal/-0/eps=0 behavior observable; NaN
        // payloads in both streams and weights exercise pinned operand priority.
        constexpr std::uint16_t special[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x0080,
            0x7f80, 0xff80, 0x7fc1, 0x7f81, 0xffc3, 0x3f80};
        const auto value = [&](std::size_t i) {
            if (flavor == 1U) return special[(i / kHidden) % std::size(special)];
            if (flavor == 2U && i % 97U == 3U) return special[(i / 97U) % std::size(special)];
            return bf16(normal(rng));
        };
        hyper_host.resize(rows * ss / 2U);
        hidden_host.resize(rows * hs / 2U); routed_host.resize(rows * hs / 2U);
        for (std::size_t i = 0; i < hyper_host.size(); ++i) hyper_host[i] = value(i);
        for (std::size_t i = 0; i < hidden_host.size(); ++i) {
            hidden_host[i] = flavor == 1U ? std::uint16_t{0x8000U} : value(i + 31U);
            routed_host[i] = flavor == 1U ? std::uint16_t{0x8000U} : value(i + 73U);
        }
        std::vector<float> g(rows * 4U, 1.0F), w(kWidth);
        for (unsigned i = 0; i < kWidth; ++i) {
            w[i] = 1.0F + 0.1F * normal(rng);
            if (flavor == 2U && i % 89U == 2U) w[i] = std::bit_cast<float>(0xffc12345U);
        }
        gates.upload(g); norm.upload(w);
    }
    void* stream_data() const { return in_place && kind ? hyper.p : streams.p; }
    void* combined_data() const { return in_place ? hidden.p : combined.p; }
    void reset() {
        f.reset(0xa5);
        hyper.upload(hyper_host); streams.upload(hyper_host);
        hidden.upload(hidden_host); routed.upload(routed_host);
        check(hipMemset(combined.p, 0xa5, combined.n), "recompute combined poison");
        check(hipMemset(inverse.p, 0xa5, inverse.n), "recompute inverse poison");
        check(hipDeviceSynchronize(), "recompute reset drain");
    }
    void enqueue(bool on, unsigned output, unsigned low_band, hipStream_t stream,
        const std::vector<float>* inverse_override = nullptr) {
        strix::testing::Qpf2kInjectNormTestViews v;
        v.hyper = hyper.p; v.hyper_stride = ss; v.streams = stream_data(); v.streams_stride = ss;
        v.hidden = hidden.p; v.hidden_stride = hs; v.gates = static_cast<const float*>(gates.p); v.gates_stride = 16U;
        if (kind == 2U) { v.routed = routed.p; v.routed_stride = hs; v.combined = combined_data(); v.combined_stride = hs; }
        v.normalized = static_cast<float*>(f.normalized.p); v.normalized_stride = kNormalizedStride;
        v.norm = static_cast<const float*>(norm.p); v.rows = f.rows; v.norm_only = kind == 0U;
        v.recompute_inverse = on ? static_cast<float*>(inverse.p) : nullptr;
        strix::testing::enqueue_qpf2k_inject_norm_test(c, v, false, stream);
        if (inverse_override) {
            if (!on || inverse_override->size() * sizeof(float) != static_cast<std::size_t>(f.rows) * 16U)
                throw std::invalid_argument("inverse override extent");
            check(hipMemcpyAsync(inverse.p, inverse_override->data(), inverse_override->size() * sizeof(float), hipMemcpyHostToDevice, stream), "inverse override");
        }
        auto wide = f.views();
        if (on) {
            // Aliasing pitch: the folded inject + norm mirrors the stream rows into the normalized workspace rows.
            const bool mirror = kind != 0U && ss % 4096U == 0U;
            wide.recompute_input = mirror ? static_cast<const void*>(f.normalized.p) : static_cast<const void*>(v.streams);
            wide.recompute_stride = mirror ? std::uint64_t{kNormalizedStride} : ss;
            wide.recompute_inverse = v.recompute_inverse; wide.recompute_norm = v.norm;
        }
        if (output == 7U) { wide.output3_band = 16U; wide.low_band = low_band; }
        auto output_config = c;
        output_config.activation = f.bf16 ? strix::HipActivationStorage::bf16 : strix::HipActivationStorage::fp32;
        strix::testing::enqueue_wide_hyper_rows_test(output_config, wide, 4U, output, stream);
    }
    std::vector<unsigned char> written() const {
        const auto n = f.normalized.download();
        std::vector<unsigned char> out;
        for (unsigned r = 0; r < f.rows; ++r)
            out.insert(out.end(), n.begin() + r * kNormalizedStride + kWidth * 4U, n.begin() + (r + 1U) * kNormalizedStride);
        for (const auto& b : {f.injection.download(), f.output.download(), hyper.download(), streams.download(),
                hidden.download(), routed.download(), combined.download()}) out.insert(out.end(), b.begin(), b.end());
        std::vector<unsigned char> low16(f.rows * kRank * 2U);
        check(hipMemcpy(low16.data(), strix::testing::wide_hyper_rows_test_low16(), low16.size(), hipMemcpyDeviceToHost), "recompute low16 read");
        out.insert(out.end(), low16.begin(), low16.end());
        return out;
    }
    void compare(unsigned output, unsigned low_band, hipStream_t stream) {
        reset(); enqueue(false, output, low_band, stream);
        check(hipStreamSynchronize(stream), "recompute OFF");
        const auto expected = written();
        reset(); enqueue(true, output, low_band, stream);
        check(hipStreamSynchronize(stream), "recompute ON");
        const auto actual = written();
        if (expected != actual) {
            const auto at = std::mismatch(expected.begin(), expected.end(), actual.begin()).first - expected.begin();
            throw std::runtime_error("HC recompute byte mismatch at " + std::to_string(at));
        }
        // The normalized image must remain entirely unwritten in the ON arm.
        // (with an aliasing stream pitch its first half holds the stream mirror and nothing else).
        const bool mirror = kind != 0U && ss % 4096U == 0U;
        const std::size_t kept = mirror ? kWidth * 2U : 0U;
        const auto stream_rows = (in_place && kind ? hyper : streams).download();
        if (const auto n = f.normalized.download(); !n.empty())
            for (unsigned r = 0; r < f.rows; ++r) {
                if (mirror && std::memcmp(n.data() + r * kNormalizedStride, stream_rows.data() + r * ss, kept))
                    throw std::runtime_error("HC recompute stream mirror differs from the stream rows");
                if (std::memcmp(n.data() + r * kNormalizedStride + kept,
                    f.normalized_host.data() + r * kNormalizedStride + kept, kWidth * 4U - kept))
                    throw std::runtime_error("HC recompute stored the normalized image");
            }
        std::cout << "HC recompute EXACT rows " << f.rows << " site " << kind << " in_place " << in_place
                  << " output " << output << " lowband " << low_band << '\n';
    }
};

void fixtures_recompute_norm(hipStream_t stream) {
    for (unsigned rows : {1U, 7U, 8U, 31U, 33U, 264U})
        for (unsigned kind : {0U, 1U, 2U})
            for (unsigned flavor : {0U, 1U, 2U}) {
                RecomputeFixture f(rows, kind, flavor, true, flavor == 1U ? 0.0F : 1e-6F);
                f.compare(14U, 0U, stream);
                f.compare(7U, 0U, stream);
            }
    // Cross the 512-row low-table band boundary, including the partial last CTA.
    for (unsigned kind : {0U, 1U, 2U}) {
        RecomputeFixture f(520U, kind, 0U, false, std::numeric_limits<float>::denorm_min());
        for (unsigned mode : {1U, 2U, 3U}) f.compare(7U, mode, stream);
    }
    for (float eps : {-0.0F, -1e-6F, std::numeric_limits<float>::min(),
            std::numeric_limits<float>::infinity(), std::bit_cast<float>(0x7fc12345U)})
        for (unsigned kind : {0U, 1U, 2U}) {
            RecomputeFixture f(8U, kind, 1U, false, eps);
            f.compare(14U, 0U, stream); f.compare(7U, 0U, stream);
        }
    RecomputeFixture alias(8U, 0U, 0U, false, 1e-6F);
    auto v = alias.f.views();
    v.recompute_input = alias.stream_data(); v.recompute_stride = RecomputeFixture::ss;
    v.recompute_inverse = static_cast<const float*>(alias.inverse.p);
    v.recompute_norm = static_cast<const float*>(alias.norm.p);
    v.output = alias.stream_data(); v.output_stride = RecomputeFixture::ss;
    bool refused = false;
    try { strix::testing::enqueue_wide_hyper_rows_test(alias.c, v, 4U, 14U, stream); }
    catch (const std::invalid_argument&) { refused = true; }
    if (!refused) throw std::runtime_error("HC recompute accepted activation/stream aliasing");
}

void time_recompute_norm(unsigned kind, unsigned output, hipStream_t stream) {
    // Immutable inputs in the timed transaction: no reset/copy cost and no drift
    // between arms, including the combine variant. Every destination is overwritten.
    const unsigned timed_rows = [] { const char* v = std::getenv("STRIX_HC_RECOMPUTE_TIME_ROWS"); return v ? static_cast<unsigned>(std::atoi(v)) : 16384U; }();
    RecomputeFixture f(timed_rows, kind, 0U, false, 1e-6F);
    const unsigned low_band = output == 7U ? 3U : 0U;
    f.compare(output, low_band, stream);
    f.reset();
    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "recompute begin event"); check(hipEventCreate(&end), "recompute end event");
    const auto run = [&](bool on) {
        check(hipEventRecord(begin, stream), "recompute start");
        f.enqueue(on, output, low_band, stream);
        check(hipEventRecord(end, stream), "recompute stop"); check(hipEventSynchronize(end), "recompute time drain");
        float ms = 0.0F; check(hipEventElapsedTime(&ms, begin, end), "recompute elapsed"); return ms;
    };
    for (unsigned warmup = 0; warmup < 2U; ++warmup) { (void)run(false); (void)run(true); }
    std::array<float, 7> off{}, on{};
    for (unsigned pair = 0; pair < 7U; ++pair) {
        if (pair % 2U) { on[pair] = run(true); off[pair] = run(false); }
        else { off[pair] = run(false); on[pair] = run(true); }
        std::cout << "HC recompute pair " << pair << " OFF_ms " << off[pair] << " ON_ms " << on[pair] << '\n';
    }
    std::sort(off.begin(), off.end()); std::sort(on.begin(), on.end());
    std::cout << "HC recompute T" << timed_rows << " site " << kind << " output " << output << " median_OFF_ms " << off[3]
              << " median_ON_ms " << on[3] << " native_not_slower " << (on[3] <= off[3]) << '\n';
    check(hipEventDestroy(begin), "recompute begin destroy"); check(hipEventDestroy(end), "recompute end destroy");
}
