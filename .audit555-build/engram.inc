EngramHasher::EngramHasher(const deepseek41::EngramContract& c) : contract_(c) {
    const std::size_t layers = c.layer_ids.size(), cols = c.rows_per_token();
    require(c.max_ngram_size >= 2 && c.n_heads >= 1 && !c.layer_ids.empty() &&
                c.primes.size() == layers * cols && c.multipliers.size() == layers * c.max_ngram_size &&
                c.pad_id < c.token_map.size(),
            "Engram contract geometry");
    pad_ = c.token_map[c.pad_id];
    offsets_.resize(layers * cols);
    for (std::size_t e = 0; e < layers; ++e) {
        std::uint64_t sum = 0;
        for (std::size_t k = 0; k < cols; ++k) {
            offsets_[e * cols + k] = sum;
            sum += c.primes[e * cols + k];
        }
        require(sum == c.rows[e], "Engram primes do not sum to the table rows");
    }
}

void EngramHasher::push(std::uint32_t token, std::span<std::uint64_t> rows) {
    const auto& c = contract_;
    const std::size_t layers = c.layer_ids.size(), cols = c.rows_per_token(), n = c.max_ngram_size;
    const bool dead = (token & kEngramDeadBit) != 0U;
    token &= ~kEngramDeadBit;
    require(token < c.token_map.size() && rows.size() == layers * cols, "Engram hash request");
    history_.push_back(dead ? kDead : c.token_map[token]);
    const std::size_t t = history_.size() - 1;
    std::vector<std::uint64_t> ids(n);
    bool blocked = false;
    for (std::size_t s = 0; s < n; ++s) {
        blocked = blocked || t < s || history_[t - s] == kDead;  // nothing reads through a dead (image) position
        ids[s] = blocked ? pad_ : history_[t - s];
    }
    for (std::size_t e = 0; e < layers; ++e) {
        std::uint64_t rolling = ids[0] * c.multipliers[e * n];
        for (std::size_t s = 1; s < n; ++s) {
            rolling ^= ids[s] * c.multipliers[e * n + s];
            for (std::uint32_t h = 0; h < c.n_heads; ++h) {
                const std::size_t k = (s - 1) * c.n_heads + h;
                rows[e * cols + k] = rolling % c.primes[e * cols + k] + offsets_[e * cols + k];
            }
        }
    }
}

std::vector<std::uint64_t> EngramHasher::hash_sequence(std::span<const std::uint32_t> tokens) {
    reset();
    const std::size_t per = static_cast<std::size_t>(layers()) * columns();
    std::vector<std::uint64_t> out(tokens.size() * per);
    for (std::size_t i = 0; i < tokens.size(); ++i) push(tokens[i], std::span<std::uint64_t>(out.data() + i * per, per));
    return out;
}

void EngramHasher::truncate(std::uint64_t positions) {
    require(positions <= history_.size(), "Engram history cannot grow by truncation");
    history_.resize(positions);
}

