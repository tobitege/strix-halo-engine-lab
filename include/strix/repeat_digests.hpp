#pragma once
// Equality of the repetitions of one run inside one process (host only; any engine's --repeat loop).
//
// A quality gate (KL, top-1, dNLL floor) or a defaults parity test does not prove that a warm repetition computes
// the bits of the cold one. Each repetition's digests are recorded here and compared with the first repetition of
// the same arm (arms of an A/B loop are compared only with themselves); report() states the verdict on its own line
// and equal() lets the tool fail loudly instead of writing a normal-looking receipt from the last run.
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace strix {

class RepeatDigests {
public:
    struct Run {
        std::uint32_t index = 0;
        std::string arm;          // "" when the loop has a single arm
        std::uint64_t logits = 0;  // e.g. FNV-1a of the logits
        std::uint64_t streams = 0; // digest of per-layer stream hashes (0: not collected)
        std::uint32_t next_token = 0;
    };

    void add(Run r) { runs_.push_back(std::move(r)); }
    [[nodiscard]] const std::vector<Run>& runs() const noexcept { return runs_; }

    // The first repetition whose digests differ from the first run of its arm (nullptr: all equal).
    [[nodiscard]] const Run* first_mismatch(const Run** reference = nullptr) const noexcept {
        for (std::size_t i = 0; i < runs_.size(); ++i)
            for (std::size_t j = 0; j < i; ++j) {
                if (runs_[j].arm != runs_[i].arm) continue;
                if (!same(runs_[j], runs_[i])) {
                    if (reference) *reference = &runs_[j];
                    return &runs_[i];
                }
                break;  // compared with the first run of the arm
            }
        return nullptr;
    }
    [[nodiscard]] bool equal() const noexcept { return first_mismatch() == nullptr; }

    // Repetitions and comparison counts per arm, then the equality verdict.
    [[nodiscard]] std::string report(const std::string& who) const {
        std::string out;
        char line[256];
        for (const auto& r : runs_) {
            std::snprintf(line, sizeof line, "%s repeat %u%s%s: next token %u, logits_fnv %016llx, streams %016llx\n", who.c_str(), r.index,
                          r.arm.empty() ? "" : " arm ", r.arm.c_str(), r.next_token, static_cast<unsigned long long>(r.logits),
                          static_cast<unsigned long long>(r.streams));
            out += line;
        }
        if (runs_.empty()) out += who + " repeat evidence: 0 comparisons (insufficient repetitions)\n";
        for (std::size_t i = 0; i < runs_.size(); ++i) {
            bool first = true;
            for (std::size_t j = 0; j < i; ++j)
                if (runs_[j].arm == runs_[i].arm) { first = false; break; }
            if (!first) continue;
            std::size_t count = 0;
            for (const auto& r : runs_)
                if (r.arm == runs_[i].arm) ++count;
            std::snprintf(line, sizeof line, "%s repeat arm %s: %zu runs, %zu comparisons%s\n", who.c_str(),
                          runs_[i].arm.empty() ? "<default>" : runs_[i].arm.c_str(), count, count - 1U,
                          count < 2U ? " (insufficient repetitions)" : "");
            out += line;
        }
        const Run* ref = nullptr;
        if (const Run* bad = first_mismatch(&ref)) {
            std::snprintf(line, sizeof line,
                          "%s REPEAT DIGEST MISMATCH: run %u (logits_fnv %016llx, streams %016llx, next %u) != run %u (logits_fnv %016llx, streams %016llx, next %u)\n",
                          who.c_str(), bad->index, static_cast<unsigned long long>(bad->logits), static_cast<unsigned long long>(bad->streams),
                          bad->next_token, ref->index, static_cast<unsigned long long>(ref->logits),
                          static_cast<unsigned long long>(ref->streams), ref->next_token);
        } else {
            std::snprintf(line, sizeof line, "%s repeat digests: %zu runs, equal\n", who.c_str(), runs_.size());
        }
        out += line;
        return out;
    }

private:
    static bool same(const Run& a, const Run& b) noexcept {
        return a.logits == b.logits && a.streams == b.streams && a.next_token == b.next_token;
    }
    std::vector<Run> runs_;
};

}  // namespace strix
