// Host tests: shared per-request reservation and CUDA Qwen policy with simulated serialized/cohort slot views.
// These test policy equivalence, not the production entry points' wiring to the policy.
#include "../apps/cuda_qwen_serve_policy.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what, int line) {
    if (!ok) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
        ++failures;
    }
}
#define CHECK(x) check((x), #x, __LINE__)

using namespace strix::serve;

// CUDA Qwen session bytes (qwen_prefill_driver.cu rows_bytes_at + fixed): 31,616 B/position (+ pooled rounding) + 233 MB
std::uint64_t qwen_bytes(std::uint64_t p) {
    const std::uint64_t one = p * 2048 + p * 256 + (p + 3) / 4 * 512;
    return 13U * one + 233000000ULL;
}

// The serialized path (one pinned slot, session 0) and the cohort admission (slot 0 pinned + empty slots) as the
// server builds them: the capacity a request gets must be the same.
std::uint64_t serialized(const cuda_qwen_serve::SessionPolicy& p, std::uint64_t n, std::uint64_t t, std::uint64_t cap0) {
    const auto need = cuda_qwen_serve::request_positions(p, n, t);
    const std::vector<SlotSession> one{{false, true, true, cap0, qwen_bytes(cap0)}};
    const auto plan = plan_admission(one, need, qwen_bytes, 64ULL << 30U, 2048ULL << 20U);
    return plan.action == AdmitAction::reuse ? cap0 : plan.action == AdmitAction::resize ? need : 0;
}
std::uint64_t cohort(const cuda_qwen_serve::SessionPolicy& p, std::uint64_t n, std::uint64_t t, std::uint64_t cap0) {
    const auto need = cuda_qwen_serve::request_positions(p, n, t);
    std::vector<SlotSession> slots(4);
    slots[0] = {false, true, true, cap0, qwen_bytes(cap0)};
    const auto plan = plan_admission(slots, need, qwen_bytes, 64ULL << 30U, 2048ULL << 20U);
    if (plan.action == AdmitAction::reuse) return plan.positions;
    return plan.action == AdmitAction::resize || plan.action == AdmitAction::build ? need : 0;
}
}  // namespace

int main() {
    // ---- reservation_positions
    ReservationRule r{34816, 256, 16, 0};
    CHECK(reservation_positions(r, 1024, 128) == 1280);   // 1168 -> 1280
    CHECK(reservation_positions(r, 1024, 96) == 1280);    // 1136 -> 1280
    CHECK(reservation_positions(r, 1008, 0) == 1024);     // 1024: already a multiple
    CHECK(reservation_positions(r, 1008, 8) == 1280);     // 1032 -> 1280
    CHECK(reservation_positions(r, 2, 1) == 256);         // floor: one granule
    CHECK(reservation_positions(r, 34000, 4096) == 34816);  // capped
    CHECK(reservation_positions(r, 100, UINT64_MAX) == 34816);  // no overflow
    CHECK(reservation_positions(ReservationRule{1000, 256, 16, 0}, 900, 50) == 1000);  // 966 -> 1024 > cap -> cap
    // DS41 --pool-positions rule (deepseek41_serve_main.cpp pool_rule): exactly the old min(context, n + max_tokens + 1)
    for (const std::uint64_t ctx : {4096ULL, 65536ULL})
        for (const std::uint64_t n : {1ULL, 2ULL, 1000ULL, 4095ULL, 65535ULL})
            for (const std::uint64_t t : {0ULL, 1ULL, 255ULL, 4096ULL, 1ULL << 40U})
                CHECK(reservation_positions(ReservationRule{ctx, 1, 1, 1}, n, t) == std::min<std::uint64_t>(ctx, n + t + 1U));
    // ---- plan_admission
    const std::uint64_t G = 1ULL << 30U, M = 1ULL << 20U;
    const auto b = [](std::uint64_t p) { return p * 1000ULL; };
    {   // reuse the smallest idle session that holds the request
        std::vector<SlotSession> s{{false, true, true, 8192, b(8192)}, {false, true, false, 2048, b(2048)}, {true, true, false, 1280, b(1280)}};
        const auto p = plan_admission(s, 1280, b, 0, 0);
        CHECK(p.action == AdmitAction::reuse && p.slot == 1 && p.positions == 2048);
    }
    {   // build in an empty slot when free memory holds it (idle small session kept)
        std::vector<SlotSession> s{{true, true, true, 1280, b(1280)}, {false, true, false, 512, b(512)}, {false, false, false, 0, 0}};
        const auto p = plan_admission(s, 1280, b, 10 * M, 2 * M);
        CHECK(p.action == AdmitAction::build && p.slot == 2 && p.release.empty() && p.bytes == b(1280));
    }
    {   // not enough free: re-size the idle too-small session (its bytes come back)
        std::vector<SlotSession> s{{true, true, true, 1280, b(1280)}, {false, true, false, 512, b(512)}, {false, false, false, 0, 0}};
        const auto p = plan_admission(s, 1280, b, b(1280) + 2 * M - b(512), 2 * M);
        CHECK(p.action == AdmitAction::resize && p.slot == 1 && p.release.empty());
    }
    {   // release idle unpinned sessions, largest first, only as many as needed; pinned session 0 is never released
        std::vector<SlotSession> s{{false, true, true, 4096, b(4096)}, {false, true, false, 3000, b(3000)}, {false, true, false, 2000, b(2000)},
                                   {false, false, false, 0, 0}};
        const auto p = plan_admission(s, 6000, b, b(1500), 0);
        // grow target = largest too-small idle (slot 0, 4096): free 1.5 + 4.096 + release slot 1 (3.0) >= 6.0
        CHECK(p.action == AdmitAction::resize && p.slot == 0 && p.release.size() == 1 && p.release[0] == 1);
    }
    {   // wait while something runs, reject when nothing does
        std::vector<SlotSession> s{{true, true, true, 34816, b(34816)}, {false, false, false, 0, 0}};
        CHECK(plan_admission(s, 34816, b, b(1000), 0).action == AdmitAction::wait);
        s[0].busy = false;
        CHECK(plan_admission(s, 40000, b, b(1000), 0).action == AdmitAction::reject);
        std::vector<SlotSession> all{{true, true, true, 100, b(100)}};
        const auto w = plan_admission(all, 100, b, G, 0);
        CHECK(w.action == AdmitAction::wait && !w.why.empty());
    }
    {   // A smaller pinned target can return bytes unavailable when growing the largest unpinned target.
        std::vector<SlotSession> s{{false, true, true, 256, qwen_bytes(256)},
                                  {false, true, false, 1280, qwen_bytes(1280)}};
        const auto margin = 64 * M;  // Also keeps the both-sessions boundary positive with fixed session bytes.
        const auto free_bytes = margin + qwen_bytes(8192) - qwen_bytes(1280) - 1;
        const auto p = plan_admission(s, 8192, qwen_bytes, free_bytes, margin);
        CHECK(p.action == AdmitAction::resize && p.slot == 0 && p.positions == 8192);
        CHECK(p.release == std::vector<std::size_t>{1});
        CHECK(p.bytes == qwen_bytes(8192));
        // At the exact boundary, reclaiming both sessions fits; one byte below it must reject.
        const auto boundary = margin + qwen_bytes(8192) - qwen_bytes(256) - qwen_bytes(1280);
        CHECK(plan_admission(s, 8192, qwen_bytes, boundary, margin).action == AdmitAction::resize);
        CHECK(plan_admission(s, 8192, qwen_bytes, boundary - 1, margin).action == AdmitAction::reject);
        // A busy pinned session cannot be a fallback target or enter the release list.
        s[0].busy = true;
        const auto busy = plan_admission(s, 8192, qwen_bytes, free_bytes, margin);
        CHECK(busy.action == AdmitAction::wait && busy.release.empty());
        // Neither the pin nor the fallback may depend on slot index zero.
        s[0].busy = false;
        std::swap(s[0], s[1]);
        const auto swapped = plan_admission(s, 8192, qwen_bytes, free_bytes, margin);
        CHECK(swapped.action == AdmitAction::resize && swapped.slot == 1);
        CHECK(swapped.release == std::vector<std::size_t>{0});
    }
    {   // slot count follows memory: 6 GB free, 2 GB margin, 1K prompt + 128 tokens at 34,816 cap (CUDA Qwen numbers)
        cuda_qwen_serve::SessionPolicy pol{34816, cuda_qwen_serve::kDefaultGranule, true};
        std::vector<SlotSession> s(16);
        s[0] = {false, true, true, 256, qwen_bytes(256)};
        std::uint64_t free = 6ULL * G;
        unsigned placed = 0;
        for (;;) {
            const auto need = cuda_qwen_serve::request_positions(pol, 1024, 128);
            const auto p = plan_admission(s, need, qwen_bytes, free, 2048ULL * M);
            if (p.action != AdmitAction::build && p.action != AdmitAction::resize && p.action != AdmitAction::reuse) break;
            const auto before = s[p.slot].built ? s[p.slot].bytes : 0;
            s[p.slot] = {true, true, p.slot == 0, need, qwen_bytes(need)};
            free = free + before - qwen_bytes(need);
            ++placed;
        }
        std::printf("1K+128 at 34816 cap, 6 GiB free, 2 GiB margin: %u sessions (fixed max_positions: %llu)\n", placed,
                    static_cast<unsigned long long>(1 + (6ULL * G - qwen_bytes(256) - 2048ULL * M) / qwen_bytes(34816)));
        CHECK(placed >= 14);
    }
    // ---- Policy equivalence with simulated serialized/cohort views, every request shape, both modes
    for (const bool per_request : {true, false}) {
        const cuda_qwen_serve::SessionPolicy pol{34816, 256, per_request};
        for (const std::uint64_t n : {2ULL, 100ULL, 1024ULL, 8192ULL, 32768ULL, 34815ULL})
            for (const std::uint64_t t : {1ULL, 128ULL, 4096ULL, 65536ULL})
                for (const std::uint64_t cap0 : {cuda_qwen_serve::initial_positions(pol), std::uint64_t{34816}}) {
                    const auto a = serialized(pol, n, t, cap0), c = cohort(pol, n, t, cap0);
                    if (a != c || a < std::min<std::uint64_t>(34816, n + t)) {
                        std::fprintf(stderr, "resolver mismatch n=%llu t=%llu cap0=%llu: serialized %llu cohort %llu\n",
                                     static_cast<unsigned long long>(n), static_cast<unsigned long long>(t),
                                     static_cast<unsigned long long>(cap0), static_cast<unsigned long long>(a),
                                     static_cast<unsigned long long>(c));
                        ++failures;
                    }
                }
    }
    CHECK(cuda_qwen_serve::initial_positions({34816, 256, false}) == 34816);
    CHECK(cuda_qwen_serve::request_positions({34816, 256, false}, 1024, 128) == 34816);
    if (failures) return 1;
    std::printf("session reservation tests: OK\n");
    return 0;
}
