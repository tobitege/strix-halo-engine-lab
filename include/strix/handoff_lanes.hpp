#pragma once
// strix::handoff::Lanes: urgent-first scheduling for several Sender sessions that leave through ONE egress link (a producer
// sending to both ranks of a pair over one NIC). Urgent pieces (data that gates the receiver's pipeline, e.g. layer-cut
// activations) go to their Sender at once. Bulk pieces (state needed only at the end, e.g. KV for the import) wait here
// and are handed to their Sender only while (a) no urgent piece of ANY member is unreleased and (b) that member holds
// fewer than bulk_ahead_bytes of handed-but-unreleased bulk. TCP shares a link fairly between connections, so without
// (a) a bulk stream to rank 1 halves the bandwidth of rank 0's urgent piece; without (b) a Sender's own queue runs ahead
// and an urgent piece waits behind it (Sender keeps submit order on the wire: seq is assigned at submit).
// Scheduling only: piece bytes, headers and the per-session seq order the receiver checks are the Sender's. Release
// contract: every callback runs exactly once (by the Sender, or here when a queued bulk piece is dropped on abort).
// Identity pieces release after their last byte reached the socket; bp-v1 pieces after encoding (bound the encoded
// backlog with SenderOptions::max_queued_wire_bytes and the kernel backlog with SenderOptions::socket_buffer).
#include "strix/handoff.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace strix::handoff {

class Lanes {
public:
    struct Options {
        std::size_t bulk_ahead_bytes = std::size_t{4} << 20;  // per member; one piece always passes when nothing is ahead
        bool group_gate = true;  // false: bulk ignores unreleased urgent pieces (urgent still jumps the local queue)
    };
    struct Stats {
        double bulk_held_s = 0;  // feeder time with bulk ready but held by an unreleased urgent piece
        std::uint64_t urgent_pieces = 0, bulk_pieces = 0, bulk_bytes = 0;
    };

    Lanes(std::vector<Sender*> senders, Options options) : s_(std::make_shared<State>()) {
        if (senders.empty()) throw std::invalid_argument("handoff lanes: no senders");
        s_->senders = std::move(senders);
        s_->opt = options;
        s_->queues.resize(s_->senders.size());
        s_->ahead.assign(s_->senders.size(), 0);
        for (auto* sender : s_->senders) sender->on_failure([weak = std::weak_ptr<State>(s_)](std::exception_ptr e) {
            if (auto s = weak.lock()) s->cancel(e);
        });
        feeder_ = std::thread([s = s_] { feed(*s, s); });
    }
    ~Lanes() { abort(); if (feeder_.joinable()) feeder_.join(); }
    Lanes(const Lanes&) = delete;
    Lanes& operator=(const Lanes&) = delete;

    // On the caller's thread (Sender::submit may wait on its own max_pending_plain_bytes).
    void urgent(unsigned dst, const PieceHeader& header, std::span<const bp_codec::Region> regions,
                std::function<void()> release, bool identity = true) {
        ReleaseGuard guard{std::move(release)};
        auto& S = *s_;
        S.check_dst(dst);
        std::function<void()> done = [s = s_, fn = guard.fn]() mutable {
            if (fn) { try { fn(); } catch (...) {} }
            { std::lock_guard<std::mutex> g(s->m); --s->urgent_open; }
            s->cv.notify_all();
        };
        {
            std::lock_guard<std::mutex> g(S.m);
            if (S.error) std::rethrow_exception(S.error);
            if (S.stop) throw std::runtime_error("handoff lanes: urgent after abort");
            ++S.urgent_open; ++S.stats.urgent_pieces;
        }
        guard.fn = {};
        S.senders[dst]->submit(header, regions, std::move(done), identity);
    }
    void bulk(unsigned dst, const PieceHeader& header, std::vector<bp_codec::Region> regions, std::function<void()> release) {
        auto& S = *s_;
        S.check_dst(dst);
        std::unique_lock<std::mutex> g(S.m);
        if (S.error || S.stop) {
            const auto e = S.error;
            g.unlock();
            if (release) release();
            if (e) std::rethrow_exception(e);
            throw std::runtime_error("handoff lanes: bulk after abort");
        }
        S.queues[dst].push_back(Bulk{header, std::move(regions), std::move(release)});
        S.cv.notify_all();
    }
    // Every queued bulk piece handed to its Sender (call before Sender::finish; submit is then quiet).
    void drain() {
        auto& S = *s_;
        std::unique_lock<std::mutex> g(S.m);
        S.cv.wait(g, [&] { return S.error || S.stop || (S.handing == 0 && S.all_empty()); });
        if (S.error) {
            const auto error = S.error;
            g.unlock();
            // Never join from a Sender callback. Cancel and join here before borrowed storage can be reclaimed.
            for (auto* sender : S.senders) sender->abort("handoff lanes failed");
            if (feeder_.joinable()) feeder_.join();
            std::rethrow_exception(error);
        }
        if (S.stop) throw std::runtime_error("handoff lanes: aborted");
    }
    // Drops queued bulk (its callbacks run here) and stops the feeder; pieces already handed belong to their Sender.
    void abort() noexcept {
        s_->cancel({});
    }
    [[nodiscard]] Stats stats() const { std::lock_guard<std::mutex> g(s_->m); return s_->stats; }

private:
    struct ReleaseGuard {
        std::function<void()> fn;
        ~ReleaseGuard() { if (fn) { try { fn(); } catch (...) {} } }
    };
    struct Bulk {
        PieceHeader header;
        std::vector<bp_codec::Region> regions;
        std::function<void()> release;
    };
    // Shared with the release callbacks, which may run on a Sender thread after this object is gone.
    struct State {
        std::vector<Sender*> senders;
        Options opt;
        std::mutex m;
        std::condition_variable cv;
        std::vector<std::deque<Bulk>> queues;
        std::vector<std::size_t> ahead;
        unsigned urgent_open = 0, handing = 0, rr = 0;
        bool stop = false;
        std::exception_ptr error;
        Stats stats;
        void cancel(std::exception_ptr e) noexcept {
            std::vector<std::deque<Bulk>> dropped;
            {
                std::lock_guard<std::mutex> g(m);
                if (!error) error = e;
                stop = true;
                dropped.swap(queues);
            }
            cv.notify_all();
            for (auto& q : dropped) for (auto& b : q)
                if (b.release) { try { b.release(); } catch (...) {} }
        }
        void check_dst(unsigned dst) const { if (dst >= senders.size()) throw std::out_of_range("handoff lanes: destination"); }
        bool all_empty() const { for (const auto& q : queues) if (!q.empty()) return false; return true; }
        // Round robin over members with queued bulk and room ahead; -1 = none.
        int pick(bool advance) {
            const auto n = static_cast<unsigned>(queues.size());
            for (unsigned i = 0; i < n; ++i) {
                const unsigned d = (rr + i) % n;
                if (!queues[d].empty() && (ahead[d] == 0 || ahead[d] < opt.bulk_ahead_bytes)) {
                    if (advance) rr = d + 1;
                    return static_cast<int>(d);
                }
            }
            return -1;
        }
    };
    static void feed(State& S, const std::shared_ptr<State>& keep) {
        for (;;) {
            Bulk b;
            unsigned dst = 0;
            std::size_t bytes = 0;
            {
                std::unique_lock<std::mutex> g(S.m);
                for (;;) {
                    if (S.stop || S.error) return;
                    const bool ready = S.pick(false) >= 0;
                    if (ready && (S.urgent_open == 0 || !S.opt.group_gate)) break;
                    const auto t0 = std::chrono::steady_clock::now();
                    S.cv.wait(g);
                    if (ready) S.stats.bulk_held_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                }
                dst = static_cast<unsigned>(S.pick(true));
                b = std::move(S.queues[dst].front());
                S.queues[dst].pop_front();
                for (const auto& r : b.regions) bytes += r.bytes;
                S.ahead[dst] += bytes;
                ++S.handing; ++S.stats.bulk_pieces; S.stats.bulk_bytes += bytes;
            }
            try {
                S.senders[dst]->submit(b.header, b.regions, [s = keep, dst, bytes, fn = std::move(b.release)]() mutable {
                    if (fn) fn();
                    { std::lock_guard<std::mutex> g(s->m); s->ahead[dst] -= bytes; }
                    s->cv.notify_all();
                }, false);
            } catch (...) {
                S.cancel(std::current_exception());
            }
            { std::lock_guard<std::mutex> g(S.m); --S.handing; }
            S.cv.notify_all();
        }
    }

    std::shared_ptr<State> s_;
    std::thread feeder_;
};

}  // namespace strix::handoff
