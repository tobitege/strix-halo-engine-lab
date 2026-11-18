// Synthetic v3/v3.1 stage comparison; not a full-model qualification gate.
#include "cuda/models/minimax/fp32_identity.cu"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "minimax/rtxpf/fp32_identity_ref.hpp"
namespace fp = strix::minimax::rtxpf;
static void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(2); }
}
template<class T> static T* upload(const std::vector<T>& h) {
    T* d = nullptr; check(cudaMalloc(&d, h.size() * sizeof(T)));
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d;
}
static int run(unsigned tr) {
    constexpr unsigned in = 768, out = 256;
    fp::ref::Gen g{99};
    std::vector<unsigned> rows = tr ? std::vector<unsigned>{130, 200} : std::vector<unsigned>{130, 200, 40, 300};
    std::vector<fp::F32Job> jobs;
    std::vector<void*> allocations;
    for (unsigned j = 0; j < rows.size(); ++j) {
        const unsigned bits = j % 2 ? 5 : 3;
        std::vector<std::uint32_t> trellis((in / 16) * (out / 16) * bits * 8);
        for (auto& w : trellis) w = static_cast<std::uint32_t>(g.next());
        std::vector<std::uint16_t> suh(in), svh(out);
        for (auto& v : suh) v = g.half_scale();
        for (auto& v : svh) v = g.half_scale();
        auto* dt = upload(trellis); auto* du = upload(suh); auto* dv = upload(svh);
        allocations.insert(allocations.end(), {dt, du, dv});
        jobs.push_back({dt, du, dv, in, out, bits, rows[j], 0, 0});
    }
    auto ext = fp::layout_jobs(jobs, tr ? tr : 128);
    const unsigned njobs = static_cast<unsigned>(jobs.size());
    std::vector<fp::F32Job> tail;
    const unsigned tail_tiles = tr ? 0 : fp::split_tails(jobs, tail, ext.tiles);
    jobs.insert(jobs.end(), tail.begin(), tail.end());
    std::vector<float> x(ext.entries * in);
    for (auto& v : x) v = g.value(3, 12);
    auto* dx = upload(x); auto* dj = upload(jobs);
    float *xt, *raw, *y;
    check(cudaMalloc(&xt, ext.entries * in * sizeof(float)));
    check(cudaMalloc(&raw, ext.entries * out * sizeof(float)));
    check(cudaMalloc(&y, ext.entries * out * sizeof(float)));
    fp::GroupedCall c;
    c.jobs_d = dj; c.jobs = njobs; c.tiles = ext.tiles; c.tile_rows = tr ? tr : 128;
    c.tail_jobs = static_cast<unsigned>(tail.size()); c.tail_tiles = tail_tiles;
    c.input = in; c.output = out; c.entries = ext.entries;
    c.x = dx; c.x_stride = in; c.y = y; c.y_stride = out;
    cudaEvent_t ev[4]; for (auto& e : ev) check(cudaEventCreate(&e));
    std::vector<float> reference_raw, reference_y;
    int failures = 0;
    float samples[2][3][3] = {};
    for (unsigned rep = 0; rep < 3; ++rep) for (unsigned order = 0; order < 2; ++order) {
        const unsigned mode = rep % 2 ? 1 - order : order;
        c.core_version = mode ? 31 : 3;
        // Warm the same production dispatch before timing each interleaved batch.
        fp::enqueue_grouped(c, xt, raw, nullptr); check(cudaDeviceSynchronize());
        for (unsigned it = 0; it < 10; ++it) {
            check(cudaEventRecord(ev[0]));
            fp::dev::transform_kernel<<<(ext.entries * (in / 128) + 7) / 8, 256>>>(
                reinterpret_cast<const fp::dev::F32Job*>(dj), njobs, dx, in, nullptr, in, ext.entries, xt);
            check(cudaEventRecord(ev[1]));
            fp::launch_grouped_core(c, xt, raw, nullptr);
            check(cudaEventRecord(ev[2]));
            fp::dev::output_kernel<<<(ext.entries * (out / 128) + 7) / 8, 256>>>(
                reinterpret_cast<const fp::dev::F32Job*>(dj), njobs, raw, out, ext.entries, y, out, nullptr);
            check(cudaGetLastError()); check(cudaEventRecord(ev[3])); check(cudaEventSynchronize(ev[3]));
            for (unsigned stage = 0; stage < 3; ++stage) {
                float ms; check(cudaEventElapsedTime(&ms, ev[stage], ev[stage + 1]));
                samples[mode][rep][stage] += ms / 10;
            }
        }
        std::vector<float> rh(ext.entries * out), yh(rh.size());
        check(cudaMemcpy(rh.data(), raw, rh.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(yh.data(), y, yh.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (rep == 0 && mode == 0) { reference_raw = rh; reference_y = yh; }
        unsigned bad_raw = 0, bad_y = 0;
        for (std::size_t i = 0; i < rh.size(); ++i) {
            bad_raw += fp::ref::bits_of(rh[i]) != fp::ref::bits_of(reference_raw[i]);
            bad_y += fp::ref::bits_of(yh[i]) != fp::ref::bits_of(reference_y[i]);
        }
        failures |= bad_raw != 0 || bad_y != 0;
        std::printf("tile %u version %u repeat %u raw/y mismatch %u/%u hashes %016llx/%016llx\n",
            tr, c.core_version, rep, bad_raw, bad_y,
            static_cast<unsigned long long>(fp::ref::fnv(rh.data(), rh.size())),
            static_cast<unsigned long long>(fp::ref::fnv(yh.data(), yh.size())));
    }
    for (unsigned mode = 0; mode < 2; ++mode) {
        float median[3];
        for (unsigned stage = 0; stage < 3; ++stage) {
            std::array<float, 3> values{samples[mode][0][stage], samples[mode][1][stage], samples[mode][2][stage]};
            std::sort(values.begin(), values.end()); median[stage] = values[1];
        }
        std::printf("tile %u version %u main/tail %u/%u median transform/core/output %.3f/%.3f/%.3f ms\n",
                    tr, mode ? 31 : 3, ext.tiles, tail_tiles, median[0], median[1], median[2]);
    }
    for (auto e : ev) check(cudaEventDestroy(e));
    for (auto p : allocations) check(cudaFree(p));
    check(cudaFree(dx)); check(cudaFree(dj)); check(cudaFree(xt)); check(cudaFree(raw)); check(cudaFree(y));
    return failures;
}
int main() { return run(64) | run(128) | run(0); }
