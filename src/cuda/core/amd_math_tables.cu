#include "strix/cuda/amd_math_tables.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace strix::cuda::amdmath {
namespace {

std::vector<std::uint32_t> read_u32(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("amd math table missing: " + path);
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<std::uint32_t> v(static_cast<std::size_t>(n) / 4);
    const bool ok = std::fread(v.data(), 4, v.size(), f) == v.size();
    std::fclose(f);
    if (!ok) throw std::runtime_error("amd math table short read: " + path);
    return v;
}

template <class T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    if (cudaMalloc(&d, std::max<std::size_t>(v.size(), 1) * sizeof(T)) != cudaSuccess) throw std::runtime_error("amd math table alloc");
    if (!v.empty() && cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("amd math table upload");
    return d;
}

// kind 0: exp (kept as is, |x| in [2^-25, 1)); 1: rsq canonicalised to [1, 4); 2: rcp canonicalised to positive [1, 2).
void hard_list(const std::string& path, int kind, std::vector<std::uint32_t>& in, std::vector<float>& out) {
    const auto raw = read_u32(path);
    const std::size_t n = raw.size() / 2;
    std::map<std::uint32_t, std::uint32_t> m;
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t b = raw[i], y = raw[n + i];
        // raw-instruction hard entries outside the emulated domain (subnormal or special inputs) are dropped
        if (kind != 0 && (((b >> 23) & 255U) == 0U || ((b >> 23) & 255U) == 255U)) continue;
        if (kind == 0) {
            const std::uint32_t mag = b & 0x7fffffffU;
            if (mag < (102U << 23) || mag >= (127U << 23)) continue;
        } else if (kind == 1) {
            const int e = static_cast<int>(b >> 23) - 127;
            const int k = e >= 0 ? e / 2 : -((-e + 1) / 2);
            b = (b & 0x007fffffU) | (static_cast<std::uint32_t>(e - 2 * k + 127) << 23);
            y = static_cast<std::uint32_t>(static_cast<int>(y) + (k << 23));
        } else {
            const int e = static_cast<int>((b >> 23) & 255U) - 127;
            b = (b & 0x007fffffU) | (127U << 23);
            y = static_cast<std::uint32_t>(static_cast<int>(y & 0x7fffffffU) + (e << 23));
            if ((y >> 23) == 0 || (y >> 23) >= 255) continue;
        }
        const auto it = m.find(b);
        if (it != m.end() && it->second != y) throw std::runtime_error("amd math hard list is not scale invariant: " + path);
        m[b] = y;
    }
    for (const auto& kv : m) {
        in.push_back(kv.first);
        float f;
        std::memcpy(&f, &kv.second, 4);
        out.push_back(f);
    }
}

}  // namespace

Tables load_tables(const std::string& dir) {
    std::vector<std::uint32_t> ein, rin, cin;
    std::vector<float> eout, rout, cout;
    hard_list(dir + "/hard_v_exp_f32.bin", 0, ein, eout);
    hard_list(dir + "/hard_v_rsq_f32.bin", 1, rin, rout);
    hard_list(dir + "/hard_v_rcp_f32.bin", 2, cin, cout);
    Tables t{};
    t.exp_bits = upload(read_u32(dir + "/tbl_exp_pm.bin"));
    t.rsq_bits = upload(read_u32(dir + "/tbl_rsq_1_4.bin"));
    t.rcp_bits = upload(read_u32(dir + "/tbl_rcp_1_2.bin"));
    t.exp_hard_in = upload(ein); t.exp_hard_out = upload(eout); t.exp_hard_n = static_cast<int>(ein.size());
    t.rsq_hard_in = upload(rin); t.rsq_hard_out = upload(rout); t.rsq_hard_n = static_cast<int>(rin.size());
    t.rcp_hard_in = upload(cin); t.rcp_hard_out = upload(cout); t.rcp_hard_n = static_cast<int>(cin.size());
    std::vector<float2> tab(129);
    for (int n = -64; n <= 64; ++n) {
        const double d = std::exp2(static_cast<double>(n) / 64.0);
        const float hi = static_cast<float>(d);
        tab[static_cast<std::size_t>(n + 64)] = make_float2(hi, static_cast<float>(d - static_cast<double>(hi)));
    }
    t.exp2_tab = upload(tab);
    return t;
}

void free_tables(Tables& t) {
    for (const void* p : {static_cast<const void*>(t.exp_bits), static_cast<const void*>(t.rsq_bits), static_cast<const void*>(t.rcp_bits),
                          static_cast<const void*>(t.exp_hard_in), static_cast<const void*>(t.exp_hard_out),
                          static_cast<const void*>(t.rsq_hard_in), static_cast<const void*>(t.rsq_hard_out),
                          static_cast<const void*>(t.rcp_hard_in), static_cast<const void*>(t.rcp_hard_out),
                          static_cast<const void*>(t.exp2_tab)})
        if (p) cudaFree(const_cast<void*>(p));
    t = Tables{};
}

}  // namespace strix::cuda::amdmath
