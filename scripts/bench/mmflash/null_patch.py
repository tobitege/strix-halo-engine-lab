#!/usr/bin/env python3
"""DIAGNOSTIC null-control hook for a scratch copy of src/minimax/engine/engine.hip (never merged into the engine).

Usage: python3 null_patch.py <scratch>/src/minimax/engine/engine.hip

STRIX_MM_DIAG_ATTN_ULP="L:seed"          every prefill window (n > 8 rows) of layer L: each finite non-zero value of the
                                         rank's attention output (before the o-projection) moves one ulp up or down by a
                                         hash of (index, seed). This is the gate's onset null; it logs "DIAG attn ulp: ...".
STRIX_MM_DIAG_ATTN_ULP="A-B:seed[:mag]"  every layer A..B, +-mag ulps (default 1), seed + 7919 x (L - A) per layer. Logs
                                         "DIAGM attn ulp: ..." so the gate witness never takes it for an onset null:
                                         diagnostic, absolute comparisons only.
Unset: no launch, bytes unchanged (measured: logits and every selection slot identical to the unpatched exact run).
"""
import sys
from pathlib import Path

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
launcher = Path(path).resolve().parents[3] / "apps/minimax_rank_main.cpp"
app = launcher.read_text(encoding="utf-8")
A3 = "namespace {"
A4 = "    const auto t0 = Clock::now();\n    // STRIX_MM_QUALITY_SELECT"
A5 = "    if (sel_dump) strix::minimax::msa::fused::set_select_dump(nullptr);\n    const double ttft = secs(t0, Clock::now());"
for anchor in (A3, A4, A5):
    assert app.count(anchor) == 1, ("launcher anchor", anchor, app.count(anchor))
A1 = "struct Engine::Impl {"
A2 = '            if (tl && !traced_batch) tl->mark("msa", false);'
assert src.count(A1) == 1, ("anchor 1", src.count(A1))
assert src.count(A2) == 1, ("anchor 2", src.count(A2))
K = r'''std::string mmflash_prompt_json;  // Set by the scratch launcher's serial session adapter.
// DIAGNOSTIC null control (scripts/bench/mmflash/null_patch.py, never merged): +-mag ulps on every finite non-zero value.
__global__ void diag_attn_ulp_kernel(float* x, unsigned long long n, unsigned seed, unsigned mag) {
    const unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned b;
    __builtin_memcpy(&b, x + i, 4);
    if ((b & 0x7f800000U) == 0x7f800000U || (b & 0x7fffffffU) == 0U) return;
    unsigned z = static_cast<unsigned>(i) * 0x9E3779B1U ^ (seed * 0x85EBCA6BU + static_cast<unsigned>(i >> 32));
    z ^= z >> 15;
    z *= 0x2C1B3C6DU;
    z ^= z >> 12;
    z *= 0x297A2D39U;
    z ^= z >> 15;
    b += (z & 1U) != 0U ? mag : 0U - mag;
    if ((b & 0x7f800000U) == 0x7f800000U) return;
    __builtin_memcpy(x + i, &b, 4);
}

'''
H = r'''            {   // DIAGNOSTIC null control (scripts/bench/mmflash/null_patch.py, never merged)
                struct DiagNull { int lo = -1, hi = -1; unsigned seed = 1, mag = 1; };
                static const DiagNull diag = [] {
                    DiagNull d;
                    const char* e = strix::envcfg::text("STRIX_MM_DIAG_ATTN_ULP");
                    if (e == nullptr || e[0] == 0) return d;
                    d.lo = d.hi = std::atoi(e);
                    const char* c = std::strchr(e, ':');
                    const char* r = std::strchr(e, '-');
                    if (r != nullptr && (c == nullptr || r < c)) d.hi = std::atoi(r + 1);
                    if (c != nullptr) d.seed = static_cast<unsigned>(std::atoi(c + 1));
                    const char* m = c != nullptr ? std::strchr(c + 1, ':') : nullptr;
                    if (m != nullptr) d.mag = static_cast<unsigned>(std::atoi(m + 1));
                    return d;
                }();
                if (diag.lo >= 0 && static_cast<int>(L) >= diag.lo && static_cast<int>(L) <= diag.hi && n > 8) {
                    const unsigned seed_l = diag.seed + 7919U * (L - static_cast<unsigned>(diag.lo));
                    const unsigned long long diag_n = static_cast<unsigned long long>(n) * kQLocal;
                    hipLaunchKernelGGL(diag_attn_ulp_kernel, dim3(static_cast<unsigned>((diag_n + 255) / 256)), dim3(256), 0, nullptr, attn, diag_n, seed_l, diag.mag);
                    const bool multi = diag.hi != diag.lo || diag.mag != 1U;
                    std::fprintf(stderr, "%s attn ulp: layer %u seed %u rows %zu pos0 %llu values %llu prompt %s\n", multi ? "DIAGM" : "DIAG", L, seed_l,
                                 static_cast<std::size_t>(n), static_cast<unsigned long long>(pos0), diag_n, mmflash_prompt_json.c_str());
                }
            }
'''
src = src.replace(A1, K + A1).replace(A2, H + A2)
app = '#include "strix/serve/json.hpp"\n' + app
app = app.replace(A3, "namespace strix::minimax::engine { extern std::string mmflash_prompt_json; }\n" + A3)
app = app.replace(A4, '''    mm::mmflash_prompt_json = strix::serve::Json(name).dump();
    const auto mmflash_wall = [] {
        return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    };
    const double mmflash_start = mmflash_wall();
''' + A4)
app = app.replace(A5, A5 + r'''
    const double mmflash_end = mmflash_wall();
    std::fprintf(stderr, "MMFLASH_SESSION {\"prompt\":%s,\"start\":%.9f,\"end\":%.9f,\"tokens\":%zu}\n",
                 mm::mmflash_prompt_json.c_str(), mmflash_start, mmflash_end, prompt.size());
    mm::mmflash_prompt_json.clear();
''')
open(path, "w", encoding="utf-8", newline="\n").write(src)
launcher.write_text(app, encoding="utf-8", newline="\n")
print("patched", path)
