"""Extract unchanged host dispatch/control code for CPU-only regression tests.
CUDA kernels are not compiled or emulated; route and ownership behavior is exercised.
"""
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
def extract(path, begin, end):
    text = (root / path).read_text()
    return text[text.index(begin):text.index(end, text.index(begin))]
(out / "dispatch.inc").write_text(extract("src/cuda_core/dense/dense.cu", "cudaError_t dense_rows(", "cudaError_t embedding_rows("))
(out / "mma_support.inc").write_text(extract("src/cuda_core/dense/dense_mma.cu", "bool dense_mma_supported(", "cudaError_t dense_mma_launch("))
text = (root / "src/cuda_core/dense/nvfp4.cu").read_text()
parts = []
for begin, end in [("using WorkspaceKey", "template <"), ("cudaError_t ensure_ws(", "__global__ void amax_kernel_bf16"), ("cudaError_t nvfp4_reserve_workspace(", "namespace {")]:
    a = text.index(begin)
    parts.append(text[a:text.index(end, a)])
(out / "workspace.inc").write_text("\n".join(parts))
text = (root / "apps/cuda_qwen35_decode_main.cu").read_text()
a = text.index("#ifdef _WIN32", text.index("struct Clock")) if "struct Clock" in text else text.index("#ifdef _WIN32")
opening = text[a:text.index("#endif", a)+len("#endif")]
a = text.index("#ifdef _WIN32", a+len(opening))
closing = text[a:text.index("#endif", a)+len("#endif")]
(out / "sampler.inc").write_text("#undef _WIN32\n#define _popen forbidden_windows_popen\n#define _pclose forbidden_windows_pclose\n#define popen qwen35_guard_popen\n#define pclose qwen35_guard_pclose\nstatic FILE* popen(const char*, const char*) { return stdout; }\nstatic int pclose(FILE*) { return 0; }\nstatic int portable_sampler() {\n" + opening + "\n" + closing + "\nreturn p ? 0 : 1;\n}\n")
(out / "maxrel.inc").write_text(extract("tests/cuda_core/cuda_core_test.cu", "static double maxrel(", "static std::uint32_t* dpos("))

# Audit 686: compile host control code verbatim against an instrumented CUDA runtime.
def body(path, marker):
    text = (root / path).read_text()
    a = text.index(marker)
    b = text.index("{", a)
    depth = 1
    i = b + 1
    while depth:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return text[a:i]

big = "src/cuda_core/dense/dense_big.cu"
eng = "src/qwen35/cuda/engine.cu"
hip = "src/qwen35/hip/engine.hip"
parts = [body(big, "struct Table {") + ";", body(big, "struct BigCtx {") + ";",
         "using Scratch = BigCtx; enum Mode { kChain, kFold, kSplit };",
         body(big, "Scratch& scratch()"), body(big, "cudaError_t bind_device("),
         body(big, "const Table* get_table("), body(big, "void dense_big_release("),
         body(big, "BigCtx::~BigCtx()"), body(big, "BigCtx* dense_big_ctx_create()"),
         body(big, "void dense_big_ctx_destroy("), body(big, "bool dense_big_fast8_supported()"),
         body(big, "bool dense_big_supported(")]
(out / "audit_big.inc").write_text("\n".join(parts))
parts = [body(eng, "struct Slot {") + ";", body(eng, "struct Pf {") + " pf;",
         body(eng, "[[nodiscard]] const char* big_fallback()"), body(eng, "[[nodiscard]] int pf_terms()"),
         body(eng, "void pf_make_slot("), body(eng, "void pf_init(")]
text = (root / eng).read_text()
a = text.index("        if (n == 0)", text.index("    RowsOut prefill("))
b = text.index("        last_rows = 0;", a)
parts.append("void admit(const std::uint32_t* tokens, std::uint32_t n) {\n" + text[a:b] + "last_rows = 0; ++work; }")
a = text.index("CK(cc::dense_rows(head, xn, logits, 1,")
b = text.index(";", a) + 1
parts.append("void final_head() { " + text[a:b] + " }")
parts.append(body(eng, "struct Events {") + ";")
(out / "audit_engine.inc").write_text("\n".join(parts))
parts = [body(hip, "bool wide_ok(")]
# The first linear marker is a declaration; use the definition after glu16.
text = (root / hip).read_text()
a = text.index("void linear(const Tensor&", text.index("void glu16("))
b = text.index("void rmsnorm(", a)
code = text[a:b]
code = code.replace("silu_mul_kernel<<<1024, 256, 0, s>>>", "silu_mul_host")
(out / "audit_hip.inc").write_text(parts[0] + "\n" + code)
text = (root / "apps/hip_qwen35_spec_main.hip").read_text()
a = text.index("        if (pf_twice) {")
b = text.index("            const unsigned S = 9;", a)
code = text[a:b] + "sampled = true; } }"
(out / "audit_bench.inc").write_text("void bench() {" + code)
text = (root / "apps/cuda_qwen35_decode_main.cu").read_text()
a = text.index("        if (pf_repeat == 0)")
# The argument checks end where the first early-exit mode (pack save, bandwidth probe) starts.
b = min(text.find(m, a) for m in ("        if (!save_pack.empty())", "        if (bwp)") if text.find(m, a) > a)
(out / "audit_cli.inc").write_text("void cli(unsigned pf_repeat, unsigned pf_slots) {" + text[a:b] + "}")

# Keep host attribute initialization verbatim, replacing only the kernel symbol/size.
import re
parts = []
for name, marker, stop, kernel in [("big", "cudaError_t launch(", "    BigArgs b = a;", 1),
                                   ("big8", "cudaError_t launch8(", "    BigArgs b = a;", 2)]:
    text = body(big, marker)
    text = text[text.index("{", text.index(marker)) + 1:text.index(stop)]
    text = re.sub(r"    using C = [^;]+;", "", text)
    text = re.sub(r"big_kernel(?:_f8)?<[^>]+>", str(kernel), text).replace("C::SMEM", "1")
    parts.append("cudaError_t attr_" + name + "() {" + text + "return cudaSuccess; }")
text = body("src/cuda_core/attention/attention_flash.cu", "cudaError_t attention_prefill_flash(")
a = text.index("const dim3 grid(pl.jobs);") + len("const dim3 grid(pl.jobs);")
text = re.sub(r"flash_kernel<\w+><<<.*?\);", "", text[a:text.index("    return cudaGetLastError();", a)], flags=re.S)
text = re.sub(r"smem_bytes<\w+>\(\)", "1", re.sub(r"flash_kernel<\w+>", "3", text)).replace("FlashPipe::swz3", "1")
parts.append("cudaError_t attr_flash(int pipe) {" + text + "return cudaSuccess; }")
(out / "audit_attributes.inc").write_text("\n".join(parts))
text = (root / "apps/hip_qwen35_spec_main.hip").read_text()
a = text.index("    std::size_t bad =")
b = text.index(";", a) + 1
(out / "audit_lengths.inc").write_text("std::size_t incomplete(const std::vector<unsigned>& out, const std::vector<unsigned>& ref, unsigned n_gen) { const unsigned n_run = n_gen;" + text[a:b] + "return bad; }")
