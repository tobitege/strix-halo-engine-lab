"""Compile production host orchestration with fake allocation/stream operations.

Only HIP launch syntax is replaced; kernel arithmetic and native event semantics
are deliberately outside this host fixture's coverage.
"""
from pathlib import Path
import re
import sys

root, out = map(Path, sys.argv[1:3])
def read(path):
    return (root / path).read_text()
source = read("src/hip_deepseek41_prefill.hip")



def function(marker):
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]

# PR720: exercise the actual allocation transaction with fault-injected HIP allocation.
(out / "shared_dev.inc").write_text("template <typename T>\n" + function("struct Dev {") + ";\n")
start = source.index("                        if (m.sh_gate.n <")
end = source.index("                        for (std::uint32_t r = 0; r < no;", start)
(out / "shared_alloc.inc").write_text(source[start:end])


methods = function("bool flash_attn_at(") + "\n"
if "std::uint32_t min_ratio() const" in source:
    methods += function("std::uint32_t min_ratio() const") + "\n"
if "void prepare_flash_mirrors()" in source:
    methods += function("void prepare_flash_mirrors()")
(out / "methods.inc").write_text(methods)
# DAYAUDIT 2: every run installs its own FAST dense / MoE precision before any dispatch
run_from = function("PrefillResult DevicePrefill::run_from(")
head = run_from[:run_from.index("const strix::session::EmbedSplice embeds")]
assert "PrefillPrecisionScope" in head and "m.options.fast_dense, m.options.fast_moe" in head, "run_from must install its own precision first"
(out / "execution_precision.inc").write_text(re.search(r"    const ds41::PrefillPrecisionScope[^;]+;", head)[0])
group = function("std::vector<PrefillResult> DevicePrefill::run_group(")
assert "common = run_from(tokens, 0U);" in group, "group dispatch must use the scoped run_from"
assert "prefill_plan::fingerprint(member_plans)" in group and "n, cls & 0xFFU, rows_fnv" in group, "exchange canonical plans, not local dense ordinals"
assert "static_cast<std::uint32_t>(dense_fingerprint >> 32)" in group, "exchange both fingerprint words"
classes = function("void rebuild_dense_classes()")
assert "if (options.chunk_rows != 0U && rows > options.chunk_rows) continue;" in classes, "unrepresented multi-chunk rows stay solo"
planner = function("prefill_plan::DensePlan dense_plan(")
(out / "planner_fast.inc").write_text("return " + re.search(r"u.fast = ([^;]+);", planner)[1] + ";")
# DAYAUDIT 5: export_state() after run_group (single sequence discarded: rows 0, last row "local") must refuse
(out / "export_state.inc").write_text(function("AttentionState DevicePrefill::export_state() const"))
exp_seq = function("static AttentionState export_sequence(")
assert exp_seq.index("last_local_row && local_rows == 0U") < exp_seq.index("local_rows - 1U"), "export_sequence must guard local_rows - 1"
grp_end = source[source.index("m.group_used = nseg;"):]
assert grp_end.index("m.solo_discarded = true;") < grp_end.index("} else {"), "run_group marks the single sequence discarded"
assert "m.solo_discarded = false;    // the single sequence is the imported one again" in function("void DevicePrefill::import_state("), "import restores it"
(out / "setter.inc").write_text(
    function("void DevicePrefill::set_precision_for_new_sequence(bool") + "\n" +
    function("void DevicePrefill::set_precision_for_new_sequence(const PrefillOptions&"))
start = source.index("                    for (std::uint32_t sg = 0; sg < nseg; ++sg) {", source.index("recv_words = static_cast<const std::uint32_t*>(entries_exchange->finish(st))"))
end = source.index("                    for (std::uint32_t sg = 0; sg < nseg; ++sg) {", start + 1)
unpack = source[start:end]
unpack = re.sub(r"final_entries_kernel<false><<<[^\n]+>>>\(fe\);", "unpack_entries(fe);", unpack)
if "<<<" in unpack:
    raise RuntimeError("unrecognized launch syntax")
# The common block must follow both receive paths and precede lane publication.
assert source.index("entries_exchange->finish(st)") < start
assert source.index("exchange_rows(m.pack_send.p, own_words") < start
assert end < source.index('hipEventRecord(m.chunk_ready, st)', end)
(out / "unpack.inc").write_text(unpack)

source = read("src/hip/weight/ggml_sources.hpp")
(out / "weight_supported.inc").write_text(function("template<class V> static bool supported("))
source = read("src/core/f16_dense_q8a8.hpp")
(out / "parts.inc").write_text(function("inline void enqueue_parts("))
(out / "dense_supported.inc").write_text(function("inline bool supported("))
source = read("src/hip_ds41_layer.hip")
(out / "dense_dispatch.inc").write_text(re.search(r"if \(!v.exact && fast_dense_on\(\) && enqueue_fast_dense\(v, stream\)\) return false;", source)[0])
helper = function("static strix::core::f16_dense_tall::q8a8::View fast_dense_view(")
(out / "dense.inc").write_text(helper + "\n" + function("static bool enqueue_fast_dense(") + "\n" +
                              function("bool q8_0_a8_fast_core_supported(") + "\n" +
                              function("bool enqueue_q8_0_a8_fast_core("))
(out / "dense_validation.inc").write_text(function("static bool valid_q8_0_a8_dense_prefill_view("))
(out / "dense_groups.inc").write_text(function("bool enqueue_q8_0_a8_fast_dense_groups("))
(out / "fast_dense_on.inc").write_text(function("static bool fast_dense_on()"))
source = read("src/hip_ds41_prefill.hip")
(out / "fast_moe.inc").write_text(function("bool fast_moe() {"))
