// src/minimax/engine/engine.hip:2056
{"decode-layout", 'a', "model+slot+position+prefix", "msa_fused_tests.hip:run_case fused vs current; rows == alone hardware gate for 1..8 rows", ""},
// src/minimax/engine/engine.hip:1456
{"msa-dense", 'a', "model+position+prefix; 128-key blocks", "msa_fused_tests.hip:run_case dense v3/v4 and n4-mixed/n8-mixed; prefill_batch_cases", ""},
// src/minimax/attn/index_proj.hip:399
{"index-proj", 'a', "index projection FP32 chain", "index_proj_tests.hip; split/ring/LDS bitwise gates", ""},
// Former opt-in was removed with its split launcher; retain the mismatch until the native split-vs-unsplit cache gate is restored.
{"index-split-retirement", 'c', "former STRIX_MMATTN_INDEX_SPLIT=1 / mf::set_index_split(true); default was OFF; index-score and cache-store order", "UNCOVERED: current msa_fused_tests.hip exercises only the unsplit launcher; restore native split-vs-unsplit full score/cache bytes and graph-replay gate before treating explicit-on behavior as equivalent", "first changed decode index score/cache byte at index_kernel"},
// src/minimax/engine/engine.hip:1372
{"index-fallback", 'a', "index projection FP32 chain", "mm_kernels.hpp dense_t16 contract; engine index fallback contract; index_proj_tests.hip; minimax_index_layout_tests.cpp ownership/admission/refusal (native load-change equality pending)", ""},
// src/minimax/moe/mmx_moe.hip:1128
{"decode-wide", 'a', "EXL3 vector shape+bits+S", "mm_defaults.cpp moe_wide; rows == alone hardware gate; mmx_moe.hip wide-column contract", ""},
// src/minimax/moe/mmx_moe.hip: make_args / launch_experts4_grid
{"decode-moe-interleave", 'a', "EXL3 matrix+fixed split/k/reduction chain; v4 N8 grid", "hip_mmx_moe_bench.hip --diverse --test four-switch output/workspace/routes and rows==alone gate", ""},
// src/minimax/moe/mmx_moe.hip: single_stream_for / singleton3_core
{"decode-moe-singleton", 'a', "EXL3 matrix+fixed split/k/reduction chain; N/U/S+slot occupancy+K3+narrow W4+Depth4", "hip_mmx_moe_bench.hip --diverse --test reader counters+resources; --diverse-fallback --test wide/K5/W8/Depth8+12", ""},
// src/minimax/moe/mmx_moe.hip: launch_experts4_grid / down4_combine_kernel
{"decode-moe-down-split", 'a', "EXL3 matrix+fixed split/k/reduction chain; same partial scratch, ascending split sum, expert fmaf, shared add", "hip_mmx_moe_bench.hip --diverse --test DOWN_SPLIT=0/1/2 output/workspace/routes, repeated calls and rows==alone gate", ""},
// src/minimax/moe/mmx_moe.hip:1195
{"decode-moe-core", 'a', "EXL3 vector shape+bits+S; router row", "hip-mmx-moe-bench exact-form gates; mmx_moe.hpp arithmetic contract; hip_minimax_g4.hip bitwise gate", ""},
// src/minimax/moe/mmx_moe.hip:1071
{"decode-moe-depth", 'a', "fixed dot2/split history", "mmx_moe.hip launch_experts/launch_experts3/launch_experts4; bench bitwise form gates", ""},
// src/minimax/moe/mmx_moe.hip:2404
{"decode-linears", 'a', "EXL3 vector shape+bits+S", "mmx_moe.hip linear schedule; mm_defaults.cpp linear_v11 bitwise gate; ids==greedy is secondary", ""},
// src/minimax/moe/mmx_moe.hip:2206
{"linear-split", 'a', "input+output+vector policy version", "mmx_trellis_device.hpp:vector_splits and declared finish order", ""},
// src/minimax/moe/mmx_moe.hip:2188
{"linear-resources", 'a', "job descriptors+fixed arithmetic family", "plan_linears; enqueue_linears_impl validation; strict source receipt", ""},
// src/minimax/moe/mmx_prefill.hip:785
{"prefill-tile", 'a', "matrix+fixed splits+WMMA chain", "mmx_prefill_core.hpp header contract; mmx_prefill.hip TILE comment", ""},
// src/minimax/moe/mmx_prefill.hip:639
{"prefill-mt", 'a', "matrix+fixed splits+per-row WMMA chain", "mm_defaults.cpp pf_mtfit bitwise gate; core contract", ""},
// src/minimax/moe/mmx_prefill.hip:1122
{"prefill-pipe-pair", 'a', "matrix+fixed splits+per-row WMMA chain; paired bands decode half the B tiles and share them through LDS, same operands", "hip_mmx_moe_bench.hip check_prefill_core8 pipe pair cells (partials/xt/at/output bitwise); --ab pfpair PAIR state/final rows", ""},
// src/minimax/moe/mmx_prefill.hip:1197
{"prefill-tail-stream", 'a', "matrix+fixed splits+per-row WMMA chain; tail classes write disjoint partials, joined before the stage consumer", "hip_mmx_moe_bench.hip check_prefill_core8 tail stream cells (partials/xt/at/output bitwise); --ab pftails PAIR state/final rows", ""},
// src/minimax/moe/mmx_prefill.hip:573
// src/minimax/moe/mmx_prefill.hip:710
{"prefill-pipe", 'a', "matrix+fixed splits+WMMA chain", "mm_defaults.cpp pf_moe_pipe bitwise gate; source core8 comments; nondefault combinations need same-partial gate", ""},
// src/minimax/moe/mmx_prefill.hip: gather_transform_token_kernel
{"prefill-gather-tok", 'a', "per-(pair, matrix) input transform statements; tile-major pair layout", "hip-mmx-moe-bench test_prefill_gather: raw tiled xt/at, partials and output bitwise (T 1..4096, crafted, gufuse 0/1); mm_defaults.cpp pf_gather_tok", ""},
// src/minimax/moe/mmx_prefill.hip: enqueue_prefill_dense; src/minimax/engine/engine.hip: dense_qkv
{"prefill-dense-qkv", 'a', "EXL3 matrix+library gemm_splits+per-row WMMA chain+ordered split finish", "hip-mmx-moe-bench test_prefill_dense == strix_gemm bitwise (q/k/v/o, q+k+v, T 9..4096); bits 3/4/5/6/8 only, else the library call; mm_defaults.cpp pf_dense_qkv", ""},
// src/minimax/engine/engine.hip: enqueue_post (STRIX_MM_PF_ROUTER_ROWS=32 -> enqueue_router_tiled)
{"prefill-router-tiled", 'a', "router lane chain (j = lane mod 32 ascending) + xor butterfly + ordered top-4 scan", "hip-minimax-moe-tests router_batch_tests: tiled 32 vs fused/4/8 bytes for every kind and tail; mm_defaults.cpp pf_router_rows (default 32 = tiled v2)", ""},
// src/minimax/engine/engine.hip: dense_o (STRIX_MM_PF_DENSE_O) - o_proj through enqueue_prefill_dense on full pieces
{"prefill-dense-o", 'a', "EXL3 matrix+library gemm_splits+per-row WMMA chain+ordered split finish", "hip-mmx-moe-bench test_prefill_dense o == strix_gemm bitwise (T 9..4096); 4096-row pieces only; mm_defaults.cpp pf_dense_o", ""},
// src/minimax/engine/engine.hip: MSA scratch cap (STRIX_MM_PF_MSA_SCRATCH_MB) - windows per batch only
{"prefill-msa-scratch", 'a', "independent MSA windows; batch size only", "msa_fused_tests.hip every-batch bytes (scratch=1/125 and full); mm_defaults.cpp pf_msa_scratch_mb", ""},
// src/minimax/attn/msa_fused.hip: enqueue_prefill_index (STRIX_IDX_WMMA, default exact), prefill_index_refine_kernel
{"prefill-index-wmma-exact", 'a', "fixed row+index head; selected block set of the VALU index kernels (WMMA scores, exact rescoring of the cut-off band)", "hip_idx_wmma_tests.hip exact selection at 2K/8K/32K positions; pair state and final-row hashes equal to STRIX_IDX_WMMA=0", ""},
// src/minimax/moe/mmx_prefill.hip:647
{"prefill-fuse", 'a', "single-split WMMA+epilogue contract", "prefill gufuse intermediate/producer debug gate in hip-mmx-moe-bench; default OFF; combinations outside gate remain unqualified", ""},
// src/minimax/moe/mmx_prefill.hip:416
{"prefill-table", 'a', "identical item table bytes", "prefill table ring event ownership; A/B pftab gate in rank app (1K/8K state identical); default ON in strix-minimax-rank only (serve off)", ""},
// src/minimax/attn/msa_fused.hip: attend4_kernel INBLOCK; engine and rank A/B opt-in selectors
{"prefill-msa-inblock", 'a', "fixed row+KV head+selected chunk IDs; mode-8 tile arithmetic and ascending FP32 chunk combine", "UNCOVERED native mode 10: msa_fused_tests.hip complete output/cache/selection bytes, graph replay, untouched partial scratch; rank 8K mode8/10 state+final-row FNV on both ranks; strict compiler/resource and full-workload gates pending", ""},
// src/minimax/attn/msa_fused.hip:1714
{"msa-attend", 'a', "MSA block128/chunk/FP32 combine contract", "msa_fused_tests.hip:run_case versions/head counts/graph replay", ""},
// src/minimax/engine/engine.hip: route::attend_n2_sparse; src/minimax/attn/msa_fused.hip: launch_attend
{"decode-attend-grid", 'a', "MSA block128/chunk/FP32 combine contract; seqs+rows_per_seq+dense+min position", "msa_fused_tests.hip:run_case v3 lds_diet and v4 heads_per_wave 1/2/4 bitwise (engine: v4 hpw1, else diet); bench_cell arms (STRIX_MMATTN_BENCH_DIET)", ""},
// src/minimax/engine/engine.hip: route::select_once; src/minimax/attn/msa_fused.hip: select_kernel / enqueue_decode
{"decode-select-once", 'a', "MSA top-k contract (order keys, choose_topk, ascending compaction, newest last); rows+dense", "msa_fused_tests.hip:run_case select_once with v3, lds_diet and v4 hpw1/2 bitwise (attention, caches, scores, selection); bench_cell arm (STRIX_MMATTN_BENCH_SELONCE)", ""},
// src/minimax/attn/msa_fused.hip:1693
{"msa-grid", 'a', "MSA query+block identity", "msa_fused_tests.hip fused grid3; index tests; same_out score bytes", ""},
// src/minimax/attn/msa_fused.hip:1934
{"msa-scratch", 'a', "fixed unit rows+absolute positions+MSA contract", "msa_fused_tests.hip:prefill_batch_cases and every-batch producer/selection/output bytes", ""},
// src/minimax/engine/engine.hip:1727
{"prefill-schedule", 'a', "immutable ordered chunk list (max_rows fixed)", "serve --resume-check with layer budgets 1/7/60 bitwise gate; engine state+final rows contract; no max_rows equivalence claim", ""},
// src/minimax/engine/engine.hip:1087
{"post-path", 'a', "rank order0,1+post arithmetic", "mm_defaults.cpp xbound bitwise gate; rank xb_check N/verify; mm_kernels post contract", ""},
// src/minimax/engine/engine.hip:1066
{"router-batch", 'a', "router row FP32 chain+expert tie order", "mm_defaults.cpp pf_router_batch bitwise gate; enqueue_router_rows byte contract", ""},
// src/minimax/engine/engine.hip:1670
{"head-path", 'a', "vocab partition+GEMV+lower-id tie", "xfer selftest/argmax_exchange; rows == alone hardware gate; hip-minimax-g4", ""},
// src/minimax/engine/engine.hip:214
{"workspace", 'a', "descriptor+required capacity", "engine Lib::mm; DraftLinears::lib_mm; enqueue_prefill_moe workspace guard", ""},
// src/minimax/engine/engine.hip:579
{"staging", 'a', "same per-row operations and rank order", "hip_minimax_g4.hip bitwise gate; shared-first source sorted expert loop", ""},
// src/minimax/xfer/minimax_xfer.hip:161
{"transport", 'a', "wire tags+rank order+payload bytes", "xfer selftest (xfer-report); mm_defaults.cpp transport hp exact; pfx CRC/tag validation", ""},
// src/minimax/rtxpf/fp32_identity.hpp:73
{"rtx-tile", 'a', "F32 identity version+matrix+input order", "fp32_identity_device.inc f32_core_body; hip-mm-fp32id-tests; mm_fp32_identity_tests.cu", ""},
// src/minimax/rtxpf/fp32_identity.hpp: resolve_core_version; cuda/fp32_identity.cu
{"rtx-core", 'a', "FP32 identity+matrix+input order; v3/v3.1 decode", "mm_fp32_identity_tests.cu default/env/explicit and all 65536 mul1 states vs host; full-model gate at 1K/8K code prompts, both ranks", ""},
// src/minimax/rtxpf/cuda/mm_rtx_prefill_main.cu:635
{"io-cache", 'a', "model tensor content+dtype+rank slices", "weight loader sizes/format validation; actual dependency notices preserved", ""},
// src/minimax/dspark/hip_minimax_dspark.hip:755
{"draft-native-linears", 'a', "drafter model dtype+matrix+reduction contract", "hip_minimax_dspark_tests.hip linears/attention bitwise gates; mm_defaults.cpp draft linears contract", ""},
// src/minimax/dspark/markov_cand.hip:26
{"markov-topc", 'a', "vocabulary+C+candidate score contract", "hip_markov_cand_tests.hip payload v1/v2 vs CPU; mm_defaults.cpp markov_topc_v2", ""},
// src/minimax/engine/engine.hip:436
{"operation-plan", 'b', "model+operation kind+arithmetic profile+ordered units", "apps/minimax_rank_main.cpp teacher-force comparison explicitly rounding not bitwise; not a gate between identities", ""},
// src/minimax/attn/msa_fused.hip:1780
{"context-plan", 'b', "model+absolute position+prefix tokens+MSA block/chunk contract", "MSA host/reference bitwise tests at boundary and long context", ""},
// src/minimax/dspark/hip_minimax_dspark.hip:990
{"draft-context", 'b', "draft model+absolute position+context window+chunk64", "mm_defaults.cpp dspark_attn bitwise gate at positions 300/1000/4095; no claim all profiles match", ""},
// src/minimax/engine/draft_lin.hip:245
{"draft-model", 'b', "target model hash+draft asset hashes+dtype+head index", "DSpark reference and target acceptance verification; explicit --dspark-exl3/--dspark-head", ""},
// apps/minimax_rank_main.cpp:488
{"draft-markov", 'c', "BLOCKED draft-history identity until request fixes method+C+H+vocabulary+seed", "UNCOVERED candidate/full arithmetic-history equality; target greedy verification is a separate gate", "Markov candidate pruning before rank-dot correction/argmax (full vocabulary vs C/H subset)"},
// src/minimax/serve/minimax_lead.hpp: admit / effective request policy
{"draft-policy", 'a', "target prefix+per-row vector arithmetic", "speculative tokens == greedy tested chats; rows == alone hardware gate for 1..8 rows; not universal workload qualification", ""},
// apps/minimax_rank_main.cpp:292
{"import-plan", 'b', "target model+arithmetic profile+original chunk list+position/cut+payload digest", "rtxpf_state_compare.py / rank import comparisons; UNCOVERED cross-profile handoff, explicit blocked gate", ""},
// src/minimax/engine/engine.hip:569
{"model-layout", 'b', "model tensor hashes+TP2 partition+rank order", "G3/G4 model gates; pack validation; F32 model fidelity remains draft pending all gates", ""},
// src/minimax/moe/mmx_prefill.hip:476
{"prefill-routed-split", 'c', "BLOCKED until matrix+arithmetic-version fixed S (no expert occupancy)", "UNCOVERED: known failed identity; existing core/mtfit receipts hold S fixed", "L3 routed gate/up WMMA k-tile24 reset vs continuation (rows1->9,S16->8)"},
// src/minimax/moe/mmx_prefill.hip:477
{"prefill-shared-split", 'c', "BLOCKED until shared matrix+arithmetic-version fixed S", "UNCOVERED: known failed max_rows identity", "L3 shared gate/up WMMA k-tile24 reset vs continuation (rows1->9,S16->8)"},
// src/minimax/engine/engine.hip:227
{"library-prefill", 'c', "BLOCKED until library exposes frozen full matrix arithmetic tuple", "UNCOVERED across row counts; bitwise gate compares fixed row count only", "L0 q_proj strix_gemm core (first internal divergent FP op unavailable without dependency receipt)"},
// src/minimax/engine/engine.hip:1734
{"prefill-unit-cuts", 'c', "BLOCKED until original request prompt+absolute chunk list+profile immutable", "UNCOVERED across max_rows/cache-cut boundaries; server deliberately has no prefix cache", "L0 q_proj GEMM after first changed unit cut (before L3 MoE)"},
// src/minimax/engine/engine.hip:658
{"arithmetic-profile", 'c', "BLOCKED until request stores target+draft arithmetic profile and rejects mismatch", "UNCOVERED profile equality; FP32 path is an opt-in distinct draft identity", "L0 q_proj input transform RN16 materialization vs FP32 transform; draft unchunked PV vs chunk partials"},
// src/minimax/dspark/hip_minimax_dspark.hip:949
{"draft-append", 'c', "BLOCKED until draft append absolute-position plan fixed or pin vector per row", "UNCOVERED append32 vs8 intermediate/cache byte identity", "draft append FC GEMM32 vs GEMV8, then K/V projected context (first library core op opaque)"},
// src/minimax/moe/mmx_prefill.hip:881
{"diagnostic-cores", 'c', "BLOCKED in production; diagnostic-only plan", "UNCOVERED for production by design; expected-wrong-output timing APIs", "first skipped A/weight load or attention compute in diagnostic producer"},
// src/minimax/rtxpf/cuda/mm_rtx_kernels.cu:601
{"rtx-attention", 'c', "BLOCKED handoff until MSA cache/output bytes match exact producer plan", "UNCOVERED full native/CUDA MSA model fidelity for all positions/cuts", "RTX MSA index score or attention QK/PV legacy vs shared (first op needs CUDA byte gate)"},
// src/minimax/engine/engine.hip: park_task / resume_parked; src/minimax/serve/minimax_lead.hpp: nested_candidate
{"prefill-nest", 'a', "model+profile+original ordered chunk/group rows+positions+slot generation; sequential bank ownership", "GPU covered on both ranks, check-only resume-check: 512-token child in 7-layer units with 4608-token parent parked at layers 7/31 beside a decoder, 10/10 IDENTICAL; whole 511-token child with parent parked at layer 17, 6/6 IDENTICAL; complete 128-token continuations, all prompt outputs, committed state, first/continuation logits, frozen groups and standalone B1/mixed decode; HTTP token IDs identical policy ON/OFF; timing evidence in serving documentation", ""},
// Grouped communicator prefill: scheduling only; unchanged numerical and wire plan.
{"prefill-async-start", 'a', "original group rows+producer/post kernels+rank sum order+wire sequence/tags/chunks/CRC", "Measured two-node TP2 IDs identical in every arm; UNCOVERED full cross-binary prompt-output/state/logit and complete-workload gates pending; hip-dist-hostpath-tests covers API lifetimes only", ""},
{"prefill-late-src-fence", 'a', "same main-stream compute/scratch order+per-member source and destination lifetime", "Measured combined balanced 4096+3844 lone-prefill -2.3/-2.5 %, 4096+443 neutral and gated off; UNCOVERED full pause/nest/cancel/bank-reuse exactness pending; hip-dist-hostpath-tests covers immediate/late fences", ""},
// Per-unit deterministic row admission; ordinary start/event schedule on rejection.
{"prefill-async-min-rows", 'a', "all ordered member rows >= STRIX_MM_PF_ASYNC_MIN_ROWS (2048 default, 0 unconditional); global async off wins", "CPU minimax-defaults-tests covers boundaries, member order, zero and global off; measured TP2 token IDs identical; UNCOVERED full prompt-output/state/logit and complete-workload lifetime gates pending", ""},
// src/minimax/dspark/minimax_auto.hpp; apps/minimax_rank_main.cpp; src/minimax/serve/minimax_lead.hpp
{"draft-auto", 'a', "target identity+unchanged OFF prefill plan+greedy verify; frozen structural-v1 or unqualified-v1", "CPU features/admission and Lead traces: prose == OFF, structural C1 uses legacy controller, C2 retires; GPU tokens/logits/committed-state and paired json/code/long/story/prose-es receipts pending", ""},
// src/minimax/session/runtime.cpp; src/minimax/serve/minimax_lead.hpp
{"session-producer", 'b', "model+numerical environment+geometry+single-row suffix+fixed-width-one verification", "session_cache_gate.sh A/L/B IDs, every FP32 row and complete-state digest on both ranks; UNCOVERED native and complete-workload qualification", ""},
// serve defaults to direct transport, which the session resolver refuses pending its own route qualification.
{"session-direct-transport", 'b', "Cli{app=serve,transport=usb4gpu}; negotiated transport is the construction input", "route_identity_tests.cpp session_direct_transport_boundary; session_cache_gate.sh A/L/B currently covers usb4-auto only; UNCOVERED direct-route native state and complete-workload gate", ""},
// src/minimax/session/session_cache.cpp; include/strix/file_snapshot.hpp
{"session-restore", 'b', "loader-fd asset identities+executable+runtime+schema+frontier+rank-group digest", "minimax_session_cache_tests.cpp envelope/transaction/retarget negatives; session_cache_gate.sh fresh restore and never-saved control; UNCOVERED GPU state equality", ""},
// src/minimax/moe/mmx_moe.hip: moe_grid_slots / launch_experts4_grid
// src/minimax/moe/mmx_moe.hip: moe_pipe4_kernel / launch_experts4_grid (calls of 2-4 rows)
{"decode-moe-pipe", 'a', "routed gate/up + down one launch; down waits per-block gate/up readiness (epoch) instead of the boundary; same partials, counters, finish, combine", "rank --xb-check cfg f vs -: decode logits bitwise N1/N2/N4 + verify window; --ab moepipe matched replay; hip_mmx_moe_bench --test (incl. epoch wrap) and --diverse --test; mm_defaults.cpp moe_pipe", ""},
{"decode-moe-tight-grid", 'a', "v4 routed grid.y = min(32, 4N)+1; map_slot rejects y > count before work/counters", "rank --xb-check cfg g vs -: decode logits bitwise N1/N4/N8 + verify window; mm_defaults.cpp moe_tight_grid", ""},
// src/minimax/engine/engine.hip: xb_warm_o
{"decode-warm-o", 'a', "cache state only (side-stream reads, sink store); no consumer of its data", "rank --xb-check cfg o vs -: decode logits bitwise N1/N4/N8 + verify window; --ab xbwarmo matched replay; mm_defaults.cpp xb_warm_o", ""},
// src/minimax/serve/minimax_lead.hpp: lookup_start / lookup_commit / cycle prompt lookup
{"serve-prompt-lookup", 'a', "draft source only (rank 0 chooses window rows from strix_lookup); greedy verify decides every token", "minimax_lead_tests prompt lookup emits the tokens of lookup off; serve A/B ids_fnv per workload with STRIX_MM_LOOKUP=0", ""},
// src/minimax/engine/engine.hip: xb_fold; src/xport/row_fold.hpp. kFoldRows = 0 (minimax_traits.hpp): the exchange has no
// fold slots unless STRIX_MMX_FOLD=rows is set, so the default route is unchanged.
{"decode-xport-row-fold", 'a', "STRIX_MMX_FOLD=rows fold slots on the exchange module (default 0 = off)+rows <= slots; the pair sum p0 + p1 moves from the consumer kernel to the channel's progress thread, one IEEE binary32 addition per element", "row_fold.hpp sum_f32 contract (one addition, no contraction); rank --ab xbfold matched replay; UNCOVERED native decode-logit receipt in this census", ""},
// src/minimax/engine/engine.hip: pf_msa_flash / pf_msa_flash_layers; src/minimax/attn/msa_flash_policy.hpp
{"prefill-msa-flash", 'b', "STRIX_MM_PF_MSA_FLASH=1|2 (default 0 = off)+STRIX_MM_PF_MSA_FLASH_LAYERS; prefill MSA windows > 8 rows on the BF16 WMMA key-split form", "FAST, lossy, opt-in: a different identity from the exact prefill, never selected by default; msa_fused_tests.hip STRIX_MSA_TEST_FLASH arm; UNCOVERED engine-level quality gate", ""},
// src/minimax/engine/engine.hip: pf_moe_group (strix/pair/prefill_group.hpp moe_width)
{"prefill-moe-group", 'a', "STRIX_MM_PF_MOE_GROUP (0 = automatic, 1 = one call per chunk)+group members+start-up buffer cap; chunks per group-wide EXL3 MoE call", "per-row arithmetic independent of the call width (same kernels, row-local chains); UNCOVERED native width-vs-width prefill state receipt in this census", ""},
// src/core/moe_i4_chain.hpp launch_chain; src/minimax/moe/mmx_prefill_iu4.inc. The resolved STRIX_MOE_I4_CHAIN_GSAVE
// boolean is read once per call and passed to the shared plan policy.
{"prefill-i4-chain-plan", 'a', "requested IU4 plan+STRIX_MOE_I4_CHAIN_GSAVE+scratch admission; a scratch-form paired launch the device rejects (spill or residency) retries the register-save form on plan 1, then a second tile list for the down", "moe_i4_chain_host_tests.cpp plan policy (admission, insufficient and zero scratch); every admitted form computes the same integers; UNCOVERED native MiniMax fallback execution", ""},
