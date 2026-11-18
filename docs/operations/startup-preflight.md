# Startup preflight and build identity

Qwen decode-token and HTTP startup emit one `strix-startup-event` JSON line on
stderr at stage `preflight`, before HIP initialization or model mapping. They
also emit one `strix-build-info` line. Both diagnostics default ON. They are
host-only, read-only observations: no option, route, budget, host setting or
model state is derived from them. They do add a few small reads and two stderr
writes before the resolver samples live memory, so exactness is qualified as
identical token IDs within the same effective configuration, not as a claim
about every timing-sensitive path. No throughput improvement is claimed.
`--print-effective-config` inspection runs emit neither line, so their output
stays a single configuration object.

`STRIX_PREFLIGHT=0` disables collection and its report, with no new preflight
file reads. Existing admission checks still run. `STRIX_RELEASE_EVENTS=0` hides
both new stderr records and, as they are the only consumer today, skips
collection too. Configure with `-DSTRIX_EMBED_BUILD_INFO=OFF` to generate an explicit
unavailable build identity, including when reusing a previously enabled build.

## Observation fields

Each numeric observation has a `value` and `status`. Missing values are
`null`; statuses distinguish unavailable, malformed, truncated and
unsupported data. Measured zero remains zero. Sampling begin/end and duration
use a monotonic clock; files are read within a non-atomic interval.

* `mem_total_bytes` and `mem_available_bytes` are checked kernel KiB counters
  converted to bytes. Available memory is not reduced again by RSS or GTT.
* `gtt_total_bytes`, `gtt_used_bytes`, `vram_total_bytes` and `vram_used_bytes`
  describe the selected amdgpu device. Contradictory used/total pairs are
  unavailable. Multiple cards without an explicit DRM mapping leave these
  unknown; `amdgpu_card_count` and `device_mapping` explain the ambiguity.
  Global host, buddy and client observations remain available. A HIP ordinal
  alone is not a DRM mapping. UMA classification must be supplied explicitly;
  otherwise the VRAM interpretation remains unknown.
* `other_kfd_clients_global` counts numeric KFD process directories, excluding
  this process. It is global, never attributed to the selected GPU. Invisible
  clients and kernel allocations remain possible. An absent or unreadable KFD
  directory is unavailable, not an empty process list.
* `buddy_higher_order_blocks` counts blocks at or above the order needed for
  2 MiB using the process page size. `buddy_equivalent_2m_units` weights each
  higher order by its size: an order-10 block on 4 KiB pages contributes two
  units. `buddy_scope` is the supplied allowed-node set or a global upper bound.
  This snapshot excludes future reclaim/compaction and does not prove backing.
* `compact_*_since_boot` are absolute counters. `compact_stall_delta` and
  `compact_interval_ns` are null for a single sample, counter resets, a different
  process or an invalid interval. Only two observations in this process can
  establish a measured increase. There is no sleep or active compaction.
* `memlock_soft_bytes` and `memlock_hard_bytes` are finite byte limits,
  `"infinite"`, or `null`. `pin_requirement` is unchecked without a known
  limit-accounted demand; actual kernel enforcement varies.
* `thp_active` reports always, madvise, never or unavailable. It is informational
  and does not establish huge-page use. `allocator_contract` remains unknown
  in this early hook.

Proc text reads are capped at 64 KiB each; sysfs scalar/mode reads at 128 bytes;
DRM enumeration at 256 entries; KFD enumeration at 4096 entries. A cap hit
invalidates the affected measurement. Trusted production sysfs links work;
test fixture links cannot escape their root. No recursive process inspection,
model reads, device calls or host-setting writes occur in preflight.

## Plan and warnings

The `plan` terms come from the same resolved allocation owner used for admission:
G is incremental peak GTT commitment, H is additional physical host commitment
with UMA overlap removed, R is host reserve, L is limit-accounted locked bytes,
and C is estimated registered/aligned backing demand. Overlapping lifetimes
determine peaks; independent maxima must not be added blindly. The early Qwen
hook has no resolved allocation plan, so all five terms are `null` and its
plan comparisons remain unchecked. The API accepts known terms for callers
with an authoritative plan. No fixed memory or block-count thresholds apply.

| Warning | Meaning |
|---|---|
| `observations-unavailable` | One or more required observations could not be established; a bounded advisory, never startup failure. |
| `gtt-plan-pressure` | Sane total minus used is below known G; `gtt_deficit_bytes` gives the shortfall. |
| `host-plan-pressure` | MemAvailable is below checked H + R; `host_deficit_bytes` gives the shortfall. |
| `firmware-reserve-review` | On an explicitly identified UMA device, reported local/aperture bytes could cover a measured plan deficit if usable. This is conditional, not a promise of recoverable memory or a firmware-setting inference. |
| `contiguous-supply-risk` | Relevant equivalent units are below ceil(C / 2 MiB). This is an estimate, not a requirement that HIP use contiguous 2 MiB pages. |
| `startup-compaction-pressure` | Stalls increased over a measured startup interval and contiguous supply or host reserve is short. High historical counters alone never warn. |
| `pin-limit-risk` | Known L exceeds finite soft memlock. |

Two codes are listed under `advisories`, not `warnings`. They record facts that
are normal on an idle node (the kernel and firmware hold some GTT) and must not
be used as alert or failure triggers on their own. Pressure is only the plan
comparison above: idle use that eats the plan's margin shows up as
`gtt-plan-pressure`.

| Advisory | Meaning |
|---|---|
| `preexisting-gtt-use` | Nonzero GTT use was observed before this process allocates. |
| `allocation-owner-unresolved` | Such use has no visible other client, or the client count is unknown; ownership is not proven. |

Unknown deficits are not evidence of fit. Warning codes never alter admission,
perform recovery, or recommend automatic host changes.

## Configured build evidence

`strix-build` revision 1 records project version, full Git SHA or null, dirty
flag or null, validated target ISA list, host and HIP compiler ID/version, CMake version,
and an explicit allowlist of Boolean STRIX build options. `flags_evidence` is
`configured`; this is not a complete per-source compiler-command receipt or
binary reproducibility claim. Source archives without usable Git metadata have
null source identity unless the packager configures
`-DSTRIX_SOURCE_GIT_SHA=<full 40-hex commit>`; `git_sha_source` then reads
`supplied` instead of `git`. The SHA is captured at configure time, so
reconfigure after changing commits. The generated header and `share/strix/build-info.json`
also work in CPU-only builds. Accessors perform no runtime I/O.

New records serialize only controlled identifiers, enumerations and numbers.
They omit paths, host/account names, PIDs, process lists, command lines, device
serials/BDFs, raw errors and raw environment/configuration. This boundary applies
to these records, not every existing engine log.

`gdn-register-fit.sh BASE_BIN_DIR CANDIDATE_BIN_DIR OUTPUT_DIR` in `scripts/bench/rtx/` compares the shared CUDA GDN fixture digest and three interleaved 8K synthetic prefill pairs, with separate logits and timing logs. Supply `STRIX_DEVICES` and run inside the shared GPU guard and operator FIFO; bound the complete workload, including initialization and idle spacing, to 210 seconds. This tests a scheduling/register change, not model quality or a new profile.
