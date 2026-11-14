# Power and energy receipt

Energy is measured once, in the shared base, and reported by every engine in the same block. The code is
[include/strix/power/sampler.hpp](../../include/strix/power/sampler.hpp) and
[src/power/sampler.cpp](../../src/power/sampler.cpp): a host library (`strix_power`) with no device runtime
dependency.

## What it does

- One worker thread reads the registered sources every 100 ms (`STRIX_POWER_SAMPLE_MS`, 10..10000) into a fixed
  ring (65,536 samples, about 1.8 hours at the default period). A finished request can ask the worker for one
  closing sample and waits for it for at most 10 ms (a stream does not wait at all: its terminal chunk carries
  what the ring holds); when the deadline expires the receipt says its tail is not covered. The worker never
  samples closer together than a tenth of the period (5 ms at most), so the ring holds at least 5 minutes even
  under a flood of completions. Nothing else reads a sensor: a status
  request and a receipt only read the ring.
- It needs no elevated rights and changes no system setting. A source that is missing or unreadable is reported as
  unavailable with the reason and skipped. A run never fails because of the sampler.
- The device gate starts it when a binary binds its device, and the HTTP server starts it at server start, so an
  engine adds nothing to be sampled.
- A receipt is a time window of the ring. The window edges are interpolated between the neighbouring samples, so the
  caller only passes timestamps it already has.
- Nothing is invented. A channel covers the stretch between two consecutive samples when both hold a valid
  reading and they are at most `max(1000 ms, 3 periods)` apart. A failed read is never bridged: the stretches on
  both sides of it are not covered. Neither is a window older than the ring or one that ends after the last
  sample. The receipt then says `complete: false`, reports the energy of the covered parts only and no
  `tok_per_j`. `covered_ms` is the time every measured domain covers at once. A window with no covered part is
  `available: false`.

## Sources

| source | reads | channels |
|---|---|---|
| `amdgpu-hwmon` | `hwmon*/power1_average` (or `power1_input`), `freq1_input`, `temp1_input`, `device/gpu_busy_percent` of every hwmon device named `amdgpu` | power, GPU clock, temperature, GPU load |
| `rapl` | `powercap/intel-rapl:N/energy_uj` of the package zones, with `max_energy_range_uj` for the wrap | package energy counter |
| `cpufreq` | `cpufreq/policy*/scaling_cur_freq`, at most once per second | mean CPU clock |

### Domains and coverage

Every energy counter and power reading belongs to one domain:

| domain | what | built-in sources |
|---|---|---|
| `package` | the CPU or SoC package, integrated GPU included | `rapl`; `amdgpu-hwmon` of an integrated device (its power value is the package power tracking value of the whole SoC, label `PPT`) |
| `accelerator` | discrete cards, outside every package | `amdgpu-hwmon` of a card |
| `system` | the whole machine (a wall or board meter); replaces the other two when present | none built in |

The hwmon source tells the two apart by what the driver exposes: `in1_input` exists on an integrated device only,
a fan (`fan1_input`, `pwm1`) on a card only. With neither marker the device is taken as `package` and the source
detail says `domain=package(assumed)`, so a package counter is never added to it a second time.

Energy is computed per physical coverage group: the counter when it covers at least as much of the window as
power readings of the same hardware, otherwise the power integral. Disjoint groups are added even within one
domain. Sources default to independent groups; `Source::coverage_id(domain)` can identify the exact same hardware
across counter and power sources. The built-in package sources retain their shared host-package assumption.
`coverage` may repeat a domain when it contains several groups;
names what the number describes, for example `package:counter+accelerator:power-integral`; a domain the sampler
measures but that has no reading in the window appears as `accelerator:none` and the receipt is not complete;
that holds from the first read of the domain on, also when every read of it fails.
The energy is the energy of the listed domains. It is the whole machine only when the coverage says `system`:
memory, storage, fans and conversion losses outside the listed domains are not in it.

A counter made of several zones is published only for a sample in which every zone was read; a sample in which
one failed carries no counter, including failures during probing (reported in source detail and retried). A zone
with an invalid or unreadable initial value establishes its baseline on its first valid read without adding energy;
a previously valid zone keeps its own baseline, so its energy is not moved into the stretch in
which it recovers. A value file must hold one complete non-negative decimal number (a temperature may be
negative): a numeric prefix followed by other bytes, a negative power or a counter past its range is a failed read.

### Package energy counter permission

Many kernels expose `energy_uj` to root only (mode 0400). The sampler does not work around that: the `rapl`
source reports `permission denied: <path>` and the receipt uses the power integral. To let the sampler read the
counter, the machine owner grants read access once per boot:

```sh
sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj
```

## The receipt block

Schema `strix-power-receipt-v1`, one object:

```json
{"schema":"strix-power-receipt-v1","enabled":true,"available":true,
 "sources":["amdgpu-hwmon","cpufreq"],
 "source_detail":[{"name":"rapl","available":false,"reason":"permission denied: ..."},
                  {"name":"amdgpu-hwmon","available":true,"detail":"hwmon5: power1_average(PPT) freq1_input temp1_input gpu_busy_percent"},
                  {"name":"cpufreq","available":true,"detail":"32 policies: scaling_cur_freq"}],
 "request":{...},"prefill":{...},"decode":{...}}
```

Each of `request`, `prefill` and `decode` is either `{"available":false,"reason":"..."}` or:

| field | meaning |
|---|---|
| `duration_ms` | length of the window |
| `energy_j` | energy in the window |
| `avg_w`, `peak_w` | mean power over the covered part and the highest power reading (with several hardware groups: the sum of their peaks, an upper bound rather than a simultaneous machine peak) |
| `tokens`, `tok_per_j` | tokens of the phase and tokens per joule (prefill: prompt tokens computed; decode: tokens emitted); `tok_per_j` is 0 unless the receipt is complete |
| `gpu_clock_mhz_median` | median GPU clock over the samples of the window |
| `temp_c_max` | highest temperature sample |
| `cpu_mhz_median`, `gpu_busy_pct_mean` | CPU clock and GPU load |
| `samples` | samples inside the window |
| `energy_source` | `counter`, `power-integral` or `mixed` |
| `complete`, `covered_ms` | whether every measured domain covers the whole window, and the time all of them cover at once |
| `coverage` | the domains in `energy_j` and how each was measured |

The energy of a window is the energy of the covered domains during that window. Requests that overlap in time
each report all of it; the block does not apportion energy between concurrent requests.

## Where it appears

- **Engine CLI** (`strix-hip-decode-token`): member `"power"` of the request receipt on stdout. With one generated
  token the request is one prefill phase (the first token is inside, as in the request timing). With more, the
  phases split at the first emitted token; a route that does not mark its first token reports the request total only.
- **Every binary that binds a device**: one line `strix power: {...}` on stderr at a normal exit with the process
  total (`energy_source` ends in `-lifetime`). It comes from the running totals, not from the ring, so it is right
  for a process that outlives the ring. A process that is killed or aborts prints no line.
- **HTTP server**: `timings.power` of a completion that carries timings, `power` on each entry of
  `GET /api/status` `requests`, and a top-level `power` object in `GET /api/status` (schema
  `strix-power-status-v1`).
  - The receipt of a request is built once, when the request finishes, over its execution span: the queue stamps
    execution begin and end on the stats of the request, `decode` is the last `decode_ms` of the span and `prefill`
    everything before. The response and the status entry print the same text; neither depends on when it is
    rendered or sent. Time inside the span in which the machine ran other requests belongs to the window.
  - The status object carries the sources, the latest stored reading (`latest.age_ms` old; a channel without a
    reading is `null`, never 0) and energy since start. `available` says a sampler runs; `energy_available`
    says it has an energy counter or a power reading (a host that only exposes clocks has the first only).
  - The status object is host telemetry: when the server has an API key, `GET /api/status` includes it only for a
    request that carries the key. The rest of the status is unchanged.

## Switch

`STRIX_POWER_SAMPLER=0` turns the sampler, the receipt members (CLI and server: no `power` member is printed) and
the exit line off. The sampler does not touch
the device, the model or any arithmetic: token ids and logits are identical with it on or off.

## Measured

Qwen, 8,192-token real-text prompt, one node, default exact route, package energy counter not readable (power
integral, 10 Hz, 46 samples per prefill):

| run | tok/s | energy | mean / peak power | tok/J | median GPU clock | max temperature |
|---|---|---|---|---|---|---|
| prefill, sampler on (1) | 1,755.9 | 576.6 J | 123.6 / 150.0 W | 14.21 | 2,514 MHz | 84 C |
| prefill, sampler off (1) | 1,752.9 | - | - | - | - | - |
| prefill, sampler on (2) | 1,744.5 | 580.0 J | 123.5 / 150.7 W | 14.12 | 2,524 MHz | 89 C |
| prefill, sampler off (2) | 1,733.5 | - | - | - | - | - |
| prefill of the decode run | 1,747.8 | 587.8 J | 125.4 / 151.7 W | 13.94 | 2,512 MHz | 92 C |
| decode, 128 tokens, serial greedy | 22.5 | 543.8 J | 95.7 / 150.5 W | 0.235 | 2,873 MHz | 74 C |

The runs alternate on / off in one session. The sampler costs nothing measurable (on 1,750.2 tok/s mean, off
1,743.2; the spread between repeats of one arm is larger than the difference between arms), and the logits file and
the generated tokens are byte-identical in all four prefill runs.

## Adding a device

A new device (a discrete card, an accelerator, another operating system) is one source and one registration; no
caller changes:

```cpp
class MyCard final : public strix::power::Source {
    strix::power::SourceStatus probe() override;        // open what is readable; say why when it is not
    void read(strix::power::Sample& sample) noexcept override;  // sample.add_power(Domain::accelerator, watts); ...
};
strix::power::register_source("my-card", [](const std::string& root) { return std::make_unique<MyCard>(root); });
```

`read()` adds its power (`add_power`) or cumulative energy (`add_energy`) to the sample under its domain, calls
`fail_power` / `fail_energy` when a channel it owns could not be read (the domain is then absent from that sample
instead of partial), takes the maximum for temperature and sets a clock that is still absent. Sources of one
domain add up, so two sources must not report the same hardware. The sampler serializes the calls. Register before the
first device bind. Tests pass a directory that holds a fake `sys/` tree as `root`
([tests/power_sampler_tests.cpp](../../tests/power_sampler_tests.cpp)).
