# Serve configuration file (`--serve-config`)

One TOML file can hold the serve options of a node or of a pair rank. The expander is shared
(`src/serve/serve_config.cpp`); each serve CLI declares the options its own parser takes (key, flag, kind) and the
file becomes command-line arguments in front of the real ones. The binary's parser and its single resolver then
see exactly what the equivalent command line gives them, so a file and the equivalent argv produce the same
effective configuration, numerics and tokens (`strix-serve-config-tests` checks this through the HIP parser).

Bound today: the HIP decode / serve CLI (`hip_serve_config_options()` in `apps/hip_decode_options.hpp`) and the
`strix-serve` shell (`--model`, `--registry-root`, `--registry`, `--description`, `--bind`, `--port`). Another
binary binds it with its own table and one `config_file::expand` call before parsing.

Rules:

- Validated before any weight is read: TOML syntax and duplicate keys (line and column), unknown keys, wrong value
  types, conflicting sources, all with `FILE:LINE`. Nothing is accepted and ignored.
- The command line wins: a flag given on the command line drops the file's value; a list given on the command
  line replaces the file's whole list. In the shell, `--` ends shell options; the family tail is passed verbatim
  and does not override or appear in the shell's effective configuration.
- Relative paths (`mtp_bundle`, `release_manifest`, `http.api_key_file`, ...) become absolute paths against the
  config file's directory. Shell `registry_root` and `registry` each resolve against that directory independently;
  an explicit resolved `registry` is not prefixed by `registry_root`. An omitted `registry` retains the default
  `models/registry.toml` under the root; relative command-line `--registry` paths also remain relative to the root.
- Shell `description` is a registry-relative selector, matching the entry's `description` string. It is not
  expanded against the config directory (for example, `description = "models/m.toml"`).
- Secrets: inline credentials (`*key`, `*token`, `*secret`, `*password`) are refused; use the `*_file` option. A
  secret given on the command line is printed as `<redacted>`.
- Untrusted files: on Linux the opened descriptor must name a regular file owned by the effective user or root
  and not writable by group/others. Validation and reading use that same descriptor. On Windows the file ACL
  is the operator's responsibility.
- `--print-serve-config` prints the effective options with their origin (`FILE:LINE` or `command line`) and exits
  before loading. The family's own `--print-effective-config` still prints the fully resolved values.

## Prefill `--config` migration

The older prefill file (`--config FILE`, keys `prefill_exact_fp16`, `prefill_disable`) keeps working. In a serve
config the same settings are `[prefill] exact_fp16` and `[prefill] disable`; setting them in a serve config while
`--config` is also on the command line is refused (no silent collision). Keep one source.

## Example (HIP serve)

```toml
# serve.toml: one node, model path stays the first positional argument
context = 65536

[prefill]
exact_fp16 = true
disable = ["hc_output_wmma"]

[http]
serve = true
port = 8080
bind = "127.0.0.1"
api_key_file = "secrets/api.key"   # relative to this file
max_queue = 64
responses = true
```

```sh
strix-hip-decode-token MODEL.gguf --serve-config serve.toml --print-serve-config   # show, do not load
strix-hip-decode-token MODEL.gguf --serve-config serve.toml --http-port 9000        # the command line wins
```
