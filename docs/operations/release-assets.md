# Release asset integrity

`strix-model-manifest` provides explicit CPU-only release asset creation and full
verification. The existing `strix-model-manifest MODEL.gguf` tensor-layout output
is unchanged. No serving path invokes the bundle tool, and it changes no engine
options or arithmetic. There is no automatic network access or asset discovery.

```sh
strix-model-manifest bundle-create --spec release/qwen-assets.spec.json --root assets --output assets.json
strix-model-manifest bundle-verify --manifest assets.json --root assets
strix-model-manifest bundle-verify --manifest assets.json --root assets --role vocabulary
```

Verification always reads all bytes of each selected file; it bypasses the hash
helper's metadata cache. There is no partial-hash mode. Repeated `--role` options
select roles; omission verifies all declared roles. A filtered result covers only
the roles listed in its summary. Successful subset checks return exit 0 with
"verification": "partial"; only checks of every required role return "full".
Release admission must require "verification": "full", not merely exit 0.
Selecting all required roles explicitly also returns "full". Every success has
"hash_mode": "full" and sorted "checked_roles" and "unchecked_roles" arrays.
This explicit CLI has no default startup work and needs no runtime off switch: omit the command to omit verification.

## Schema revision 1

The spec and generated manifest use `format: "strix-assets"`, `revision: 1`,
an `engine` (one of `qwen`, `deepseek41`, `minimax`, `glm`), author-declared
`model_key` and `profile_pair` labels, `provenance` and
`licenses` reference arrays, `required` and `files`. Unknown fields and duplicate
JSON object keys are rejected. Each required object member maps a role to its
complete list of logical keys. Its closure must equal the file list, even during
filtered verification. Roles are `target`, `target_shard`, `draft` (distilled MTP
head), `vocabulary`, `tokenizer`, `profile` and `calibration`. Every shard must be
listed separately; this tool does not discover shards from an external index.

Every file declares `role`, `key`, `member` and `source_revision`. Optional
`expected_sha256` binds a published full-file pin; optional integer `bytes` pins
the size. Creation checks supplied pins and adds exact `bytes` and `sha256`.
Verification requires both fields and checks sizes, full hashes and expected
pins. `audit: "gguf"` reads the tensor manifest; `audit: "mtp"` additionally
requires the existing draft tensor-family audit to pass. A matching hash alone
never implies structural compatibility when no audit was requested.

Optional `content_key` requests a separate `content_identity` using the existing
single-model framing. It is checked during verification and never replaces or
aliases the raw file SHA-256.

Provenance and license arrays contain typed objects with exactly "type" and
"value" string fields. Bare strings are rejected. Values are bounded to 512 UTF-8
bytes. Supported reference types are:

- "member": a portable package-relative reference containing a slash, or a
  root notice file named LICENSE, NOTICE, COPYING or README, optionally followed
  by a short `.`/`-` suffix (LICENSE.txt, NOTICE.md, LICENSE-MIT). Ship referenced attribution
  material with the package; references are not automatically fetched or audited.
- "public-id": exactly Qwen-Community-License, Apache-2.0, MIT, BSD-3-Clause or
  CC-BY-4.0. Arbitrary identifiers, email addresses and local host names are rejected.
- "https": an HTTPS source URL with exactly one of these public hosts:
  huggingface.co, github.com, www.apache.org, spdx.org, creativecommons.org.
  A nonempty path is required. ASCII path characters are letters, digits, slash,
  dot, underscore and hyphen; valid non-ASCII Unicode scalars are also allowed.
  Parent traversal segments, credentials, ports, query strings, fragments and
  percent escapes are rejected.

Retain required notices in referenced files; new license identifiers or public
hosts require an explicit schema allowlist update. References must describe
public attribution, never machine-local provenance or secrets. These syntax and
host restrictions do not authenticate a source or classify arbitrary path text
as public. Keep private provenance outside the exported package.

Privacy scope: everything the tool itself derives (verification summaries,
failure reports, error messages) contains only fixed codes, roles, indexes,
digests and allowlisted public names, never filesystem paths or input text.
Labels written into a spec (`model_key`, `profile_pair`, keys, `source_revision`,
member names) are copied into the manifest as the author declared them; the tool
cannot decide whether an arbitrary label is public. Release specs are therefore
checked into the public repository (`release/*.spec.json`) and reviewed there,
and a manifest is published only from such a reviewed spec.

The asset-only strict JSON entry point rejects malformed UTF-8 (including
overlong, surrogate, truncated and out-of-range encodings) and unpaired surrogate
escapes before creation writes a manifest. Serving JSON parsing retains its
existing byte and surrogate behavior.

Canonical UTF-8 JSON sorts object keys and files by `(role,key)`, sorts each
required key list, emits unsigned integer byte counts and contains no insignificant
whitespace or self-digest field. The manifest digest is SHA-256 of these bytes
(excluding a CLI output newline). Each role digest is SHA-256 of compact JSON
`[[key,bytes,sha256],...]` sorted by key. The one-line verification summary contains
`manifest_digest`, selected `roles` with `count` and `digest`, `hash_mode`,
`checked_roles`, `unchecked_roles` and `verification` as described above.
The manifest digest covers the complete declaration, including unchecked roles;
it does not attest that their bytes were checked. Summaries contain no asset
paths. Failure diagnostics use fixed messages and bounded entry identifiers.

Inputs are capped at 1 MiB and 1,024 files. Identifiers are bounded to 128 ASCII
letters, digits, `.`, `_`, `+` and `-`. Members are bounded to 512 characters and
use controlled slash-separated components. Absolute, drive, UNC, traversal,
duplicate members/role-key pairs and symlinks escaping the root are rejected.
Components ending in dot or space and Windows device names CON, PRN, AUX, NUL,
COM1 through COM9 and LPT1 through LPT9 (case-insensitive, with any extension)
are rejected on every platform. Member collisions are checked case-insensitively.
Before hashing, checked entries must resolve to distinct files according to
`std::filesystem::equivalent`, including hard-link and in-root symlink aliases.
Filtered checks only resolve and inspect selected entries; proper subsets remain partial.
There is no glob or environment expansion. SHA-256 strings use 64 lowercase hex
digits. Integer overflow, missing revisions and malformed hashes are errors.
Size and modification time are checked before and after hashing. Keep deployment
assets read-only: these checks detect observed changes, not every adversarial
concurrent filesystem replacement.

## Exit status and limits

- `0`: creation succeeded, or selected assets verified.
- `2`: asset mismatch, missing member, changed file, role closure failure in a
  manifest, or requested structural audit failure. Stdout then carries one JSON
  object with `verification: "failed"` and a fixed `reason`. Entry failures add
  the schema-enum `role` and zero-based `index` in the complete file list sorted
  by (role,key), even under filtering. A raw `key` is emitted only for these
  schema-owned public names: target, target_shard, draft, vocabulary, tokenizer,
  profile, calibration and distill-g. All other keys emit `key_digest` instead:
  the first 16 lowercase hexadecimal characters of SHA-256 of the key's UTF-8
  bytes. This bounded identifier is diagnostic, not an authenticity proof.
  Closure failures have no entry and omit role/index/key/key_digest.
- `1`: invalid spec/schema/CLI, unsafe or aliased member, malformed input or I/O
  failure. Invalid creation inputs do not write a manifest.

The CPU CTest qualification includes `strix-asset-manifest-cli-tests` by default
(`STRIX_ASSET_CLI_TESTS=ON`). Without Python3 it remains registered and fails
with "asset CLI qualification requires Python3". An explicit configure option
`-DSTRIX_ASSET_CLI_TESTS=OFF` omits this gate for general builds; such a build
does not establish asset CLI release qualification. Tests also check release
target/head pins against the bench shell text and the distill-g profile header.

Verification proves consistency with the supplied manifest and pins. It does
not prove publisher authenticity, model quality, numerical qualification or
complete workload correctness. A self-created manifest is not an authenticated
release. Obtain expected pins and source revisions through a trusted channel.
Hashing warms the page cache; account for verification separately from load time.

The Qwen spec lists the bench-kit target, distill-g head and every repository
`data/draft-vocab` file explicitly, including its notice. Vocabulary pins are
computed with this tool. Tokenizer metadata resides in the target GGUF. Source
shards, merge provenance details, calibration/golden assets and deployment profile
plans are not supplied by that bench asset set; add their explicit roles and
trusted pins before claiming a complete serving release. Toolchain pins and
bench-kit schema import/export coordination remain release work.

As described in [reproduction prerequisites](../reproduce.md), the distill-g head is
published at [hyukudan/Qwen3.8-Flash-Next-MTP-distillG-Q8_0](https://huggingface.co/hyukudan/Qwen3.8-Flash-Next-MTP-distillG-Q8_0) and the Qwen fetch recipe downloads it at a pinned revision, verified by its `SHA256SUMS`. Its in-code size/hash pin agrees
with this spec and with the published file. Never substitute the
base MTP asset. No full target/head verification was performed by the tiny CPU
fixture tests; they also establish no model qualification.
