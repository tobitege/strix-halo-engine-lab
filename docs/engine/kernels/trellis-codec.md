# Shared trellis (EXL3) weight codec

`src/core/trellis_format.hpp` and `src/core/trellis_codec.hpp` are the one implementation of the EXL3 trellis
weight format in the engine. GLM, MiniMax, Gemma and the shared tiled core read packed trellis weights through
it. One private reader remains: the Gemma exact many-row route keeps its `Column` reader (own ring addressing) in
`src/gemma4/exl3/device.hip`.

## Format

- A matrix of `I` inputs and `O` outputs is stored as `(I / 16) x (O / 16)` tiles, input-tile major. A tile is a
  ring of `256 * Bits` bits (`8 * Bits` little-endian `u32` words, most-significant bit first).
- State `p` is the 16 ring bits ending at bit `(p + 1) * Bits`. Coefficient `(k, n)` of a tile is state
  `Format::position(k, n)`; its value is the codebook value of that state.
- The mul1 codebook is arithmetic, not a table: `RN16((1024 + bytesum(s * 0x83dcd12d)) * F16(0x1eee) + F16(0xc931))`.
  The product and the sum are exact in FP32, so an FP16 FMA, an FP32 FMA followed by one RN16 and a separate
  FP32 multiply and add followed by RN16 are the same function. The tests enumerate all 65,536 states for every
  form.
- Nothing is carried from tile to tile. Random access is one tile (`32 * Bits` bytes), and any state inside a tile
  is addressable from at most two words. The input and output transforms (sign scales and H128) act on
  activations and results; they are not part of the codec.

## What a kernel gets

`strix::core::trellis::Codec<Bits, Book>` is the operand-source trait:

| Need | Member |
|---|---|
| Block size, bytes | `kTile` (16), `kWords`, `kTileBytes`, `kDecoderStateBytes` (0) |
| Random access | `Format::tile`, `Format::position`, `Format::state` (host and device reference) |
| Row-vector geometries (lane owns states `8t .. 8t + 7`) | `LaneGeom`, `LaneWords::issue` / `issue_at`, `align_words`, `TileStates::state<J>` |
| Matrix-operation geometries (lane owns one output column, 16 inputs) | `Fragment`: `issue`, `normalize`, `state<R>`, `decode16`; tables `Window`, `Norm` |
| Whole ring shared by a wave | `WaveTile` |
| 16-byte windows of a whole or half tile | `LaneTile` |
| Decode | `Book::value` (FP32 reference), `Book::half`, `Book::half16`, `Book::pair`, `PairDecode8`, `ScalarDecode8` |

Every reader is split into an issue step (raw words, no dependence on earlier loads) and a decode step (register
arithmetic only). A tile geometry chooses where to issue: three words per lane for the lane reader, four to seven
for the fragment reader, for as many tiles ahead as it wants.

`TileStates::state<J>()` returns a `State16`: converting it to an integer masks it to 16 bits; `mul1_pair` takes it
unmasked, because the pair decode reads only the low 16 bits of each state.

Layout policy stays with the engine: where the tile row of an input tile starts and how experts are addressed
(for example `moe_core` `ExpertLayout::packed_stride`, or Gemma's payload with its scale vectors in front).

## Adapters

- `src/moe_core/exl3_trellis.hpp`: `Exl3Trellis<Bits>` derives from `Codec<Bits>` and adds the input transform. The
  grouped cores keep using `WeightDecode<Bits>::Window`, `::Norm`, `::codebook_value` and the `decode8` policies.
- `src/core/tiled_core.hpp`: `RawLane`, `Stream` and `decode_pair` are the lane reader behind their former names.
- `src/gemma4/exl3`: the host reference (`format.hpp`) and the device readers and codebooks (`device.hip`) forward to
  the codec. The 64x64 matrix kernel reads its B fragments with `Fragment` when no resident panels exist.

## Tests

- `strix-trellis-codec-tests` (CPU): random tensors for Bits 1..8, single tiles and odd tile counts, decoded
  through the host reference, the codebook forms, the reader tables and frozen transcriptions of the decoders the
  codec replaced; all images are bit-identical and equal to hashes recorded from the replaced implementations.
- `strix-hip-trellis-codec-tests` (device): the whole-tile readers and the mul1 codebook forms against the host
  reference, all 65,536 states per codebook form (the half-tile reader variants and codebooks 0 / 1 are guarded on
  the CPU by frozen transcriptions and recorded image hashes, not executed on the device); `STRIX_TRELLIS_CODEC_BENCH=<repeats>` times each decoder.
- `strix-gemma4-exl3-panels-tests`: the packed matrix route against the panel route.

## Measured

Single node, bitwise in every row (old and new decoders give the same hashes; the engine tests named above pass).

- Gemma E4B, the 64x64 matrix kernel reading packed trellis weights, 512 rows: gate (2560 -> 10240, 4 bits)
  5.5 -> 2.3 ms, down (10240 -> 2560) 6.6 -> 3.0 ms, q (2560 -> 2048, 5 bits) 1.45 -> 0.55 ms, k/v (6 bits)
  0.53 -> 0.21 ms. The kernel went from one byte-wise state extraction and scalar codebook per coefficient to the
  fragment reader with the packed pair decode: 936 -> 717 vector instructions and 120 -> 56 loads per step, no
  quarter-rate multiplies, 124 -> 105 vector registers.
- MiniMax prefill (dense and MoE) and the GLM and MiniMax kernels that only moved onto the shared codec compile to
  the same machine code as before.
- GLM routed-MoE prefill (288 experts, 4 bits, top-8, synthetic): the one-split core decodes with the packed pair
  form instead of a scalar codebook per coefficient. Grouped GPU time 43.0 -> 38.1 ms at 4096 rows, 23.1 -> 21.0 ms
  at 2048, 15.8 -> 13.8 ms at 1024 (gate/up stage 21.9 -> 18.2 ms at 4096); 1224 -> 950 vector instructions per
  128-row kernel body, none at quarter or half rate.
- Lane-reader kernels (GLM and MiniMax decode, GLM exact prefill, attention projections, the shared tiled core):
  unmasked states for the pair decode remove 3 to 4 vector instructions per 8 coefficients (49 -> 45 at 4 bits);
  1 to 5 % fewer vector instructions per kernel, the same loads.
- A decode-only pass is bound by memory (the packed stream is read at the bandwidth floor whatever the decoder), so
  the decoder's cost shows inside the matrix stages above, not in a standalone decode.
