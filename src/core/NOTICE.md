Shared tiled core and stored INT4 support

The tiled core, its operand/contract interfaces, IU4 pack builder and support
headers are adopted from the project strix-trellis library, policy interface v1.
The small-row W4A16 decoder is adopted from that library and extended here with
streaming stages and shared dequantisation. Apache License 2.0 applies; see
LICENSES/strix-trellis-Apache-2.0.txt. Existing attribution is retained.

The EXL3 spelling identifies the adopted exllamav3 format/quantizer basis.
FAST_I4 is a separate lossy numerical identity. The library's fused EXL3
requantisation operand is not provided; the one convert_exl3_iu4 builds packs.

This directory is the one copy of the shared core in this repository: GLM,
MiniMax and DeepSeek V4.1 include it (namespace strix::core, renamed from the
library's strix_trellis so it never collides with an external strix-trellis
archive linked by the same engine). Local extensions: the integer operand
(integer_operand.hpp, integer_math.hpp, integer_mainloop.inc) for DS41 GGUF
blocks, W4A16 streaming decode and the activation quantiser.

Shared trellis codec

trellis_format.hpp and trellis_codec.hpp are the one EXL3 trellis weight codec
of this repository. They gather, without changing their arithmetic, the ring,
codebook and reader primitives adopted from the strix-trellis library that were
previously kept in trellis_device.hpp, trellis_optimized.hpp and
packed_decoder.hpp here and as copies in src/moe_core/exl3_trellis.hpp and
src/gemma4/exl3/device.hip (whose notices/ directory is retained). Apache
License 2.0 applies as above. The EXL3 format is defined by exllamav3 (MIT
License) and QTIP is described by Tseng et al. (NeurIPS 2024); no code from
either project is included.
