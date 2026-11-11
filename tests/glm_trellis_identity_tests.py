"""Host-only GLM trellis extraction identity; no HIP compiler or device required.

Fingerprints cover the complete original declarations, including template arguments,
from src/glm/moe/glm_trellis_device.hpp at 2261cb42d. Each imported declaration was
byte-identical to moe_core before extraction (LF-normalized, no token rewriting).
This establishes source construction identity, not compiled/device qualification.
Shared primitives that later gained a host emulation are compared through their device
view: `__host__ ` dropped and only the `__HIP_DEVICE_COMPILE__` branch kept.
"""
from hashlib import sha256
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
# Original GLM declarations: (exact declaration prefix, SHA-256).
EXPECTED = {'Half2': ('using Half2 =', 'b6555aa71356cdb36d3994f7842cbe84710742e393bcdff3ca96c05e40f20528'),
 'materialize': ('__device__ __forceinline__ float materialize(',
                 '0b7da74d5a20e370769dcfb7bafd59d11ad44117fadf9920303ec8f44218dd12'),
 'hadamard128': ('__device__ __forceinline__ void hadamard128(',
                 'db4de7814203ea941bcbf882af82e1babc8ad42cfd0629436e369d1df624bc6d'),
 'half_bits': ('__device__ __forceinline__ float half_bits(',
               '400a96d242e35c0e5c5ce571d9a3d961cfcc1b59e1ea3ed5164c16d5c9dc9b26'),
 'transform_block': ('__device__ __forceinline__ void transform_block(',
                     'b80294ea076bb566c207cb3d3cb7eb918c8cc9aade169887f76cff7802f4c567'),
 'transform_block_at': ('__device__ __forceinline__ void transform_block_at(',
                        'ca3ec83be67cae50e55a320ad27be0ed3024855645fc23af2641654ed3ee051e'),
 'transform_block_values': ('__device__ __forceinline__ void transform_block_values(',
                            '9d05fdb97c360160fc2c8c0e98fc4a01987b1695418105a95a26a006b12a3df3'),
 'hash16': ('template <std::uint32_t Multiplier> __device__ __forceinline__ std::uint32_t hash16(',
            'd7d2186bbf360419b55b3641173e4f7f3ac2bde1bb3ad5d3b09b9f91cc8f539c'),
 'mul1_pair': ('__device__ __forceinline__ Half2 mul1_pair(',
               'bab89ad1a2099b6e9ac8d5cda86ec31f71542b00023b3304c7fad1077fc1fbab'),
 'LaneGeom': ('template <int Bits> struct LaneGeom',
              'a09fa525682d8ce3bc2d6445f64e4337d053decda5d0f6adac692cbf48f8ce50'),
 'GlobalU32': ('using GlobalU32 =', '732234177a16c9da5fe7baa7eb767daf261bf72946839196da7828adb63342cf'),
 'LocalU32': ('using LocalU32 =', '339b89324eba027880e85678bdd1015a9f59ac5332548ca92bafe601d67c8a63'),
 'uniform': ('template <class T> __device__ __forceinline__ const T* uniform(',
             '7fc25172fdd4fbbee530fd5024740edb85f3c32d5450ea24b66ad1b4656f7993'),
 'load_u32': ('__device__ __forceinline__ std::uint32_t load_u32(',
              '4e0f0a28553caff0cb72c7af1d334ef2c5d86db61fe9818c74d2f8593cfc0174'),
 'TileStates': ('template <int Bits> struct TileStates',
                '798aa9c356d1e7288aa37fbe191906a705d5375287c124ffb21c26b881458607'),
 'reduce4': ('__device__ __forceinline__ float reduce4(',
             '4ad51702f5a02a032a7c3729ec84949e8640c0366e58d31324ffb7cf861fd6d5'),
 'Slot': ('template <int Bits, int R> struct Slot',
          '04ef312cc30c03ba24a21f800ab460a76af638be01bd2cbd6605bf4044b651a2'),
 'compute_slot': ('template <int Bits, int R>\n__device__ __forceinline__ void compute_slot(',
                  'cd8719287c82d138e57a9c9f63cea026a31c7dcd49ae64708d23ab5bd9989f69'),
 'transform_values': ('__device__ __forceinline__ void transform_values(',
                      '3bd06fa97279278867af1dee16c0283614519949ecd456df0713acd950f4bba7'),
 'vector_splits': ('__host__ __device__ constexpr unsigned vector_splits(',
                   'b1df182d0fd0a443356251c8705c9b4e99a00906a4a2c631aa6017286848dcb2')}


def declaration(source, marker):
    start = source.index(marker)
    assert source.count(marker) == 1, marker
    if marker.startswith("using "):
        return source[start:source.index(";", start) + 1]
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    if source[end:end + 1] == ";":
        end += 1
    return source[start:end]


DEVICE_BRANCH = re.compile(
    r"^#if defined\(__HIP_DEVICE_COMPILE__\)\n(.*?)^#else\n.*?^#endif\n", re.MULTILINE | re.DOTALL)


def device_view(source):
    """Device-compiled text of declarations that also carry a host emulation."""
    source = DEVICE_BRANCH.sub(r"\1", source)
    return source.replace("__host__ __device__ __forceinline__", "__device__ __forceinline__")


def main():
    adapter = (ROOT / "src/glm/moe/glm_trellis_device.hpp").read_text()
    shared = device_view((ROOT / "src/moe_core/exl3_trellis.hpp").read_text())
    imports = re.findall(r"^using strix::moe_core::(\w+);$", adapter, re.MULTILINE)
    assert sorted(imports) == sorted(set(EXPECTED) - {"vector_splits"}), imports
    for name, (marker, expected) in EXPECTED.items():
        source = adapter if name == "vector_splits" else shared
        actual = sha256(declaration(source, marker).encode()).hexdigest()
        assert actual == expected, name + " original GLM declaration changed"
    # Permit only imports and the preserved local split policy in the adapter.
    remainder = adapter.replace(declaration(adapter, EXPECTED["vector_splits"][0]), "")
    remainder = re.sub(r"//[^\n]*", "", remainder)
    remainder = re.sub(r"^using strix::moe_core::\w+;$", "", remainder, flags=re.MULTILINE)
    remainder = remainder.replace("#pragma once", "").replace(
        '#include "../../moe_core/exl3_trellis.hpp"', "")
    remainder = remainder.replace("namespace strix::glm::moe::dev {", "").replace("}", "")
    assert not remainder.strip(), "adapter must not acquire primitive bodies"
    print("PASS: 18 imported primitive declarations are byte-identical to original GLM")
    print("PASS: original GLM split policy and thin-adapter contract")


if __name__ == "__main__":
    main()
