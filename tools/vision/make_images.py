#!/usr/bin/env python3
"""Fixture generator for the shared vision preprocessing tests.

Writes small synthetic images (no photographs) and the reference table the host test compares against:

  tests/vision/fixtures/*.png, *.jpg     decode fixtures (colour modes, EXIF orientations)
  tests/vision/fixtures/*.rgb            the reference loader's RGB8 pixels of each JPEG fixture
  tests/vision/preprocess_reference.inc  resize hashes (reference resampler), smart-resize and any-res tables

Every pixel comes from integer formulas that the C++ test repeats, so no source image is stored for the lossless
cases. Needs Pillow; the output is deterministic for a given Pillow release (the release is recorded in the table).

    python tools/vision/make_images.py [--out tests/vision]
"""
import argparse
import math
import struct
import zlib
from pathlib import Path

import PIL
from PIL import Image, ImageOps

BLOCK_COLOURS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (250, 240, 10)]


def sample(kind, x, y, c, w, h):
    """Pattern value of channel c at (x, y): 0 gradient, 1 noise, 2 four-colour blocks."""
    if kind == 0:
        if c == 0:
            return x * 255 // max(w - 1, 1)
        if c == 1:
            return y * 255 // max(h - 1, 1)
        return (x + y) * 255 // max(w + h - 2, 1)
    if kind == 1:
        v = (x * 73856093 ^ y * 19349663 ^ (c + 1) * 83492791) & 0xFFFFFFFF
        return ((v * 2654435761) & 0xFFFFFFFF) >> 24
    return BLOCK_COLOURS[((x // 5) + (y // 3) * 2) % 4][c]


def pattern(kind, w, h):
    data = bytes(sample(kind, x, y, c, w, h) for y in range(h) for x in range(w) for c in range(3))
    return Image.frombytes("RGB", (w, h), data)


def fnv64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def png_rgb16(path, w, h):
    """RGB PNG with 16 bits per sample: each sample is the noise byte replicated in both bytes."""
    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))
    raw = b""
    for y in range(h):
        raw += b"\0" + b"".join(bytes([sample(1, x, y, c, w, h)] * 2) for x in range(w) for c in range(3))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """The upstream rule (Qwen-VL class image processors), restated."""
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def select_best_resolution(original, pinpoints):
    """The upstream any-res rule (LLaVA-NeXT class image processors), restated; sizes are (width, height)."""
    ow, oh = original
    best, best_eff, best_waste = None, 0, float("inf")
    for w, h in pinpoints:
        scale = min(w / ow, h / oh)
        dw, dh = int(ow * scale), int(oh * scale)
        eff = min(dw * dh, ow * oh)
        waste = w * h - eff
        if eff > best_eff or (eff == best_eff and waste < best_waste):
            best, best_eff, best_waste = (w, h), eff, waste
    return best


def decode_fixtures(fx):
    w, h = 37, 23
    noise = pattern(1, w, h)
    noise.save(fx / "rgb_noise.png")
    rgba = noise.copy()
    rgba.putalpha(Image.frombytes("L", (w, h), bytes(x * 255 // (w - 1) for _ in range(h) for x in range(w))))
    rgba.save(fx / "rgba_noise.png")
    grey = noise.getchannel(0)
    grey.save(fx / "grey_noise.png")
    la = grey.copy()
    la.putalpha(rgba.getchannel(3))
    la.save(fx / "grey_alpha_noise.png")
    pal = Image.new("P", (w, h))
    pal.putpalette([v for colour in BLOCK_COLOURS for v in colour])
    pal.putdata([((x // 5) + (y // 3) * 2) % 4 for y in range(h) for x in range(w)])
    pal.save(fx / "palette_blocks.png", transparency=1)
    assert Image.open(fx / "palette_blocks.png").convert("RGB").tobytes() == pattern(2, w, h).tobytes()
    png_rgb16(fx / "rgb16_noise.png", w, h)
    assert Image.open(fx / "rgb16_noise.png").convert("RGB").tobytes() == noise.tobytes()

    jw, jh = 48, 32
    grad = pattern(0, jw, jh)
    grad.save(fx / "rgb_grad_444.jpg", quality=95, subsampling=0)
    grad.save(fx / "rgb_grad_420_progressive.jpg", quality=90, subsampling=2, progressive=True)
    grad.getchannel(0).save(fx / "grey_grad.jpg", quality=95)
    grad.convert("CMYK").save(fx / "cmyk_grad.jpg", quality=95)
    exif = Image.Exif()
    exif[0x0112] = 6
    grad.transpose(Image.Transpose.ROTATE_90).save(fx / "exif6_grad.jpg", quality=95, subsampling=0, exif=exif)
    for name in ("rgb_grad_444", "rgb_grad_420_progressive", "grey_grad", "cmyk_grad", "exif6_grad"):
        im = ImageOps.exif_transpose(Image.open(fx / f"{name}.jpg")).convert("RGB")
        assert im.size == (jw, jh)
        (fx / f"{name}.rgb").write_bytes(im.tobytes())

    # EXIF orientations 1..8 on a lossless file: the stored image is the inverse transform of the pattern.
    ew, eh = 9, 6
    upright = pattern(1, ew, eh)
    inverse = {1: None, 2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
               4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_90,
               7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_270}
    for k, op in inverse.items():
        stored = upright if op is None else upright.transpose(op)
        exif = Image.Exif()
        exif[0x0112] = k
        stored.save(fx / f"exif{k}_noise.png", exif=exif)
        assert ImageOps.exif_transpose(Image.open(fx / f"exif{k}_noise.png")).tobytes() == upright.tobytes()


RESIZE_CASES = [  # pattern, w, h, out w, out h
    (1, 97, 61, 64, 64), (1, 97, 61, 224, 160), (0, 256, 32, 32, 256), (2, 16, 16, 224, 224),
    (1, 333, 251, 126, 84), (0, 64, 64, 64, 64), (1, 1, 1, 8, 8), (1, 300, 7, 42, 42),
    (1, 64, 48, 64, 21), (1, 64, 48, 23, 48), (2, 200, 150, 14, 14),
]
FILTERS = [("bilinear", Image.Resampling.BILINEAR), ("bicubic", Image.Resampling.BICUBIC),
           ("lanczos", Image.Resampling.LANCZOS)]
SMART_CASES = [  # height, width, factor, min pixels, max pixels
    (32, 32, 32, 3136, 12845056), (32, 32, 32, 65536, 16777216), (4096, 4096, 32, 65536, 16777216),
    (480, 640, 32, 3136, 12845056), (1080, 1920, 32, 3136, 1003520), (17, 4000, 32, 3136, 12845056),
    (5000, 7000, 32, 3136, 12845056), (48, 80, 32, 3136, 12845056), (1, 1, 32, 3136, 12845056),
    (544, 544, 42, 295936, 1806336), (300, 500, 42, 295936, 1806336), (3000, 4000, 42, 295936, 1806336),
    (756, 1008, 42, 295936, 1806336), (100, 2400, 42, 295936, 1806336), (2401, 97, 28, 200704, 1003520),
    (1344, 1344, 42, 295936, 1806336), (63, 63, 42, 295936, 1806336), (720, 1280, 28, 200704, 1003520),
]
PINPOINTS = [(336 * a, 336 * b) for a in range(1, 7) for b in range(1, 7) if a * b <= 6]
BEST_CASES = [(640, 480), (480, 640), (1920, 1080), (300, 300), (4000, 500), (500, 4000), (336, 336), (2016, 336),
              (1000, 1000), (700, 1300)]


def reference_table(path):
    lines = [f"// Generated by tools/vision/make_images.py (Pillow {PIL.__version__}); do not edit.",
             "struct ResizeRef { int pattern, w, h, ow, oh, filter; unsigned long long fnv; };",
             "static const ResizeRef kResizeRef[] = {"]
    for kind, w, h, ow, oh in RESIZE_CASES:
        src = pattern(kind, w, h)
        for index, (_, resample) in enumerate(FILTERS):
            lines.append(f"    {{{kind}, {w}, {h}, {ow}, {oh}, {index}, 0x{fnv64(src.resize((ow, oh), resample).tobytes()):016X}ULL}},")
    lines += ["};", "struct SmartRef { int h, w, factor; double min_pixels, max_pixels; int oh, ow; };",
              "static const SmartRef kSmartRef[] = {"]
    for h, w, factor, lo, hi in SMART_CASES:
        oh, ow = smart_resize(h, w, factor, lo, hi)
        lines.append(f"    {{{h}, {w}, {factor}, {lo}, {hi}, {oh}, {ow}}},")
    lines += ["};", "static const int kPinpoints[][2] = {"]
    lines += [f"    {{{w}, {h}}}," for w, h in PINPOINTS]
    lines += ["};", "struct BestRef { int w, h, ow, oh; };", "static const BestRef kBestRef[] = {"]
    for w, h in BEST_CASES:
        ow, oh = select_best_resolution((w, h), PINPOINTS)
        lines.append(f"    {{{w}, {h}, {ow}, {oh}}},")
    lines.append("};")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="tests/vision", help="output directory (default: tests/vision)")
    out = Path(parser.parse_args().out)
    fixtures = out / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    decode_fixtures(fixtures)
    reference_table(out / "preprocess_reference.inc")
    print(f"wrote {len(list(fixtures.iterdir()))} fixtures and {out / 'preprocess_reference.inc'}")


if __name__ == "__main__":
    main()
