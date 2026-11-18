#!/usr/bin/env python3
"""Vision content battery rubric: score greedy image answers against facts computed from the pixels.

Colour classes per pixel (HSV of the RGB the upstream processor sees, PIL convert("RGB")): black v < 0.15;
white / grey s < 0.18 (white v > 0.85); else hue: red, orange (brown if v < 0.55), yellow, green, cyan, blue,
purple, pink. A class is PRESENT at >= 5 % of the pixels, ABSENT below 0.5 % (in between: not scored).
Per answer: colour recall (present classes named), false colours (absent classes named), pattern word (from the
generator: tools/vision/make_images.py / tools/vision/ds41_ref.py synthetic images), orientation word (where the generator has a direction).
palette_blocks additionally must name red, green, blue and yellow.
usage: score.py IMGDIR battery-A.json [battery-B.json ...] > table   (needs Pillow)
The arm label of every answer is its "terms" field (battery.py --tag).
"""
import colorsys, json, re, sys
from PIL import Image

WORDS = {"red": ["red", "crimson", "scarlet", "maroon"], "orange": ["orange"], "brown": ["brown", "tan", "beige"],
         "yellow": ["yellow", "gold", "golden"], "green": ["green", "lime", "olive"], "cyan": ["cyan", "turquoise", "teal", "aqua"],
         "blue": ["blue", "navy", "indigo", "azure"], "purple": ["purple", "violet", "lavender", "magenta"],
         "pink": ["pink", "magenta", "fuchsia", "rose"], "white": ["white"], "black": ["black"], "grey": ["grey", "gray", "silver"]}
PATTERN = {"grad": ["gradient", "gradual", "fade", "fading", "transition", "blend", "smooth"],
           "checker": ["checker", "square", "grid", "block", "tile", "rectang", "stripe", "band"],
           "palette": ["checker", "square", "grid", "block", "tile", "rectang", "mosaic"],
           "text": ["text", "line", "writing", "letter", "word", "dash", "stripe", "horizontal"],
           "noise": ["noise", "random", "static", "speckl", "grain", "pixel", "scatter"],
           "rings": ["ring", "circle", "concentric", "circular", "ripple", "wave"]}
ORIENT = {"grad": ["left", "right", "top", "bottom", "corner", "diagonal"], "rings": ["center", "centre", "concentric", "middle"],
          "text": ["horizontal", "row", "across"], "checker": ["alternat", "row", "column", "grid"], "palette": ["alternat", "row", "column", "grid"]}


def kind(name):
    for k in ("grad", "checker", "palette", "text", "noise", "rings"):
        if k in name:
            return k
    return "grad" if "grad" in name else "noise"


def classes(path):
    im = Image.open(path).convert("RGB")
    if im.width * im.height > 65536:  # nearest sampling keeps pixel colours (no blending)
        f = (im.width * im.height / 65536) ** 0.5
        im = im.resize((max(1, int(im.width / f)), max(1, int(im.height / f))), Image.NEAREST)
    n, cnt = 0, {}
    for r, g, b in im.getdata():
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if v < 0.15: c = "black"
        elif s < 0.18: c = "white" if v > 0.85 else "grey"
        else:
            d = h * 360
            c = ("red" if d < 15 or d >= 345 else ("brown" if v < 0.55 else "orange") if d < 40 else "yellow" if d < 70 else
                 "green" if d < 165 else "cyan" if d < 195 else "blue" if d < 255 else "purple" if d < 290 else "pink")
        cnt[c] = cnt.get(c, 0) + 1
        n += 1
    present = sorted(c for c in WORDS if cnt.get(c, 0) >= 0.05 * n)
    absent = sorted(c for c in WORDS if cnt.get(c, 0) < 0.005 * n)
    return present, absent


def named(text, c):
    t = text.lower()
    return any(re.search(r"\b" + w, t) for w in WORDS[c])


def score(text, present, absent, k, name):
    t = text.lower()
    hit = [c for c in present if named(text, c)]
    false = [c for c in absent if named(text, c)]
    pat = any(w in t for w in PATTERN[k])
    ori = any(w in t for w in ORIENT[k]) if k in ORIENT else None
    rule = all(named(text, c) for c in ("red", "green", "blue", "yellow")) if name == "palette_blocks" else None
    return hit, false, pat, ori, rule


def main():
    imgdir, files = sys.argv[1], sys.argv[2:]
    runs, manifests, tags = [], [], []
    for f in files:
        with open(f + ".manifest.json") as stream:
            manifest = json.load(stream)  # legacy results without a denominator cannot be certified
        with open(f) as stream:
            runs.append({r["name"]: r for r in json.load(stream)})
        manifests.append(manifest["requested"])
        tags.append(manifest["terms"])
    names = sorted({item["name"] for manifest in manifests for item in manifest})
    failed = 0
    print("image | pixels present / absent | T | colours hit | false | pattern | orient | palette rule | answer")
    tot = {}
    for name in names:
        present, absent = classes("%s/%s.png" % (imgdir, name))
        for run, tag in zip(runs, tags):
            r = run.get(name, {})
            ok = r.get("status") == 200 and bool(r.get("text")) and not r.get("error")
            failed += not ok
            text = r.get("text", "") if ok else ""
            if not ok:
                print("ERROR %s T%s: missing/failed answer" % (name, tag))
            hit, false, pat, ori, rule = score(text, present, absent, kind(name), name)
            T = tag
            s = tot.setdefault(T, [0, 0, 0, 0, 0, 0])
            s[0] += len(hit); s[1] += len(present); s[2] += len(false); s[3] += pat; s[4] += bool(ori); s[5] += 1
            print("%s | %s / %s | T%s | %d/%d %s | %s | %d | %s | %s | %s" % (
                name, ",".join(present), ",".join(absent), T, len(hit), len(present), ",".join(hit), ",".join(false) or "-",
                pat, "-" if ori is None else int(ori), "-" if rule is None else ("PASS" if rule else "FAIL"), text.replace("\n", " ")[:150]))
    for T, s in sorted(tot.items()):
        print("TOTAL T%s: colours %d/%d, false colours %d, pattern %d/%d, orientation %d" % (T, s[0], s[1], s[2], s[3], s[5], s[4]))

    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
