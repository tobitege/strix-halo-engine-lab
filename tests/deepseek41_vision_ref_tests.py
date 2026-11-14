"""Host test of the resume state of tools/vision/ds41_ref.py: a state is continued only by the computation that
saved it. A small tower in a GGUF file written here; no model file, no network."""
import argparse
import contextlib
import io
import json
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "vision"))
import ds41_ref  # noqa: E402

CFG = {"layers": 3, "width": 8, "heads": 2, "mlp": 6, "down": 1, "eps": 1e-6, "theta": 10000.0}
FEATURES = 12
failures = 0


def check(cond, what):
    global failures
    if not cond:
        failures += 1
        print("FAIL " + what)


def tensors(seed):
    rng = np.random.default_rng(seed)
    w, m = CFG["width"], CFG["mlp"]
    out = {"vision.patch_embed.proj.weight": (w, FEATURES), "vision.patch_embed.proj.bias": (w,), "vision.norm.weight": (w,),
           "aligner.w1.weight": (w, w), "aligner.w1.bias": (w,), "aligner.w2.weight": (w, w), "aligner.w2.bias": (w,),
           "image_start": (w,), "image_newline": (w,), "image_end": (w,)}
    for layer in range(CFG["layers"]):
        p = "vision.blocks.%d." % layer
        out.update({p + "norm1.weight": (w,), p + "attn.wqkv.weight": (3 * w, w), p + "attn.wqkv.bias": (3 * w,),
                    p + "attn.wo.weight": (w, w), p + "attn.wo.bias": (w,), p + "norm2.weight": (w,),
                    p + "mlp.w1.weight": (2 * m, w), p + "mlp.w2.weight": (w, m)})
    return {name: (rng.standard_normal(shape) * 0.3).astype("<f4") for name, shape in out.items()}


def write_gguf(path, data):
    def text(s):
        return struct.pack("<Q", len(s)) + s.encode("utf-8")

    head, body = b"GGUF" + struct.pack("<IQQ", 3, len(data), 0), b""
    for name, a in data.items():
        head += text(name) + struct.pack("<I", a.ndim) + struct.pack("<%dQ" % a.ndim, *reversed(a.shape)) + struct.pack("<IQ", 0, len(body))
        body += a.tobytes() + b"\0" * (-a.nbytes % 32)
    with open(path, "wb") as f:
        f.write(head + b"\0" * (-len(head) % 32) + body)


class Stop(Exception):
    pass


class Interrupted:
    """The tower up to (not including) one layer, then a stop: an interrupted run."""

    def __init__(self, g, layer):
        self.g, self.stop = g, "vision.blocks.%d." % layer

    def digest(self):
        return self.g.digest()

    def f32(self, name):
        if name.startswith(self.stop):
            raise Stop()
        return self.g.f32(name)


def options(**changes):
    a = argparse.Namespace(gate_second=False, rope_adjacent=False, aligner_act="gelu", unshuffle_col_major=False,
                           no_last_newline=False)
    for k, v in changes.items():
        setattr(a, k, v)
    return a


def refused(fn, needle):
    try:
        fn()
    except ValueError as e:
        return needle in str(e)
    return False


def main():
    test_dump_comparison()
    ds41_ref.STATE_SECONDS = 0.0  # a state after every layer
    gh, gw = 2, 3
    patches = np.random.default_rng(7).standard_normal((gh * gw, FEATURES)).astype("<f4")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        model, other_model, state = (os.path.join(folder, n) for n in ("a.gguf", "b.gguf", "case.resume"))
        write_gguf(model, tensors(1))
        write_gguf(other_model, tensors(2))
        g, other = ds41_ref.Gguf(model), ds41_ref.Gguf(other_model)
        check(g.digest() != other.digest() and len(g.digest()) == 64, "file digests")
        args = options()
        want_hidden, want_rows = ds41_ref.encode_one(g, CFG, patches, gh, gw, args)

        def interrupt():
            try:
                ds41_ref.encode_one(Interrupted(g, 2), CFG, patches, gh, gw, args, state)
            except Stop:
                pass
            check(os.path.exists(state + ".npz"), "an interrupted run leaves a state")

        # The same computation continues, bit for bit, and removes its state.
        interrupt()
        hidden, rows = ds41_ref.encode_one(g, CFG, patches, gh, gw, args, state)
        check(np.array_equal(hidden, want_hidden) and np.array_equal(rows, want_rows), "resumed run equals the uninterrupted run")
        check(not os.path.exists(state + ".npz"), "a finished run removes its state")

        # Every other computation of the same shape refuses the state and leaves it in place.
        interrupt()
        moved = patches.copy()
        moved[3, 5] = np.nextafter(moved[3, 5], np.float32(9.0))
        for what, run in (
                ("gate order", lambda: ds41_ref.encode_one(g, CFG, patches, gh, gw, options(gate_second=True), state)),
                ("rotary pairing", lambda: ds41_ref.encode_one(g, CFG, patches, gh, gw, options(rope_adjacent=True), state)),
                ("weights", lambda: ds41_ref.encode_one(other, CFG, patches, gh, gw, args, state)),
                ("patches", lambda: ds41_ref.encode_one(g, CFG, moved, gh, gw, args, state)),
                ("grid", lambda: ds41_ref.encode_one(g, CFG, patches, gw, gh, args, state))):
            check(refused(run, "another computation"), "state refused after a change of " + what)
            check(os.path.exists(state + ".npz"), "refused state kept after a change of " + what)
        # Options applied after the tower do not invalidate the stream.
        tail = options(aligner_act="silu")
        want_tail = ds41_ref.encode_one(g, CFG, patches, gh, gw, tail)
        got_tail = ds41_ref.encode_one(g, CFG, patches, gh, gw, tail, state)
        check(np.array_equal(got_tail[0], want_tail[0]) and np.array_equal(got_tail[1], want_tail[1]), "projector option resumes")

        # Malformed states.
        key = ds41_ref.state_key(g, CFG, patches, gh, gw, args)
        shape, x = (gh * gw, CFG["width"]), np.zeros((gh * gw, CFG["width"]), dtype=np.float32)
        ds41_ref.save_state(state, x, 1, key)
        check(ds41_ref.load_state(state, key, shape, CFG["layers"])[1] == 1, "a valid state loads")
        for layer in (0, CFG["layers"], -1):
            ds41_ref.save_state(state, x, layer, key)
            check(refused(lambda: ds41_ref.load_state(state, key, shape, CFG["layers"]), "layer outside"), "layer %d refused" % layer)
        ds41_ref.save_state(state, x.astype(np.float64), 1, key)
        check(refused(lambda: ds41_ref.load_state(state, key, shape, CFG["layers"]), "float32"), "float64 stream refused")
        bad = x.copy()
        bad[0, 0] = np.nan
        ds41_ref.save_state(state, bad, 1, key)
        check(refused(lambda: ds41_ref.load_state(state, key, shape, CFG["layers"]), "finite"), "NaN stream refused")
        ds41_ref.save_state(state, x[:-1], 1, key)
        check(refused(lambda: ds41_ref.load_state(state, key, shape, CFG["layers"]), "float32"), "wrong shape refused")
        np.savez(state + ".npz", x=x, layer=1)  # the format before the key existed
        check(refused(lambda: ds41_ref.load_state(state, key, shape, CFG["layers"]), "tool version"), "keyless state refused")
        del g, other  # release the file mappings before the folder goes
    print("deepseek41-vision-ref: %s (%d failures)" % ("PASS" if failures == 0 else "FAIL", failures))
    return 0 if failures == 0 else 1


def test_dump_comparison():
    # Handwritten metadata and arrays exercise the reader independently of its writer.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        args = argparse.Namespace(out=folder, a="", b="", strict=True, max_rel_l2=0.0)

        def compare():
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                ds41_ref.cmd_compare(args)
            return output.getvalue()

        for gh, gw, down, last in ((39, 39, 3, False), (5, 7, 2, False), (6, 8, 2, True)):
            mh, mw = -(-gh // down), -(-gw // down)
            span = mh * (mw + 1) + 1 + int(last)
            ds41_ref.write_cases(folder, [["case", gw * 2, gh * 2, gw * 2, gh * 2, gh, gw, span, 0]])
            widths = {"patches": 12, "hidden": 8, "rows": 5120}
            metadata = {"downsample": down, "recipe": {"no_last_newline": not last}, "widths": widths}
            with open(os.path.join(folder, "case.layout.json"), "w") as f:
                json.dump(metadata, f)
            for kind, width in widths.items():
                count = span if kind == "rows" else gh * gw
                np.ones((count, width), dtype="<f4").tofile(os.path.join(folder, "case." + kind + ".f32"))
            check("compared 3, skipped 0, outside threshold 0" in compare(), "recorded span/merge self comparison")

            # A wrong width that still divides the span must be refused, even for a self comparison.
            path = os.path.join(folder, "case.rows.f32")
            np.ones((span, widths["rows"] + 1), dtype="<f4").tofile(path)
            check(refused(compare, "feature width"), "divisible but wrong feature width")
            np.ones((span, widths["rows"]), dtype="<f4").tofile(path)

        args.b = "copy"
        with open(os.path.join(folder, "copy.cases.txt"), "w") as f:
            f.write("case %d %d %d %d %d\n" % (gh, gw, mh, mw, span))
        for kind in widths:
            values = np.fromfile(os.path.join(folder, "case." + kind + ".f32"), dtype="<f4")
            values.tofile(os.path.join(folder, "case.copy." + kind + ".f32"))
        with open(os.path.join(folder, "case.copy.layout.json"), "w") as f:
            json.dump(metadata, f)
        path = os.path.join(folder, "case.copy.rows.f32")
        values = np.ones((span, widths["rows"]), dtype="<f4")
        values[1, 0] = 2.0  # image row: reported separately from learned special rows
        values.tofile(path)
        check(refused(compare, "strict comparison failed"), "strict numerical threshold")
        args.max_rel_l2 = 1.0
        check("special rows max abs diff 0" in compare(), "image/special mask")
        values[0, 0] = 2.0
        values.tofile(path)
        check(refused(compare, "strict comparison failed"), "strict special row equality")
        os.remove(path)
        check(refused(compare, "strict comparison failed"), "strict missing file coverage")
        args.strict = False
        check("compared 2, skipped 1" in compare(), "informational coverage counts")
        args.strict = True
        with open(os.path.join(folder, "copy.cases.txt"), "w") as f:
            f.write("case %d %d %d %d %d\n" % (gh, gw, mh, mw, span - 1))
        check(refused(compare, "strict comparison failed"), "strict layout mismatch")
        ds41_ref.write_cases(folder, [])
        with open(os.path.join(folder, "copy.cases.txt"), "w"):
            pass
        check(refused(compare, "strict comparison failed"), "strict empty comparison")


if __name__ == "__main__":
    sys.exit(main())
