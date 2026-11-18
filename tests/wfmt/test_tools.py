"""Host-only regressions for qualification tooling; all fixtures stay in the checkout."""
import contextlib
import io
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/bench/wfmt"))
import capture
import compare
import isa
import matrix_from_records
import records
import run
import sync_source


def invoke(module, args):
    # Windows platform discovery itself spawns a subprocess; keep it outside adapter mocks.
    with mock.patch.object(sys, "argv", [module.__name__] + list(map(str, args))), \
            mock.patch.object(capture.platform, "platform", return_value="test-host"), \
            contextlib.redirect_stdout(io.StringIO()):
        return module.main()


def matrix():
    return {"schema": "wfmt.matrix@1", "cases": [{
        "file": "decode.wfmt", "case": "decode", "arithmetic": "legacy@1", "shape": [4],
        "required_sections": {"output": 4}, "optional_sections": {"sidecars": 0}}]}


def write_output(directory, payload=b"1234", sections=None):
    directory.mkdir(parents=True, exist_ok=True)
    records.write_record(directory / "decode.wfmt", {"case": "decode", "arithmetic": "legacy@1", "shape": [4]},
                         sections if sections is not None else {"output": payload})


def junit(names):
    root = ET.Element("testsuite")
    for name in names:
        ET.SubElement(root, "testcase", name=name, status="run")
    return root


def device_object(resource=0, descriptor=True):
    names = b"\0kernel.kd\0" if descriptor else b"\0other\0"
    symbol_offset = 128 + len(names)
    section_offset = symbol_offset + 24
    data = bytearray(section_offset + 4 * 64)
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", data, 18, 224)
    struct.pack_into("<Q", data, 40, section_offset)
    struct.pack_into("<HH", data, 58, 64, 4)
    struct.pack_into("<I", data, 64 + 48, resource)
    data[128:symbol_offset] = names
    struct.pack_into("<IBBHQQ", data, symbol_offset, 1, 1, 0, 1, 0, 64)
    for index, section in enumerate(((0,) * 10, (0, 1, 0, 0, 64, 64, 0, 0, 64, 0),
                                     (0, 3, 0, 0, 128, len(names), 0, 0, 1, 0),
                                     (0, 2, 0, 0, symbol_offset, 24, 2, 0, 8, 24))):
        struct.pack_into("<IIQQQQIIQQ", data, section_offset + 64 * index, *section)
    return bytes(data)


NOTES = """amdhsa.kernels:
  - .args: []
    .name: kernel
    .symbol: kernel.kd
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .vgpr_count: 16
    .sgpr_count: 32
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .max_flat_workgroup_size: 256
amdhsa.version: [1, 2]
"""


class ToolTests(unittest.TestCase):
    def archive(self, entries):
        path = self.path / "source.tar"
        with tarfile.open(path, "w") as archive:
            for name, data in entries.items():
                member = tarfile.TarInfo(name)
                member.mode = 0o644
                if isinstance(data, tuple):
                    member.type = tarfile.SYMTYPE
                    member.linkname = data[0]
                    archive.addfile(member)
                else:
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
        return path

    def sync(self, entries, revision="a" * 40):
        invoke(sync_source, ["--archive", self.archive(entries), "--source", self.path / "source",
                             "--revision", revision])

    def test_sync_file_directory_transitions_and_mtimes(self):
        self.sync({"keep": b"same", "p": b"file"})
        root = self.path / "source"
        stamp = (root / "keep").stat().st_mtime_ns
        self.sync({"keep": b"same", "p/deep/x": b"child"})
        self.assertEqual((root / "p/deep/x").read_bytes(), b"child")
        self.sync({"keep": b"same", "p": b"replacement"}, "b" * 40)
        self.assertEqual((root / "p").read_bytes(), b"replacement")
        self.assertEqual((root / "keep").stat().st_mtime_ns, stamp)
        self.assertEqual(json.loads((root / ".wfmt-source-inventory.json").read_text())["revision"], "b" * 40)

    def test_sync_conflict_preflight_and_recovery(self):
        self.sync({"a": b"old", "p/x": b"tracked"})
        root = self.path / "source"
        receipt = (root / ".wfmt-source-inventory.json").read_bytes()
        (root / "p/untracked").write_bytes(b"local")
        with self.assertRaises(ValueError):
            self.sync({"a": b"new", "p": b"file"})
        self.assertEqual((root / "a").read_bytes(), b"old")
        self.assertEqual((root / "p/x").read_bytes(), b"tracked")
        self.assertEqual((root / "p/untracked").read_bytes(), b"local")
        self.assertEqual((root / ".wfmt-source-inventory.json").read_bytes(), receipt)
        (root / "p/untracked").unlink()
        (root / "p/empty-local").mkdir()
        with self.assertRaises(ValueError):
            self.sync({"a": b"new", "p": b"file"})
        (root / "p/empty-local").rmdir()
        self.sync({"a": b"new", "p": b"file"})
        self.assertEqual((root / "p").read_bytes(), b"file")

    def test_sync_untracked_ancestor_and_file_conflicts(self):
        self.sync({"a": b"old"})
        root = self.path / "source"
        (root / "p").write_bytes(b"local")
        for name in ("p/x", "p"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.sync({"a": b"new", name: b"tracked"})
            self.assertEqual((root / "a").read_bytes(), b"old")
            self.assertEqual((root / "p").read_bytes(), b"local")

    def test_sync_retry_after_interrupted_kind_change(self):
        root = self.path / "source"
        original_write = Path.write_bytes
        for before, after, failing in (({"p/x": b"old"}, {"p": b"new"}, "p"),
                                        ({"p": b"old"}, {"p/x": b"new"}, "p/x")):
            self.sync(before)
            receipt = (root / ".wfmt-source-inventory.json").read_bytes()
            def interrupt(path, data):
                if path == root / failing:
                    raise OSError("simulated interrupted extraction")
                return original_write(path, data)
            with mock.patch.object(Path, "write_bytes", interrupt), self.assertRaises(OSError):
                self.sync(after)
            self.assertEqual((root / ".wfmt-source-inventory.json").read_bytes(), receipt)
            self.sync(after)
            self.assertEqual((root / failing).read_bytes(), b"new")

    def test_sync_preserves_untracked_descendants_of_obsolete_directory(self):
        self.sync({"p/x": b"tracked"})
        root = self.path / "source"
        (root / "p/local").write_bytes(b"local")
        self.sync({"new": b"tracked"})
        self.assertFalse((root / "p/x").exists())
        self.assertEqual((root / "p/local").read_bytes(), b"local")

    def test_sync_symlink_transitions(self):
        probe = self.path / "probe"
        try:
            probe.symlink_to("missing")
        except OSError as error:
            self.skipTest("host cannot create symlinks: " + str(error))
        probe.unlink()
        root = self.path / "source"
        for first, second in (({"p": b"file"}, {"p": ("target",)}),
                              ({"p": ("target",)}, {"p/x": b"child"}),
                              ({"p/x": b"child"}, {"p": ("target",)}),
                              ({"p": ("target",)}, {"p": b"file"})):
            self.sync({"target": b"safe", **first})
            self.sync({"target": b"safe", **second})
            self.assertEqual((root / "target").read_bytes(), b"safe")
            self.assertEqual((root / "p").is_symlink(), isinstance(second.get("p"), tuple))

    def test_lane_diagnostic_host_contract(self):
        # A source guard is possible on hosts without HIP; output equivalence
        # still requires running wfmt.hip-dense on a native device.
        source = (ROOT / "tests/wfmt/hip_dense.hip").read_text()
        body = source.split("template<unsigned R> void row_partials()", 1)[1].split("template<unsigned R> void signed_partials()", 1)[0]
        self.assertIn("constexpr unsigned k=512U,chunks=k/256U;", body)
        self.assertIn("Buffer codes(16U*chunks*1024U),scales(chunks*64U*16U),xs(R*k*4U);", body)
        self.assertIn("dim3(chunks*4U)", body)
        self.assertIn("dim3(R*chunks)", body)
        self.assertIn("if(!byte_equal(wide,lane))", body)
        self.assertRegex(body, r'#ifdef STRIX_WFMT_CAPTURE\s+if\(!total\)')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="wfmt-test-", dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        # The adapter is mocked: hash a real fixture, not a possibly redirected
        # or inaccessible interpreter executable (e.g. Microsoft Store Python).
        self.exe = self.path / "adapter.bin"
        self.exe.write_bytes(b"wfmt test adapter\n")

    def test_transfer_adapter_uses_environment_and_preserves_test_arguments(self):
        out = self.path / "transfers"
        contract = self.path / "transfer-matrix.json"
        contract.write_text(json.dumps(matrix()))
        def adapter(command, **kwargs):
            self.assertEqual(command, [str(self.exe), "gsq"])
            self.assertEqual(kwargs["env"]["STRIX_WFMT_CAPTURE_DIR"], str(out.resolve()))
            write_output(out)
            return subprocess.CompletedProcess(command, 0)
        with mock.patch.object(capture.subprocess, "run", side_effect=adapter):
            self.assertEqual(invoke(capture, ["--exe", self.exe, "--matrix", contract,
                "--out", out, "--compiler", "compiler@1", "--architecture", "gfx1151",
                "--adapter", "hip-transfer", "--", "gsq"]), 0)

    def test_archive_revision_is_explicit_and_validated(self):
        out = self.path / "archive"
        contract = self.path / "archive-matrix.json"
        contract.write_text(json.dumps(matrix()))
        revision = "0123456789abcdef" * 2 + "01234567"
        def adapter(command, **kwargs):
            write_output(out)
            return subprocess.CompletedProcess(command, 0)
        with mock.patch.object(capture.subprocess, "run", side_effect=adapter):
            self.assertEqual(invoke(capture, ["--exe", self.exe, "--matrix", contract,
                "--out", out, "--compiler", "compiler@1", "--architecture", "gfx1151",
                "--source-revision", revision]), 0)
        self.assertEqual(json.loads((out / "summary.json").read_text())["source"], revision)
        for invalid in ("HEAD", revision[:9], "z" * 40):
            with self.assertRaises(capture.argparse.ArgumentTypeError):
                capture.source_revision(invalid)

    def test_frozen_transfer_inventory_cannot_be_replaced_or_drift(self):
        old = self.path / "old"
        write_output(old)
        frozen = self.path / "frozen.json"
        self.assertEqual(invoke(matrix_from_records, ["--baseline", old, "--out", frozen]), 0)
        original = frozen.read_bytes()
        self.assertEqual(invoke(matrix_from_records, ["--baseline", old, "--out", frozen]), 1)
        self.assertEqual(frozen.read_bytes(), original)
        new = self.path / "new"
        write_output(new, b"12345")
        with self.assertRaisesRegex(ValueError, "lengths"):
            records.validate_inventory(new, records.read_matrix(frozen))
        damaged = self.path / "damaged"
        damaged.mkdir()
        (damaged / "broken.wfmt").write_bytes(b"WFMT")
        self.assertEqual(invoke(matrix_from_records,
            ["--baseline", damaged, "--out", self.path / "bad.json"]), 1)
        self.assertFalse((self.path / "bad.json").exists())

    def make_capture(self, name, exit_code=0):
        out = self.path / name
        contract = self.path / "input-matrix.json"
        contract.write_text(json.dumps(matrix()))
        def adapter(command, **kwargs):
            write_output(out)
            return subprocess.CompletedProcess(command, exit_code)
        with mock.patch.object(capture.subprocess, "run", side_effect=adapter) as launch:
            code = invoke(capture, ["--exe", self.exe, "--matrix", contract, "--out", out,
                                   "--compiler", "compiler@1", "--architecture", "host"])
        launch.assert_called_once()
        self.assertEqual(code, int(exit_code != 0), (out / "summary.json").read_text())
        return out

    def run_gate(self, old, new, names=None, fixture=False):
        out = self.path / "gate"
        def command(argv, **kwargs):
            if "ctest" in argv:
                ET.ElementTree(junit(names if names is not None else run.required_tests("P01", "host"))).write(
                    argv[argv.index("--output-junit") + 1])
                return subprocess.CompletedProcess(argv, 0)
            return subprocess.CompletedProcess(argv, invoke(compare, argv[2:]))
        args = ["--build", self.path, "--baseline", old, "--candidate", new,
                "--backend", "host", "--pr", "P01", "--output", out]
        if fixture:
            args.append("--protocol-fixtures-only")
        with mock.patch.object(run.subprocess, "run", side_effect=command):
            code = invoke(run, args)
        return code, json.loads((out / "summary.json").read_text())

    def test_mandatory_inventory(self):
        for pr in run.COMMON:
            for backend in run.BACKENDS:
                if pr in run.HIP_REQUIRED and backend != "hip":
                    with self.assertRaisesRegex(ValueError, "requires the HIP backend"):
                        run.required_tests(pr, backend)
                    continue
                required = run.required_tests(pr, backend)
                self.assertEqual(run.validate_tests(junit(required), required), len(required))
                for names in ([], sorted(required)[1:], list(required) + [sorted(required)[0]]):
                    with self.assertRaises(ValueError):
                        run.validate_tests(junit(names), required)
        self.assertIn("nvfp4_host", run.required_tests("P02", "nvfp4"))
        self.assertIn("strix-ds41-formats-tests", run.required_tests("P02", "cuda-ds41"))
        self.assertIn("strix-intel-dense-boundaries", run.required_tests("P05", "intel"))
        self.assertEqual(run.required_tests("P06", "hip"), {
            "wfmt.hip-dense", "strix-hip-dense-a8-wmma-tests", "strix-hip-gsq-dense-a8-tests",
            "strix-hip-gsq-typed-sites-tests", "strix-hip-dense-f16-tiled-tests",
            "strix-hip-ggml-weight-formats-tests"})
        self.assertEqual(run.required_tests("P07", "hip"), {
            "wfmt.hip-gsq", "strix-hip-moe-prefill-gsq-tests", "strix-hip-q20-a8-probe-tests",
            "strix-hip-gsq-dense-a8-tests", "strix-hip-gsq-typed-sites-tests"})
        self.assertEqual(run.required_tests("P08", "hip"), {
            "wfmt.hip-trellis", "strix-trellis-codec-tests", "strix-hip-trellis-codec-tests",
            "strix-ds41-trellis-identity-tests", "strix-glm-trellis-identity-tests",
            "strix-moe-grouped-prefill-host-tests"})
        required = run.required_tests("P01", "host")
        for tag in ("skipped", "failure", "error"):
            root = junit(required)
            ET.SubElement(root[0], tag)
            with self.assertRaises(ValueError):
                run.validate_tests(root, required)
        root = junit(required)
        root[0].set("status", "notrun")
        with self.assertRaises(ValueError):
            run.validate_tests(root, required)

    def test_missing_tests_cannot_pass_runner(self):
        old, new = self.make_capture("old"), self.make_capture("new")
        self.assertEqual(self.run_gate(old, new, ["wfmt.compare"])[0], 1)
        self.assertEqual(self.run_gate(old, new)[0], 0)

    def test_partitioned_tickets_preserve_complete_inventory_and_queue_budget(self):
        old, new = self.make_capture("old"), self.make_capture("new")
        required = run.required_tests("P06", "hip")
        out = self.path / "partitions"
        launched = []
        def command(argv, **kwargs):
            if "ctest" in argv:
                self.assertEqual(argv[:2], ["ticket", "--"])
                self.assertIsNone(kwargs["timeout"])
                self.assertEqual(argv[argv.index("--timeout") + 1], "120")
                expression = argv[argv.index("-R") + 1]
                selected = [name for name in required if expression == "^(" + re.escape(name) + ")$"]
                self.assertEqual(len(selected), 1)
                launched.extend(selected)
                ET.ElementTree(junit(selected)).write(argv[argv.index("--output-junit") + 1])
                return subprocess.CompletedProcess(argv, 0)
            return subprocess.CompletedProcess(argv, invoke(compare, argv[2:]))
        args = ["--build", self.path, "--baseline", old, "--candidate", new,
                "--backend", "hip", "--pr", "P06", "--output", out,
                "--runner", "ticket --", "--partition-tests"]
        with mock.patch.object(run.subprocess, "run", side_effect=command):
            self.assertEqual(invoke(run, args), 0)
        self.assertEqual(set(launched), required)
        self.assertEqual(len(launched), len(required))
        self.assertEqual(run.validate_tests(ET.parse(out / "ctest.xml").getroot(), required), len(required))
        # Queue-aware runners cannot silently fall back to an unbounded group.
        self.assertEqual(invoke(run, args[:-1]), 1)

    def test_zero_payload_rejected(self):
        path = self.path / "empty.wfmt"
        metadata = {"case": "decode", "shape": [4], "arithmetic": "legacy@1"}
        with self.assertRaises(ValueError):
            records.write_record(path, metadata, {"output": b""})
        metadata["sections"] = [{"name": "output", "bytes": 0, "fnv1a64": records.fnv1a64(b"")}]
        header = json.dumps(metadata).encode()
        path.write_bytes(records.MAGIC + struct.pack("<Q", len(header)) + header)
        with self.assertRaises(ValueError):
            records.read_record(path)
        self.assertEqual(invoke(compare, ["--baseline", self.path, "--candidate", self.path, "--require-bytes"]), 1)

    def test_capture_matrix_enforced(self):
        contract = self.path / "matrix.json"
        contract.write_text(json.dumps(matrix()))
        parsed = records.read_matrix(contract)
        out = self.path / "records"
        for sections in ({"output": b"123"}, {"source": b"1234"}, {"output": b"", "source": b"1234"},
                         {"output": b"1234", "logits": b"1"}):
            write_output(out, sections=sections)
            with self.assertRaises(ValueError):
                records.validate_inventory(out, parsed)
        write_output(out, sections={"output": b"1234", "sidecars": b""})
        self.assertEqual(len(records.validate_inventory(out, parsed)), 1)
        (out / "extra.wfmt").write_bytes((out / "decode.wfmt").read_bytes())
        with self.assertRaises(ValueError):
            records.validate_inventory(out, parsed)
        invalid = matrix()
        invalid["cases"][0]["required_sections"]["output"] = 0
        contract.write_text(json.dumps(invalid))
        with self.assertRaises(ValueError):
            records.read_matrix(contract)

    def test_capture_rejects_incomplete_adapter(self):
        contract = self.path / "input.json"
        contract.write_text(json.dumps(matrix()))
        out = self.path / "out"
        def adapter(command, **kwargs):
            write_output(out, payload=b"1")
            return subprocess.CompletedProcess(command, 0)
        with mock.patch.object(capture.subprocess, "run", side_effect=adapter) as launch:
            self.assertEqual(invoke(capture, ["--exe", self.exe, "--matrix", contract, "--out", out,
                                             "--compiler", "c", "--architecture", "host"]), 1)
        launch.assert_called_once()
        self.assertIn("capture lengths differ", json.loads((out / "summary.json").read_text())["error"])

    def test_failed_capture_cannot_be_promoted(self):
        old, new = self.make_capture("old"), self.make_capture("new", exit_code=1)
        self.assertTrue((new / "decode.wfmt").exists())
        self.assertEqual(self.run_gate(old, new)[0], 1)
        code, receipt = self.run_gate(old, new, fixture=True)
        self.assertEqual(code, 0)
        self.assertEqual(receipt["mode"], "protocol-fixtures")

    def test_receipts_bind_environment_inventory_and_contents(self):
        old, new = self.make_capture("old"), self.make_capture("new")
        receipt_path = new / "summary.json"
        original = json.loads(receipt_path.read_text())
        for field in ("compiler", "flags", "architecture", "device", "platform", "adapter_args",
                      "matrix_sha256", "record_sha256", "records", "pass"):
            changed = dict(original)
            changed[field] = (["different"] if field == "adapter_args" else {} if field == "record_sha256"
                              else 99 if field == "records" else False if field == "pass" else "changed")
            receipt_path.write_text(json.dumps(changed))
            with self.subTest(field=field), self.assertRaises(ValueError):
                run.validate_captures(old, new)
        receipt_path.write_text(json.dumps(original))
        write_output(new, payload=b"4321")  # valid FNV, stale whole-record SHA256
        with self.assertRaises(ValueError):
            run.validate_captures(old, new)
        write_output(new)
        (new / "extra.wfmt").write_bytes((new / "decode.wfmt").read_bytes())
        with self.assertRaises(ValueError):
            run.validate_captures(old, new)
        (new / "extra.wfmt").unlink()
        (new / "matrix.json").write_text(json.dumps(matrix(), indent=2))
        with self.assertRaises(ValueError):
            run.validate_captures(old, new)
        receipt_path.unlink()
        self.assertEqual(self.run_gate(old, new)[0], 1)

    def test_hip_metadata_and_descriptor_resources(self):
        original = isa.hip_resources(NOTES, device_object(), ["kernel"])
        nested = NOTES.replace(".args: []", ".args:\n      - .name: argument\n        .size: 8")
        self.assertEqual(original, isa.hip_resources(nested, device_object(), ["kernel"]))
        for key, value in (("group_segment_fixed_size", 0), ("private_segment_fixed_size", 0),
                           ("vgpr_count", 16), ("sgpr_count", 32)):
            changed = NOTES.replace(f".{key}: {value}", f".{key}: {value + 1}")
            self.assertNotEqual(original, isa.hip_resources(changed, device_object(), ["kernel"]))
            with self.assertRaises(ValueError):
                isa.hip_resources(NOTES.replace(f"    .{key}: {value}\n", ""), device_object(), ["kernel"])
        self.assertNotEqual(original, isa.hip_resources(NOTES, device_object(resource=1), ["kernel"]))
        for notes, data in (("", device_object()), (NOTES, device_object(descriptor=False))):
            with self.assertRaises(ValueError):
                isa.hip_resources(notes, data, ["kernel"])

    def test_explicit_symbol_pairs_preserve_instructions_resources_and_coverage(self):
        old_map, new_map = isa.symbol_maps([{"old": "kernel_old", "new": "kernel_new"}])
        old = ["FUNCTION kernel_old", "v_fma_f32 v0, v1, v2, v0",
               "s_branch 4 <kernel_old+0x10>", "FUNCTION unchanged", "s_endpgm"]
        new = ["FUNCTION unchanged", "s_endpgm", "FUNCTION kernel_new",
               "v_fma_f32 v0, v1, v2, v0", "s_branch 4 <kernel_new+0x10>"]
        self.assertEqual(isa.mapped_stream(old, old_map), isa.mapped_stream(new, new_map))
        self.assertNotEqual(isa.mapped_stream(old, old_map),
                            isa.mapped_stream([line.replace("v_fma", "v_mul") for line in new], new_map))
        self.assertNotEqual(isa.mapped_resources({"kernel_old": {"vgpr_count": 16}}, old_map),
                            isa.mapped_resources({"kernel_new": {"vgpr_count": 17}}, new_map))
        with self.assertRaisesRegex(ValueError, "missing"):
            isa.mapped_stream(["FUNCTION other", "s_endpgm"], new_map)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            isa.mapped_stream(old + old, old_map)
        self.assertNotEqual(isa.mapped_stream(old, old_map),
                            isa.mapped_stream(new + ["FUNCTION extra", "s_endpgm"], new_map))
        for pairs in ([], [{"old": "a", "new": "b"}, {"old": "a", "new": "c"}],
                      [{"old": "a", "new": "b"}, {"old": "c", "new": "b"}],
                      [{"old": "a", "new": "b", "ignore": True}]):
            with self.assertRaises(ValueError):
                isa.symbol_maps(pairs)

    def test_isa_runner_rejects_resource_only_changes(self):
        old, new = self.path / "old.co", self.path / "new.co"
        old.write_bytes(device_object())
        new.write_bytes(device_object(resource=1))
        symbols = self.path / "symbols.json"
        symbols.write_text('["kernel"]')
        def output(command, **kwargs):
            return NOTES if "--notes" in command else "0000 <kernel>:\n  v_add_f32 v0, v1, v2\n"
        args = ["--old", old, "--new", new, "--backend", "hip", "--symbols", symbols, "--out", self.path / "isa"]
        with mock.patch.object(isa.subprocess, "check_output", side_effect=output):
            self.assertEqual(invoke(isa, args), 1)
            new.write_bytes(device_object())
            self.assertEqual(invoke(isa, args), 0)

    def test_hip_host_image_extraction(self):
        host = self.path / "host.o"
        host.write_bytes(b"host-object")
        out = self.path / "device"
        def execute(command, **kwargs):
            if command[0] == "bundler":
                Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output="))).write_bytes(device_object())
            return subprocess.CompletedProcess(command, 0)
        with mock.patch.object(isa.subprocess, "run", side_effect=execute), mock.patch.object(
                isa.subprocess, "check_output", return_value="hip-amdgcn-amd-amdhsa--gfx1151\n"):
            image = isa.hip_image(host, out, "gfx1151", "objcopy", "bundler")
            self.assertEqual(image.read_bytes(), device_object())
        with mock.patch.object(isa.subprocess, "run", side_effect=execute), mock.patch.object(
                isa.subprocess, "check_output", return_value="hip-amdgcn-amd-amdhsa--gfx1100\n"):
            with self.assertRaises(ValueError):
                isa.hip_image(host, out, "gfx1151", "objcopy", "bundler")

    def test_binding_storage_support(self):
        bindings = json.loads((ROOT / "tests/wfmt/bindings.json").read_text())["bindings"]
        by_source = {binding["source"]: binding for binding in bindings}
        ds = "src/ds41_formats.cpp"
        decoded = {name.upper() if name != "engram_row" else "Engram"
                   for name in re.findall(r"\bvoid\s+decode_([a-z0-9_]+)\s*\(", (ROOT / ds).read_text())}
        self.assertEqual(set(by_source[ds]["formats"]), decoded)
        cuda = "src/cuda/weight/ggml_sources.cuh"
        source = (ROOT / cuda).read_text()
        geometry = source.split("QQ_HD Geometry geometry(")[1].split("namespace detail")[0]
        decoder = source.split("QQ_HD void decode_group(")[1].split("QQ_HD void decode_row_group(")[0]
        for section in (geometry, decoder):
            self.assertEqual(set(by_source[cuda]["formats"]),
                             {name.upper() for name in re.findall(r"\bcase\s+t_([a-z0-9_]+)\s*:", section)})
        images = {"CB4", "U5", "U6", "Q6P32", "prepared-image", "exl3.mul1.tiles16", "exl3.gemma.transport"}
        for binding in bindings:
            self.assertFalse(set(binding["formats"]) & images)

    def test_existing_byte_negatives(self):
        # compare.self_test uses TemporaryDirectory too; constrain it to this checkout.
        with mock.patch.object(tempfile, "tempdir", str(self.path)):
            compare.self_test()


if __name__ == "__main__":
    unittest.main()
