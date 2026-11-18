#!/usr/bin/env python3
"""CPU-only regression checks for CUDA instruction comparison boundaries."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('sass_compare', Path(__file__).with_name('sass_compare.py'))
sass_compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sass_compare)


def image(symbol=None, instruction='IADD3 R0, R0, 0x1, RZ; /* 0x001 */'):
    header = ('Fatbin ptx code:\narch = sm_120\ncode version = [9,0]\ncompressed\n'
              'ptxasOptions = --fmad false\nFatbin elf code:\narch = sm_120\ncode version = [1,8]\n')
    return header + (f'Function : {symbol}\n/*0000*/ {instruction}\n' if symbol else '')


class SassCompareTests(unittest.TestCase):
    def parse(self, output):
        with patch.object(sass_compare.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=output)):
            return sass_compare.sass('unused', 'unused')

    def test_image_order_and_empty_images_are_not_instructions(self):
        left = image('_Z4leftv')
        right = image('_Z5rightv', 'EXIT; /* 0x002 */')
        self.assertEqual(self.parse(left + right), self.parse(image() + right + image() + left + image()))

    def test_namespace_move_is_normalised(self):
        # A reviewed move is declared explicitly (--symbol-rename OLD=NEW); an undeclared move still differs.
        moved = {'_ZN5model4tileEv': '_ZN4core4tileEv'}
        self.assertEqual(sass_compare.parse_sass(image('_ZN5model4tileEv'), moved),
                         sass_compare.parse_sass(image('_ZN4core4tileEv'), moved))
        self.assertNotEqual(self.parse(image('_ZN5model4tileEv')), self.parse(image('_ZN4core4tileEv')))

    def test_function_count_is_preserved(self):
        self.assertNotEqual(self.parse(image('_Z4leftv')), self.parse(image('_Z4leftv') + image('_Z5rightv')))

    def test_instruction_change_fails_cli_gate(self):
        outputs = [image('_Z4tilev'), image('_Z4tilev', 'IADD3 R0, R0, 0x2, RZ; /* 0x002 */')]
        results = [SimpleNamespace(returncode=0, stdout=s) for s in outputs]
        with patch.object(sass_compare.sys, 'argv', ['sass_compare.py', '--base', 'base', '--new', 'new']), \
             patch.object(sass_compare, 'collect', side_effect=lambda root, _: {'tile.cu.obj': [root]}), \
             patch.object(sass_compare.subprocess, 'run', side_effect=results), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sass_compare.main(), 1)

    def split_layout(self, *, adapter='', instruction=None, declare=True,
                     extra=None, missing=False, dump_error=False, rename=True):
        old = 'CMakeFiles/strix_ds41_cuda.dir/src/cuda/models/ds41/ds41_core_mma.cu.obj'
        core = 'CMakeFiles/strix_ds41_cuda.dir/src/cuda/core/dense/q8_a8_imma.cu.obj'
        original = image('_ZN5model4tileEv')
        moved = image('_ZN4core4tileEv') if instruction is None else image('_ZN4core4tileEv', instruction)
        # Real artifact discovery, mocked cuobjdump only; keep temporary files in the checkout.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]) as tmp:
            base, new = Path(tmp) / 'base', Path(tmp) / 'new'
            outputs = {}
            layouts = {base: {old: original, 'decode.exe': original},
                       new: {core: moved, 'decode.exe': moved}}
            if not missing:
                layouts[new][old] = adapter
            if extra:
                layouts[new].update(extra)
            for root, artifacts in layouts.items():
                for name, output in artifacts.items():
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                    outputs[str(path)] = SimpleNamespace(
                        returncode=int(dump_error and root == new and name == old), stdout=output)
            args = ['sass_compare.py', '--base', str(base), '--new', str(new),
                    '--cuobjdump', 'fixture-cuobjdump',
                    '--symbol-rename', '_ZN5model4tileEv=_ZN4core4tileEv']
            if rename:
                args += ['--rename', old + '=' + core]
            if declare:
                args += ['--new-host-only', old]
            report = io.StringIO()
            with patch.object(sass_compare.sys, 'argv', args), \
                 patch.object(sass_compare.subprocess, 'run', side_effect=lambda argv, **_: outputs[argv[-1]]), \
                 contextlib.redirect_stdout(report):
                result = sass_compare.main()
            return result, report.getvalue()

    def test_split_layout_passes_cli_gate(self):
        result, report = self.split_layout()
        self.assertEqual(result, 0, report)
        self.assertIn('2 equal, 0 different/errors', report)
        self.assertIn('NEW-HOST-ONLY ', report)

    def test_split_layout_rejects_adapter_kernel(self):
        result, report = self.split_layout(adapter=image('_Z7surprisev'))
        self.assertEqual(result, 1, report)
        self.assertIn('declared host-only object has SASS', report)

    def test_split_layout_rejects_changed_instruction(self):
        result, report = self.split_layout(instruction='IADD3 R0, R0, 0x2, RZ; /* 0x002 */')
        self.assertEqual(result, 1, report)
        self.assertIn('DIFF ', report)

    def test_split_layout_requires_declaration(self):
        result, report = self.split_layout(declare=False)
        self.assertEqual(result, 1, report)
        self.assertIn('NEW-ONLY ', report)

    def test_split_layout_rejects_unknown_artifacts(self):
        for output in ('', image('_Z7surprisev')):
            with self.subTest(output=output):
                result, report = self.split_layout(extra={'unknown.cu.obj': output})
                self.assertEqual(result, 1, report)
                self.assertIn('NEW-ONLY unknown.cu.obj', report)

    def test_split_layout_requires_declared_object(self):
        result, report = self.split_layout(missing=True)
        self.assertEqual(result, 1, report)
        self.assertIn('host-only declaration is not an added object', report)

    def test_split_layout_rejects_failed_or_unrecognized_dump(self):
        for kwargs in ({'dump_error': True}, {'adapter': 'unrecognized output'}):
            with self.subTest(kwargs=kwargs):
                result, report = self.split_layout(**kwargs)
                self.assertEqual(result, 1, report)
                self.assertIn('ERROR ', report)

    def test_split_layout_requires_artifact_move(self):
        result, report = self.split_layout(rename=False)
        self.assertEqual(result, 1, report)
        self.assertIn('host-only declaration is not an added object', report)


if __name__ == '__main__':
    unittest.main()
