"""Compile the engine adapter against the complete driver with kernel-API/DMA doubles.

USB4_SOURCE_TREE must be an independently materialized nine-patch driver tree.
This does not execute a kernel, GPU instructions or cache-coherence behavior.
"""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CASES = ('group_success', 'group_network', 'group_admission', 'group_failure', 'group_bad_reply', 'reset_wire', 'paired_controller', 'paired_network', 'concurrent', 'reuse', 'geometry', 'partial_events', 'credits', 'ordering',
         'paired_duplex', 'paired_oneway', 'paired_recovery', 'ambiguous_release', 'ambiguous_recovery',
         'failed_dependency', 'recovery', 'stop_retention', 'identity', 'wire',
         'session', 'session_rx', 'session_tx', 'session_recovery', 'session_bad_credit', 'session_bad_gpu',
         *(f'corruption{x}' for x in range(1, 9)))

class LinuxGpuQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'USB4_SOURCE_TREE' not in os.environ:
            raise RuntimeError('Set USB4_SOURCE_TREE to the materialized nine-patch driver tree')
        source = Path(os.environ['USB4_SOURCE_TREE']).resolve()
        if (source / 'include/uapi/linux/strix_gpu_stream.h').read_bytes() != (ROOT / 'include/strix/linux_direct_gpu_stream_uapi.h').read_bytes():
            raise RuntimeError('Engine/driver UAPI drift: byte-for-byte header verification failed')
        cls.tmp = tempfile.TemporaryDirectory(prefix='engine-gpu-queue-')
        cls.addClassCleanup(cls.tmp.cleanup)
        tmp = Path(cls.tmp.name)
        implementation = re.sub(r'^#include(?! \"gpu_group_reset.h\")[^\n]*\n', '', (source / 'drivers/thunderbolt/gpu_stream.c').read_text(), flags=re.MULTILINE)
        unit = tmp / 'driver.c'
        unit.write_text('#include "sgu_test_kernel.h"\n#include "sgu_dependency_status.h"\n#include "gpu_stream_logic.h"\n'
                        + implementation + '\n#include "bridge.c.inc"\n')
        flags = ['-g', '-O1', '-Wall', '-Wextra', '-Werror', '-fno-omit-frame-pointer', '-fsanitize=address,undefined', '-pthread', '-fno-pie', '-no-pie']
        bridge = ROOT / 'tests/gpu_queue_driver_bridge'
        driver_repo = Path(os.environ.get('USB4_DRIVER_REPO', ROOT.parent / 'strix-halo-usb4-driver')).resolve()
        driver_tests = driver_repo / 'tests'
        required = ('sgu_test_kernel.h', 'sgu_queue_cases.h', 'sgu_group_cases.h',
                    'sgu_detached_cases.h', 'sgu_recovery_cases.h', 'sgu_dependency_status.h')
        if any(not (driver_tests / name).is_file() for name in required):
            raise RuntimeError('Set USB4_DRIVER_REPO to the matching driver checkout (canonical tests are reused, not duplicated)')
        subprocess.run([os.environ.get('CC', 'cc'), '-std=gnu11', *(flag for flag in flags if flag != '-no-pie'),
                        '-Wno-unused-function', '-Wno-unused-parameter', '-Wno-unused-const-variable',
                        '-I', str(bridge), '-I', str(driver_tests), '-I', str(source/'drivers/thunderbolt'), '-I', str(source/'include/uapi'),
                        '-c', str(unit), '-o', str(tmp/'driver.o')], check=True)
        cls.binary = tmp / 'queue-tests'
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++20', *flags, '-Wno-unused-result',
                        '-I', str(ROOT/'include'), '-I', str(ROOT/'tests'),
                        str(ROOT/'src/linux_gpu_queue.cpp'), str(ROOT/'src/gpu_queue_session.cpp'), str(ROOT/'tests/linux_gpu_queue_test.cpp'),
                        str(tmp/'driver.o'), '-o', str(cls.binary)], check=True)

    def run_case(self, case, lean=False):
        result = subprocess.run([str(self.binary), case], capture_output=True, text=True, timeout=30,
                                env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:abort_on_error=1',
                                     'STRIX_TEST_GPU_QUEUE_LEAN': '1' if lean else '0'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS ' + case, result.stdout)
        return result.stdout

    def test_counts(self):
        # Identity checks and ioctls around one exchange, both modes (printed for the record).
        print(''.join(line + '\n' for line in self.run_case('counts').splitlines() if line.startswith('COUNT')), end='')

for case in CASES:
    setattr(LinuxGpuQueueTests, 'test_' + case, lambda self, case=case: self.run_case(case))
# The same cases on the lean hot path (identity is per call there by design: covered by 'lean').
for case in (*(c for c in CASES if c != 'identity'), 'lean', 'lean_recheck', 'lean_ioctl_errors'):
    setattr(LinuxGpuQueueTests, 'test_lean_' + case, lambda self, case=case: self.run_case(case, lean=True))

if __name__ == '__main__':
    unittest.main()
