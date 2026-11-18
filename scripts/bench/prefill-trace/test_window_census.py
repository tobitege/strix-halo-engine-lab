#!/usr/bin/env python3
"""Hand-computed layout through window_census (python3 -m unittest test_window_census.py)."""
import contextlib
import csv
import io
import os
import tempfile
import unittest

import window_census as wc

COLS = ["Kernel_Name", "Start_Timestamp", "End_Timestamp", "Queue_Id", "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z",
        "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z", "LDS_Block_Size"]


def row(name, s_us, e_us, queue, ctas, lds, wg=256):
    return [name, s_us * 1000, e_us * 1000, queue, ctas * wg, 1, 1, wg, 1, 1, lds]


class Census(unittest.TestCase):
    def setUp(self):
        rows = [row("embedding_kernel", 0, 10, 1, 1, 0)]
        # two 16-CTA flash launches at a time on 4 queues for 400 us: 32 of 40 CUs, no idle
        for i in range(4):
            for q in (0, 1):
                rows.append(row("void strix::attn_core::sparse::flash_rows_kernel<P, 32u>", 100 + 100 * i, 200 + 100 * i,
                                1 + q + 2 * (i % 2), 16, 60416))
        # a select pass that cannot share a CU with a flash CTA (LDS) fills the other 8 CUs for the whole loop
        rows.append(row("void strix::ds41::attn::hip::select_slice_kernel<false>", 100, 500, 4, 8, 52736, 1024))
        rows.append(row("dense_kernel", 500, 600, 1, 100, 0))  # ends the cluster
        fd, self.path = tempfile.mkstemp(suffix=".csv", dir=os.path.dirname(__file__))
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            w.writerows(rows)

    def tearDown(self):
        os.unlink(self.path)

    def write_rows(self, rows):
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            w.writerows(rows)

    def measure(self, rows, edge_us=300):
        self.write_rows(rows)
        return wc.census(wc.load(self.path), ("fill",), "fill", 40, edge_us * 1000)

    def test_layout(self):
        k = wc.load(self.path)
        r = wc.census(k, wc.DS41_KINDS, "flash_rows", 40, 50 * 1000)
        self.assertEqual(r["clusters"], 1)
        self.assertEqual(r["loop_launches"], 9)
        self.assertAlmostEqual(r["span_ms"], 0.4)
        self.assertAlmostEqual(r["idle_ms"], 0.0)
        self.assertAlmostEqual(r["deficit_ms"], 0.4 * 8 / 40)  # 32 flash CTAs of 40 CUs for 400 us
        self.assertAlmostEqual(r["deficit_edge_ms"], 0.1 * 8 / 40)  # 50 us at each end
        self.assertAlmostEqual(r["free_ms"], 0.0)  # the select CTAs take the other 8 CUs
        self.assertEqual(r["queues"], {1: 2, 2: 2, 3: 2, 4: 3})

    def test_cli_last_run(self):
        prefix = "void " + "long_namespace::" * 5
        rows = []
        for start, count in ((0, 8), (1000, 9)):
            rows.append(row(prefix + "embedding_kernel()", start, start + 1, 0, 1, 0))
            rows.extend(row(prefix + "flash_rows_kernel()", start + 10 + i, start + 11 + i,
                            0, 1, 0) for i in range(count))
        self.write_rows(rows)
        for run, expected in (("-1", "run 1 of 2: wall 0.0 ms, launches 10; window clusters 1, launches 9"),
                              ("0", "run 0 of 2: wall 0.0 ms, launches 9; window clusters 1, launches 8")):
            with self.subTest(run=run), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(wc.main([self.path, "--run", run]), 0)
                self.assertIn(expected, out.getvalue())

    def test_overlapping_and_touching_edges(self):
        for duration, edge in ((100, 300), (100, 75), (200, 100)):
            with self.subTest(duration=duration, edge=edge):
                r = self.measure([row("fill", 0, duration, q, 1, 0) for q in range(8)], edge)
                self.assertAlmostEqual(r["deficit_ms"], duration / 1000 * 0.8)
                self.assertAlmostEqual(r["deficit_edge_ms"], r["deficit_ms"])
                self.assertEqual(r["deficit_inner_ms"], 0)
                self.assertLessEqual(r["deficit_ms"], r["span_ms"])

    def test_other_queue_covers_loop_gaps(self):
        rows = [row("other", 0, 100, 1, 40, 65536)]
        rows += [row("fill", s, s + 1, 0, 1, 60416) for s in (*range(1, 8), 20)]
        r = self.measure(rows)
        self.assertAlmostEqual(r["span_ms"], 0.020)
        self.assertAlmostEqual(r["busy_ms"], 0.020)
        self.assertAlmostEqual(r["idle_ms"], 0)
        self.assertAlmostEqual(r["fill_busy_ms"], 0.008)
        self.assertAlmostEqual(r["deficit_ms"], 0.0198)
        self.assertAlmostEqual(r["free_ms"], 0)

    def test_overlapping_clusters_count_wall_time_once(self):
        rows = [row("fill", 0, 100, q, 1, 60416) for q in range(8)]
        rows += [row("other", 10, 90, 8, 32, 65536)]
        rows += [row("fill", 20, 80, q, 1, 60416) for q in range(8)]
        r = self.measure(rows)
        self.assertEqual(r["clusters"], 2)
        self.assertEqual(r["loop_launches"], 16)
        for metric in ("wall_ms", "span_ms", "busy_ms", "fill_busy_ms", "under_ms"):
            self.assertAlmostEqual(r[metric], 0.100, msg=metric)
        self.assertAlmostEqual(r["idle_ms"], 0)
        self.assertAlmostEqual(r["deficit_ms"], 0.068)
        self.assertAlmostEqual(r["free_ms"], 0.016)

    def test_disjoint_clusters_clip_crossing_kernels(self):
        rows = [row("fill", 0, 10, 0, 1, 60416), row("other", 5, 25, 1, 40, 65536),
                row("fill", 20, 30, 0, 1, 60416)]
        r = self.measure(rows, edge_us=2)
        self.assertEqual(r["clusters"], 2)
        self.assertAlmostEqual(r["span_ms"], 0.020)
        self.assertAlmostEqual(r["busy_ms"], 0.020)
        self.assertAlmostEqual(r["deficit_ms"], 0.0195)
        self.assertAlmostEqual(r["deficit_edge_ms"], 0.0078)
        self.assertAlmostEqual(r["deficit_inner_ms"], 0.0117)
        self.assertAlmostEqual(r["free_ms"], 0.00975)

    def test_full_symbols_classified_and_grouped(self):
        prefix = "void " + "long_namespace::" * 5
        names = [prefix + "fill_kernel<P, " + str(i % 2) + ">(int)" for i in range(8)]
        r = self.measure([row(n, 0, 100, 0, 1, 0) for n in names])
        self.assertEqual({entry[0] for entry in r["kinds"]}, set(names))
        self.assertEqual([entry[1] for entry in r["kinds"]], [4, 4])
        self.assertAlmostEqual(r["deficit_ms"], 0.080)

    def test_short_clusters_included(self):
        for count in (1, 7):
            with self.subTest(count=count):
                rows = [row("other", 0, 1, 0, 1, 0)]
                rows += [row("fill", i + 1, i + 2, 0, 1, 0) for i in range(count)]
                rows += [row("other", count + 2, count + 3, 0, 1, 0)]
                r = self.measure(rows)
                self.assertEqual(r["clusters"], 1)
                self.assertEqual(r["loop_launches"], count)
                self.assertAlmostEqual(r["span_ms"], count / 1000)

    def assert_cli_error(self, args):
        with contextlib.redirect_stderr(io.StringIO()) as err, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                wc.main([self.path] + args)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("error:", err.getvalue())

    def test_invalid_arguments(self):
        for args in (("--run", "1"), ("--run", "-2"), ("--cus", "0"), ("--cus", "-1"),
                     ("--cu-lds", "0"), ("--edge-us", "-1"), ("--kinds", ""),
                     ("--fill", ""), ("--split", "")):
            with self.subTest(args=args):
                self.assert_cli_error(list(args))

    def test_empty_and_malformed_csv(self):
        for content in ("", ",".join(COLS) + "\n", "wrong,columns\na,b\n",
                        ",".join(COLS) + "\nfill,not-a-number\n"):
            with self.subTest(content=content):
                with open(self.path, "w") as f:
                    f.write(content)
                self.assert_cli_error([])
        for bad in (row("fill", 2, 1, 0, 1, 0), row("fill", 0, 1, 0, 1, 0, wg=0),
                    row("fill", 0, 1, 0, 1, -1)):
            self.write_rows([bad])
            self.assert_cli_error([])

    def test_cli_echoes_assumptions(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(wc.main([self.path, "--cus", "20", "--cu-lds", "32768",
                                      "--kinds", "select", "--fill", "select"]), 0)
        self.assertIn("assumptions: CUs=20, LDS=32768 B/CU, kinds=select, fill=select", out.getvalue())


if __name__ == "__main__":
    unittest.main()
