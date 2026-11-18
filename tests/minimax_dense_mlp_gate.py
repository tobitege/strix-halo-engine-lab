"""Run the dense-v4 device gate for both rank shapes and every supported width.

Usage: python tests/minimax_dense_mlp_gate.py PATH_TO_HIP_MMX_MOE_BENCH
Synthetic weights vary by rank. For actual model slices also run the binary with
--tests test_prefill_dense --model MODEL_DIR --rank 0 (and --rank 1).
This matrix requires HIP and the existing operator library; it is not a host gate.
"""
import re
import subprocess
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    for rank in (0, 1):
        for bits in (3, 4, 5, 6, 8):
            result = subprocess.run(
                [sys.argv[1], "--tests", "test_prefill_dense", "--rank", str(rank),
                 "--dense-bits", str(bits), "--no-moe"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            receipt = rf"dense_v4 chain rank {rank} bits {bits}/{bits}/{bits} T4096: bitwise"
            if result.returncode or len(re.findall(receipt, result.stdout)) != 3:
                print("\n".join(result.stdout.splitlines()[-20:]))
                raise SystemExit(f"dense MLP gate failed: rank={rank}, bits={bits}, exit={result.returncode}")
            print(f"rank={rank} bits={bits}: operator and composed chain gates passed (bands 0/4/8)")


if __name__ == "__main__":
    main()
