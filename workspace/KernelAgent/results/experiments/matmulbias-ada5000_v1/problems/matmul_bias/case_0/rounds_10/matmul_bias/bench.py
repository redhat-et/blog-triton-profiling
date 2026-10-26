"""Benchmark the generated matmul + bias kernel against the PyTorch baseline.

Contract: executed with cwd containing kernel.py; prints a final
"BENCH_JSON <json>" line and exits 0 on success. Case data, input
construction, and the reference op live in spec.py; the CUDA guard,
kernel import, and timing/contract scaffolding live in _harness.py —
both shared with test_fixed.py, profile-kernel.py, and
generate-kernel.py's optimizer adapter.
"""
import sys

import spec
from _harness import run_bench_main

if __name__ == "__main__":
    sys.exit(run_bench_main(spec))
