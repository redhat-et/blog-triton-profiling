"""Fixed correctness test for the fused matmul + bias Triton kernel.

Contract: executed with cwd containing kernel.py; exits 0 on success.
Covers aligned shapes, non-tile-aligned shapes, and both fp16/bf16. Case
data, input construction, and the reference op live in spec.py; the CUDA
guard, kernel import, and contract-check scaffolding live in _harness.py
— both shared with bench.py, profile-kernel.py, and generate-kernel.py's
optimizer adapter.
"""
import sys

import spec
from _harness import run_fixed_main

if __name__ == "__main__":
    sys.exit(run_fixed_main(spec))
