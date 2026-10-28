"""Case data, input construction, and reference ops for matmul + bias.

The single source of truth for what a benchmark/test case *is*: bench.py
and test_fixed.py drive this module through problems/_harness.py, and
generate-kernel.py's optimizer adapter embeds make_inputs()/reference()
verbatim (via inspect.getsource) rather than re-deriving them — so there is
exactly one copy of this logic, and profile-kernel.py, bench.py, and the
optimizer's Model can never silently disagree about what case 0 means.

No torch import at module level: generate-kernel.py's and
profile-kernel.py's parent processes load this module just to read
BENCH_CASES (shapes, dtypes, a seed base) before forking worker processes,
and must not initialize CUDA or even import torch to do it. Dtypes are
therefore plain attribute names ("float16"), and make_inputs()/reference()
import torch lazily, inside the function body, so only the processes that
actually build tensors pay for it.
"""
from __future__ import annotations

# Benchmark cases: performance-oriented shapes (large, realistic), used by
# bench.py, profile-kernel.py, and the optimizer's problem module.
BENCH_CASES = [
    (1024, 1024, 1024, "float16"),
    (2048, 1024, 2048, "float16"),
    (2048, 2048, 2048, "bfloat16"),
    (4096, 4096, 4096, "float16"),
]

# Fixed-test cases: correctness-oriented shapes (aligned, non-tile-aligned,
# tiny), used by test_fixed.py only.
TEST_CASES = [
    (128, 128, 128, "float16"),
    (256, 512, 128, "bfloat16"),
    (64, 32, 16, "float16"),
    (100, 77, 51, "bfloat16"),
    (333, 65, 129, "float16"),
    (1024, 256, 512, "bfloat16"),
]

BENCH_SEED_BASE = 2000
TEST_SEED_BASE = 1000
FP16_TOL = 1e-2


def make_inputs(case, seed):
    """(A, B, bias) for one case, matching the kernel's expected signature."""
    import torch

    M, K, N, dtype_name = case
    dt = getattr(torch, dtype_name)
    torch.manual_seed(seed)
    A = torch.randn(M, K, device="cuda", dtype=dt) * 0.25
    B = torch.randn(K, N, device="cuda", dtype=dt) * 0.25
    bias = torch.randn(N, device="cuda", dtype=dt) * 0.5
    return A, B, bias


def output_shape(case):
    M, K, N, _dtype_name = case
    return (M, N)


def describe(case):
    M, K, N, dtype_name = case
    return f"M={M} K={K} N={N} {dtype_name}"


def case_fields(case):
    M, K, N, _dtype_name = case
    return {"M": M, "K": K, "N": N}


def flops(case):
    M, K, N, _dtype_name = case
    return 2.0 * M * K * N


def reference(A, B, bias):
    """Native-dtype PyTorch baseline: the speed comparison in bench.py and
    profile-kernel.py, and the Model the optimizer benchmarks/profiles
    against."""
    import torch

    return torch.addmm(bias, A, B)


def reference_precise(A, B, bias):
    """Upcast reference for test_fixed.py's correctness check."""
    import torch

    return (torch.matmul(A.float(), B.float()) + bias.float()).to(A.dtype)
