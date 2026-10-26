"""Case data, input construction, and reference ops for flash attention.

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
    (2, 8, 1024, 64, False, "float16"),
    (2, 8, 1024, 64, True, "float16"),
    (1, 16, 2048, 64, True, "bfloat16"),
    (4, 8, 2048, 128, True, "float16"),
]

# Fixed-test cases: correctness-oriented shapes (causal/non-causal, both
# dtypes, D in {32, 64, 128}, non-tile-aligned sequence lengths), used by
# test_fixed.py only.
TEST_CASES = [
    (1, 2, 128, 64, False, "float16"),
    (2, 4, 256, 64, False, "bfloat16"),
    (2, 4, 256, 64, True, "float16"),
    (1, 8, 512, 32, False, "bfloat16"),
    (1, 8, 512, 32, True, "bfloat16"),
    (2, 3, 333, 64, True, "float16"),
    (2, 3, 333, 64, False, "bfloat16"),
    (1, 2, 777, 64, True, "float16"),
    (1, 4, 256, 128, True, "float16"),
    (2, 2, 513, 128, False, "bfloat16"),
]

BENCH_SEED_BASE = 4000
TEST_SEED_BASE = 3000
FP16_TOL = 2e-2


def make_inputs(case, seed):
    """(Q, K, V, causal) for one case, matching the kernel's expected
    signature."""
    import torch

    batch, heads, seq, head_dim, causal, dtype_name = case
    dt = getattr(torch, dtype_name)
    torch.manual_seed(seed)
    Q = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=dt)
    K = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=dt)
    V = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=dt)
    return Q, K, V, causal


def output_shape(case):
    batch, heads, seq, head_dim, _causal, _dtype_name = case
    return (batch, heads, seq, head_dim)


def describe(case):
    batch, heads, seq, head_dim, causal, dtype_name = case
    return f"B={batch} H={heads} S={seq} D={head_dim} causal={causal} {dtype_name}"


def case_fields(case):
    batch, heads, seq, head_dim, causal, _dtype_name = case
    return {"B": batch, "H": heads, "S": seq, "D": head_dim, "causal": causal}


def flops(case):
    """FLOPs for one case: both matmuls at 2 FLOPs per multiply-add
    (softmax overhead excluded). Causal keeps only the lower triangle,
    diagonal included — S*(S+1)/2 score entries per (batch, head)."""
    batch, heads, seq, head_dim, causal, _dtype_name = case
    if causal:
        return 2.0 * batch * heads * head_dim * seq * (seq + 1)
    return 4.0 * batch * heads * seq * seq * head_dim


def reference(Q, K, V, causal):
    """Native-dtype SDPA baseline: the speed comparison in bench.py and
    profile-kernel.py, and the Model the optimizer benchmarks/profiles
    against."""
    import torch.nn.functional as F

    return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)


def reference_precise(Q, K, V, causal):
    """From-scratch fp32 softmax reference for test_fixed.py's correctness
    check — independent of SDPA, so it cannot share a bug with it."""
    import torch

    q, k, v = Q.float(), K.float(), V.float()
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    if causal:
        seq = q.shape[-2]
        keep = torch.ones(seq, seq, device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~keep, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v).to(Q.dtype)
