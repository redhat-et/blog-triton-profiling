"""Shared scaffolding for the fixed problem specs' tests and benchmarks.

Every problem's test_fixed.py and bench.py need the same scaffolding: a
CUDA guard, the output-contract check (tensor / right shape / right
dtype), a tolerance-gated allclose comparison against a reference, and —
for the benchmark only — timing and the BENCH_JSON emission. Keeping it
here means a change to any of those rules (an added contract check, a
different timing methodology) is made once and applies to every problem,
instead of being copy-pasted per problem and per script.

Each problem supplies its own shapes, seeds, input construction, and
reference op through its ``spec.py`` (see e.g. problems/matmul_bias/spec.py).
This module and spec.py together are the one place that logic lives;
test_fixed.py and bench.py are thin drivers over them, and
generate-kernel.py's optimizer adapter embeds spec.py's functions verbatim
rather than re-deriving them, so all four consumers can never disagree
about what a benchmark case actually is.

This module itself needs real torch (do_bench, tensor comparisons), so it
is only ever imported by test_fixed.py/bench.py, which always run as real
subprocesses with torch installed — never by the parent CLIs' case-loading
path, which reads spec.py instead and must stay torch-free.

The drivers' main() glue also lives here (run_fixed_main/run_bench_main):
the CUDA guard, the lazy kernel import, and the pass-everything-from-spec
call that every problem's test_fixed.py and bench.py would otherwise
copy verbatim. Each driver is a docstring plus one call into these.
"""
from __future__ import annotations

import json
import os

import torch
from triton.testing import do_bench


def require_cuda() -> bool:
    """True if CUDA is available; otherwise prints the contract's FAIL line."""
    if torch.cuda.is_available():
        return True
    print("FAIL: CUDA not available")
    return False


def time_ms(fn, warmup: int = 25, rep: int = 100) -> float:
    """Mean latency of fn in milliseconds, via triton.testing.do_bench.

    do_bench runs its own warmup and measurement loop (warmup/rep are time
    budgets in milliseconds, not iteration counts, and it discards outliers),
    so callers need no separate warmup pass — unlike a hand-rolled
    CUDA-event loop, this is the same methodology KernelAgent's own
    optimizer uses to time candidates, so the acceptance gate and the
    optimizer's search signal agree.
    """
    return do_bench(fn, warmup=warmup, rep=rep, return_mode="mean")


def resolve_dtype(name: str):
    """A case tuple's dtype name ("float16") as a real torch dtype."""
    return getattr(torch, name)


def check_contract(out, case_index: int, shape: tuple[int, ...], dtype) -> str | None:
    """None if `out` satisfies the problem contract, else a FAIL message.

    `dtype` is a resolved torch dtype (see resolve_dtype), not a case
    tuple's plain name string.
    """
    if not isinstance(out, torch.Tensor):
        return f"FAIL: case {case_index}: output is not a tensor"
    if tuple(out.shape) != tuple(shape):
        return f"FAIL: case {case_index}: shape {tuple(out.shape)} != {tuple(shape)}"
    if out.dtype != dtype:
        return f"FAIL: case {case_index}: dtype {out.dtype} != {dtype}"
    return None


def tol_for(dtype, fp16_tol: float = 1e-2) -> float:
    """Allclose tolerance for a case's dtype (bf16 needs more slack)."""
    return 3e-2 if dtype == torch.bfloat16 else fp16_tol


def check_allclose(out, ref, case_index: int, tol: float) -> str | None:
    """None if `out` matches `ref` within tol, else a FAIL message."""
    if torch.allclose(out.float(), ref.float(), atol=tol, rtol=tol):
        return None
    err = (out.float() - ref.float()).abs().max().item()
    return f"FAIL: case {case_index}: max abs err {err:.3e} (tol {tol})"


def _evaluate_case(
    case,
    index: int,
    seed_base: int,
    make_inputs,
    kernel_fn,
    reference_fn,
    output_shape,
    fp16_tol: float,
):
    """Run and validate one case: (dtype name, args, ref, failure).

    ``ref`` is None when the contract check short-circuits, since there is
    no point computing the reference for an output that already failed.
    """
    dt_name = case[-1]
    dt = resolve_dtype(dt_name)
    args = make_inputs(case, seed_base + index)
    out = kernel_fn(*args)
    fail = check_contract(out, index, output_shape(case), dt)
    if fail:
        return dt_name, args, None, fail
    ref = reference_fn(*args)
    fail = check_allclose(out, ref, index, tol_for(dt, fp16_tol))
    return dt_name, args, ref, fail


def run_fixed_test(
    cases,
    seed_base: int,
    make_inputs,
    kernel_fn,
    reference_fn,
    output_shape,
    describe,
    *,
    fp16_tol: float = 1e-2,
) -> int:
    """Run every fixed-test case; print PASS and return 0 iff all pass."""
    for i, case in enumerate(cases):
        _, _, _, fail = _evaluate_case(
            case,
            i,
            seed_base,
            make_inputs,
            kernel_fn,
            reference_fn,
            output_shape,
            fp16_tol,
        )
        if fail:
            print(fail)
            return 1
        print(f"case {i}: {describe(case)} OK")
    print("PASS")
    return 0


def _bench_case_filter(case_count: int) -> int | None:
    """The single benchmark case selected by KA_BENCH_CASE (an index into
    the spec's BENCH_CASES), or None for the full run.

    generate-kernel.py sets it for the optimizer's internal candidate
    checks, where only the selected case's timing feeds the acceptance
    verdict. The `verify` command's published records always run the
    full benchmark; the optimize path's published verify_optimized.json
    carries only the optimized case (its original index and seed
    preserved), which is a subset of the full record and still the only
    case the experiment manifest and report read."""
    raw = (os.environ.get("KA_BENCH_CASE") or "").strip()
    if not raw:
        return None
    try:
        index = int(raw)
    except ValueError as exc:
        raise SystemExit(
            f"ERROR: KA_BENCH_CASE='{raw}' is not an integer"
        ) from exc
    if not 0 <= index < case_count:
        raise SystemExit(
            f"ERROR: KA_BENCH_CASE={index} is out of range "
            f"0..{case_count - 1}"
        )
    return index


def run_benchmark(
    cases,
    seed_base: int,
    make_inputs,
    kernel_fn,
    reference_fn,
    output_shape,
    describe,
    case_fields,
    *,
    fp16_tol: float = 1e-2,
    flops=None,
    only_case: int | None = None,
) -> int:
    """Run the benchmark cases, print a BENCH_JSON line, return 0 on
    success.

    Validates the output contract and correctness for each case before
    timing it — a kernel that violates the contract (or is simply wrong)
    must not be benchmarked, let alone published as BENCH_JSON: the
    acceptance rule for optimized kernels is "strictly faster", which is
    a real path to publishing an invalid kernel if speed were the only
    gate. only_case restricts the run to that case index (keeping its
    original index and seed, so a filtered BENCH_JSON is a subset of the
    full one) — the optimizer's internal checks use it to time only the
    case the acceptance verdict reads.
    """
    results = []
    for i, case in enumerate(cases):
        if only_case is not None and i != only_case:
            continue
        dt_name, args, ref, fail = _evaluate_case(
            case,
            i,
            seed_base,
            make_inputs,
            kernel_fn,
            reference_fn,
            output_shape,
            fp16_tol,
        )
        if fail:
            print(fail)
            return 1

        k_ms = time_ms(lambda: kernel_fn(*args))
        r_ms = time_ms(lambda: reference_fn(*args))
        entry = {
            "case": i,
            **case_fields(case),
            "dtype": dt_name,
            "kernel_ms": round(k_ms, 4),
            "ref_ms": round(r_ms, 4),
            "speedup": round(r_ms / k_ms, 3) if k_ms > 0 else None,
        }
        tflops_note = ""
        if flops is not None:
            f = flops(case)
            entry["kernel_tflops"] = round(f / (k_ms * 1e-3) / 1e12, 2) if k_ms > 0 else None
            entry["ref_tflops"] = round(f / (r_ms * 1e-3) / 1e12, 2) if r_ms > 0 else None
            tflops_note = f" kernel={entry['kernel_tflops']}TFLOPS"
        results.append(entry)
        print(
            f"case {i}: {describe(case)} kernel={k_ms:.3f}ms ref={r_ms:.3f}ms "
            f"speedup={entry['speedup']}x{tflops_note}"
        )

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "cases": results,
    }
    print("BENCH_JSON " + json.dumps(payload))
    return 0


def run_fixed_main(spec) -> int:
    """The whole of a problem's test_fixed.py main(), driven by its spec.

    The kernel is imported lazily, after the CUDA guard, so merely loading
    the spec and this harness never requires kernel.py — only running the
    test does, from its cwd.
    """
    if not require_cuda():
        return 1
    from kernel import kernel_function

    return run_fixed_test(
        spec.TEST_CASES,
        spec.TEST_SEED_BASE,
        spec.make_inputs,
        kernel_function,
        spec.reference_precise,
        spec.output_shape,
        spec.describe,
        fp16_tol=spec.FP16_TOL,
    )


def run_bench_main(spec) -> int:
    """The whole of a problem's bench.py main(), driven by its spec.

    Same shape as run_fixed_main but against the native-datatype
    reference, plus the case metadata run_benchmark's BENCH_JSON carries.
    A spec's optional ``flops`` (e.g. matmul_bias's 2*M*K*N) opts a
    problem into the TFLOPS columns; without one they are omitted.
    KA_BENCH_CASE (resolved here, see _bench_case_filter) restricts the
    run to one case for the optimizer's internal checks.
    """
    if not require_cuda():
        return 1
    from kernel import kernel_function

    return run_benchmark(
        spec.BENCH_CASES,
        spec.BENCH_SEED_BASE,
        spec.make_inputs,
        kernel_function,
        spec.reference,
        spec.output_shape,
        spec.describe,
        spec.case_fields,
        fp16_tol=spec.FP16_TOL,
        flops=getattr(spec, "flops", None),
        only_case=_bench_case_filter(len(spec.BENCH_CASES)),
    )
