#!/usr/bin/env python3
"""Profile a generated kernel (or the PyTorch baseline) for one benchmark case.

Loads results/<problem>/final_kernel.py — or optimized_kernel.py with
--impl optimized — plus the benchmark cases from the problem spec, runs
warmup followed by measured iterations of the selected case, and captures
the measured region with the requested profiler:

  proton  in-process Proton capture (writes <stem>.hatchet and prints a tree)
  nsys    re-executes this script under Nsight Systems with
          --capture-range=cudaProfilerApi (writes <stem>.nsys-rep)
  ncu     re-executes this script under Nsight Compute with
          --profile-from-start off (writes <stem>.ncu-rep)

The measured region is delimited with torch.cuda.profiler.start()/stop()
(cudaProfilerStart/Stop), so timeline captures skip input creation, JIT
compilation, autotuning, and warmup launches.

Usage:
  profile-kernel.py <problem> [--case N] [--impl kernel|optimized|ref]
                     [--profiler proton|nsys|ncu] [--iters N] [--warmup N]
                     [--kernel PATH] [--out CAPTURE_DIR] [--ncu-set SET]
                     [--label NAME]

CAPTURE_DIR is the exact directory the report files are written to; it
defaults to results/<problem>/profiles.

Examples:
  profile-kernel.py matmul_bias --case 3 --profiler proton
  profile-kernel.py matmul_bias --case 3 --impl optimized --profiler proton
  profile-kernel.py flash_attention --case 3 --impl ref --profiler nsys
  profile-kernel.py matmul_bias --case 0 --profiler ncu --ncu-set detailed
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import ka_common
from ka_common import (
    IMPL_ARTIFACTS,
    _load_module,
    validate_label,
)


def _profiler_failed(tool: str, returncode: int, report: Path) -> int:
    """Diagnose a failed profiler run and map its status to an exit code.

    subprocess reports a signal death as a negative returncode, which
    SystemExit would truncate modulo 256 (a SIGSEGV death would exit
    245, and -256 would exit 0); map it to the shell's 128+signal
    convention instead so the caller sees what actually happened.
    """
    if returncode < 0:
        detail, code = f"was killed by signal {-returncode}", 128 - returncode
    else:
        detail, code = f"exited with {returncode}", returncode
    print(
        f"ERROR: {tool} {detail}"
        + ("" if report.is_file() else " and no report was generated"),
        file=sys.stderr,
    )
    return code


def _profiler_invocation(
    tool: str, stem: Path, ncu_set: str
) -> tuple[list[str], str, dict[str, str] | None]:
    """(argv prefix, report extension, env override) for nsys/ncu.

    ncu spawns its own python for report post-processing and picks the
    first interpreter on PATH; the venv interpreter's .pth processing
    breaks under ncu's injection (harmless but noisy), so its env override
    points ncu at the system python. The profiled child itself is always
    invoked by absolute path, so this never affects what gets profiled.
    """
    if tool == "nsys":
        return (
            [
                "nsys",
                "profile",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--trace=cuda,nvtx",
                "--force-overwrite=true",
                "-o",
                str(stem),
            ],
            "nsys-rep",
            None,
        )
    return (
        [
            "ncu",
            "--profile-from-start",
            "off",
            "--set",
            ncu_set,
            "--force-overwrite",
            "-o",
            str(stem),
        ],
        "ncu-rep",
        {"PATH": "/usr/local/bin:/usr/bin:/bin"},
    )


def _merged_env(env_override: dict[str, str] | None) -> dict[str, str] | None:
    """The child environment for a profiler-owned subprocess: the current
    environment with the override applied, or None to inherit unchanged.

    The override exists to repoint ncu's PATH at the system interpreter —
    it was never meant to strip HOME, LD_LIBRARY_PATH, or the rest of the
    environment from the version probe and the CSV exports, which run
    the same ncu binary the profiler launch does. The launch path has
    always merged; these subprocesses must see the same world.
    """
    return {**os.environ, **env_override} if env_override else None


def _export_result(name: str, ok: bool, paths: list[Path]) -> dict:
    """One export entry for the metadata record: status plus the files it
    produced (empty and marked failed when the export could not run)."""
    return {
        "name": name,
        "status": "ok" if ok else "failed",
        "paths": [str(p) for p in paths if p is not None],
    }


def _export_ncu(stem: Path, env_override: dict[str, str] | None) -> list[dict]:
    """Export the ncu report's details and raw pages as CSV beside it.

    The details page carries the human-readable SOL/occupancy sections;
    the raw page carries every collected metric by name. Both feed the
    experiment report's normalized metrics (parsed offline by
    report-experiment.py from these CSVs, never from the .ncu-rep). The
    two exports are independent reads of the same report, so they run
    concurrently — ncu's report processing dominates the wall clock of
    this step."""
    report = Path(f"{stem}.ncu-rep")
    pages = [
        (page, Path(f"{stem}{suffix}"))
        for page, suffix in (("details", ".details.csv"), ("raw", ".raw.csv"))
    ]
    env = _merged_env(env_override)

    # (page, dest, out, proc) — entries whose Popen itself failed carry
    # None, so the collect loop still records a failed export for them.
    started = []
    for page, dest in pages:
        out, proc = None, None
        try:
            out = open(dest, "w", encoding="utf-8")
            proc = subprocess.Popen(
                ["ncu", "--import", str(report), "--csv", "--page", page],
                stdout=out,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        except OSError as exc:
            print(f"WARNING: export failed: {exc}", file=sys.stderr)
            if out is not None:
                out.close()
                out, proc = None, None
        started.append((page, dest, out, proc))

    exports = []
    for page, dest, out, proc in started:
        ok = False
        if proc is not None:
            try:
                _, stderr = proc.communicate(timeout=600)
                ok = proc.returncode == 0
                if not ok:
                    print(
                        f"WARNING: export exited {proc.returncode}: "
                        f"{(stderr or '').strip()[:200]}",
                        file=sys.stderr,
                    )
            except (OSError, subprocess.TimeoutExpired) as exc:
                proc.kill()
                proc.communicate()
                print(f"WARNING: export failed: {exc}", file=sys.stderr)
        if out is not None:
            out.close()
        exports.append(_export_result(f"ncu_{page}_csv", ok, [dest]))
    return exports


def _export_nsys(stem: Path) -> list[dict]:
    """Export nsys stats summaries (GPU kernels, CUDA API) as CSV.

    nsys stats writes <stem>_<report>.csv files next to the stem for each
    requested report; the export collects whatever appeared."""
    report = Path(f"{stem}.nsys-rep")
    print(f"==> export: nsys stats {report}")
    try:
        result = subprocess.run(
            [
                "nsys", "stats",
                "--report", "cuda_gpu_kern_sum",
                "--report", "cuda_api_sum",
                "--format", "csv",
                "--force-export=true",
                "--output", str(stem),
                str(report),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"WARNING: nsys stats failed: {exc}", file=sys.stderr)
        return [_export_result("nsys_stats", False, [])]
    produced = sorted(stem.parent.glob(f"{stem.name}_cuda_*.csv"))
    ok = result.returncode == 0 and bool(produced)
    if not ok:
        print(
            f"WARNING: nsys stats exited {result.returncode} "
            f"({len(produced)} CSV files): {(result.stderr or '').strip()[:200]}",
            file=sys.stderr,
        )
    return [_export_result("nsys_stats", ok, produced)]

# Default kernel profiled per --impl (ka_common.IMPL_ARTIFACTS); ref
# profiles the PyTorch baseline instead of any saved kernel.
PROBLEMS_HOME = Path(
    os.environ.get("PROBLEMS_HOME", ka_common.PROBLEMS_HOME_DEFAULT)
)
RESULTS_MOUNT = Path(
    os.environ.get("RESULTS_MOUNT", ka_common.RESULTS_MOUNT_DEFAULT)
)


def _load_kernel(problem: str, kernel_path: Path | None, impl: str = "kernel"):
    # The ref case never reaches this loader (it profiles the PyTorch
    # baseline), so _resolve_kernel's ref handling does not apply here.
    path = _resolve_kernel(problem, kernel_path, impl)
    module = _load_module("profiled_kernel", path)
    fn = getattr(module, "kernel_function", None)
    if fn is None:
        raise SystemExit(f"ERROR: {path} does not define kernel_function")
    return fn


def _resolve_kernel(problem: str, kernel_path: Path | None, impl: str) -> Path | None:
    """The original kernel source this run profiles, resolved once before
    any capture path can import or re-exec it: the explicit --kernel
    path, or the default saved artifact for the impl. None for ref
    captures, which profile the PyTorch baseline instead of any saved
    kernel."""
    if impl == "ref":
        return None
    path = kernel_path or RESULTS_MOUNT / problem / IMPL_ARTIFACTS[impl]["kernel"]
    if not path.is_file():
        how = (
            f"run 'generate-kernel.py generate {problem}' first"
            if impl == "kernel"
            else f"run 'generate-kernel.py optimize {problem}' first"
        )
        raise SystemExit(f"ERROR: {path} not found; {how} (or pass --kernel)")
    return path


@contextlib.contextmanager
def _capture_source(kernel_path: Path | None):
    """A private snapshot of the kernel source, kept alive for the
    capture: yields (snapshot path, sha256 of its bytes).

    The source bytes are read once, before any import or child launch
    can run; both execution paths and the metadata hash derive from that
    one read, so a capture can never measure one kernel while its
    metadata describes a replacement published mid-run. The snapshot
    lands under a fresh temporary name — no stale bytecode can be
    associated with it — and is removed (with any bytecode importlib
    cached beside it) when the capture and its exports finish, however
    the run ends. Reference captures pass None and snapshot nothing.
    """
    if kernel_path is None:
        yield None, None
        return
    source = kernel_path.read_bytes()
    sha256 = ka_common.sha256_bytes(source)
    fd, name = tempfile.mkstemp(
        prefix=f"profiled_{kernel_path.stem}_", suffix=".py"
    )
    snapshot = Path(name)
    bytecode = Path(importlib.util.cache_from_source(name))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(source)
        yield snapshot, sha256
    finally:
        for leftover in (snapshot, bytecode):
            try:
                leftover.unlink()
            except OSError:
                pass
        try:
            bytecode.parent.rmdir()  # the __pycache__ dir, if now empty
        except OSError:
            pass


def _make_case(spec_mod, index: int, case: tuple):
    """Build (kernel args, PyTorch reference fn) for one case, via the
    problem's own spec.make_inputs()/spec.reference() — the same functions
    bench.py drives, so the profiler can never measure different tensors
    than the benchmark that gated acceptance."""
    seed = spec_mod.BENCH_SEED_BASE + index
    args = spec_mod.make_inputs(case, seed)
    return args, lambda: spec_mod.reference(*args)


def _run_measured(
    problem: str,
    case_index: int,
    impl: str,
    iters: int,
    warmup: int,
    kernel_path: Path | None,
    proton_stem: Path | None = None,
    case: tuple | None = None,
    spec_mod=None,
    meta_out: Path | None = None,
) -> None:
    """Warm up, then run the measured region between cudaProfilerStart/Stop.

    With proton_stem set, Proton is started after warmup and finalized after
    the measured region (in-process capture, no re-exec). case/spec_mod let
    an in-process caller that already loaded them (main()'s proton path)
    skip reloading spec.py; the --internal-child subprocess path has no
    such caller in its own process and loads them itself. meta_out, when
    set, receives the runtime facts (GPU name, torch/triton versions) from
    this process — the only one that already imported torch.
    """
    import torch

    if spec_mod is None:
        spec_mod = ka_common._load_spec(PROBLEMS_HOME, problem)
    if case is None:
        case = spec_mod.BENCH_CASES[case_index]
    args, ref = _make_case(spec_mod, case_index, case)

    if impl != "ref":
        kernel_function = _load_kernel(problem, kernel_path, impl)

        def run():
            return kernel_function(*args)

    else:
        def run():
            return ref()

    print(f"==> warmup ({warmup} iterations; includes JIT compile and autotuning)")
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    if proton_stem is not None:
        import triton.profiler as proton

        # The triton hook adds per-launch names for Triton kernels; the
        # baseline (cuBLAS/cuDNN kernels) is captured by the device backend.
        # Start after warmup so the capture window matches ncu/nsys.
        hook = "triton" if impl != "ref" else None
        print(f"==> proton start ({proton_stem}.hatchet)")
        proton.start(str(proton_stem), context="shadow", hook=hook)

    print(f"==> measured region ({iters} iterations, impl={impl})")
    torch.cuda.profiler.start()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

    if proton_stem is not None:
        proton.finalize()
        print("==> proton finalize")
    if meta_out is not None:
        # Runtime facts recorded by whichever process actually imported
        # torch (this one), so a capture-batch parent never pays a torch
        # import just to describe the capture it orchestrated.
        try:
            import triton

            triton_version = triton.__version__
        except Exception:
            triton_version = None
        ka_common._atomic_write_json(
            meta_out,
            {
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "unknown"
                ),
                "torch": torch.__version__,
                "triton": triton_version,
            },
        )
    print("==> measured region complete")


def _print_proton_tree(hatchet_path: Path) -> None:
    try:
        import triton.profiler.viewer as proton_viewer
    except ImportError:
        print("proton-viewer unavailable; inspect the .hatchet file directly")
        return
    tree, metrics = proton_viewer.parse(["time/ms"], str(hatchet_path))
    proton_viewer.print_tree(tree, metrics)


def _read_runtime_fragment(path: Path | None) -> dict:
    """The measured child's runtime facts (GPU name, torch/triton
    versions), recorded by the child itself so this process never
    imports torch just to describe a capture. A missing or malformed
    fragment degrades to empty — the capture itself already succeeded
    or failed on its own."""
    if path is None:
        return {}
    return ka_common.load_json_object(path, {})


def _write_metadata(
    stem: Path,
    args,
    case: tuple,
    kernel_path: Path | None,
    kernel_sha256: str | None,
    destination: Path,
    exports: list[dict] | None = None,
    report_path: Path | None = None,
    runtime_fragment: Path | None = None,
) -> Path:
    # kernel_path is the *original* source this capture profiled (None for
    # ref), resolved before capture began; kernel_sha256 was computed from
    # the snapshotted bytes that were actually executed, so the pair pins
    # exactly which saved kernel source produced this capture — a report
    # can refuse to attribute the metrics to a kernel that was replaced
    # after the fact. report_path is this capture's published report,
    # recorded here (not re-derived by readers) so the file-naming
    # convention has one owner. runtime facts come from the measured
    # child's fragment, keeping this process torch-free on the ncu/nsys
    # paths. The profiler version prefers KA_TOOL_VERSION_<TOOL>, which
    # the experiment runner probes once per capture batch — only a
    # standalone invocation pays for its own `--version` subprocess.
    profiler_version = None
    if args.profiler in ("ncu", "nsys"):
        profiler_version = (os.environ.get(
            f"KA_TOOL_VERSION_{args.profiler.upper()}"
        ) or "").strip() or None
        if profiler_version is None:
            _, _, env_override = _profiler_invocation(
                args.profiler, stem, args.ncu_set
            )
            profiler_version = ka_common.probe_tool_version(
                args.profiler, _merged_env(env_override)
            )
    runtime = _read_runtime_fragment(runtime_fragment)
    meta = {
        "problem": args.problem,
        "case": args.case,
        "case_spec": str(case),
        "impl": args.impl,
        "label": args.label,
        "profiler": args.profiler,
        "iters": args.iters,
        "warmup": args.warmup,
        "ncu_set": args.ncu_set if args.profiler == "ncu" else None,
        "gpu": runtime.get("gpu", "unknown"),
        "torch": runtime.get("torch"),
        "triton": runtime.get("triton"),
        "profiler_version": profiler_version,
        "report": str(report_path) if report_path else None,
        "kernel_path": str(kernel_path) if kernel_path else None,
        "kernel_sha256": kernel_sha256,
        "exports": exports or [],
        "command": " ".join(sys.argv),
        "created_at": datetime.now().isoformat(),
    }
    meta_path = destination
    ka_common._atomic_write_json(meta_path, meta)
    return meta_path


def _prepare_published_exports(
    exports: list[dict], staging_stem: Path, public_stem: Path
) -> tuple[list[dict], list[tuple[Path, Path]]]:
    """Map successful staged exports to their final paths.

    Export failures remain visible in metadata but do not publish partial
    files or point at temporary paths that disappear after the capture.
    """
    records = []
    moves = []
    for export in exports:
        record = {**export, "paths": []}
        paths = []
        for raw_path in export.get("paths", []):
            source = Path(raw_path)
            if (
                export.get("status") == "ok"
                and source.is_file()
                and source.parent == staging_stem.parent
            ):
                destination = public_stem.parent / source.name
                paths.append(str(destination))
                moves.append((source, destination))
        if export.get("status") == "ok" and not paths:
            record["status"] = "failed"
        record["paths"] = paths
        records.append(record)
    return records, moves


def _publish_capture(
    staging_stem: Path,
    public_stem: Path,
    report_ext: str,
    export_moves: list[tuple[Path, Path]],
    profiler: str,
    staged_metadata: Path,
) -> Path:
    """Publish a complete capture and its metadata under the public stem.

    The capture report is mandatory and checked before this function is
    called. Export files are replaced first, stale exports are removed,
    and metadata is replaced last so it never advertises staged paths.
    """
    report_source = Path(f"{staging_stem}.{report_ext}")
    report_target = Path(f"{public_stem}.{report_ext}")
    os.replace(report_source, report_target)

    published = set()
    for source, destination in export_moves:
        if source == report_source:
            published.add(destination)
            continue
        os.replace(source, destination)
        published.add(destination)

    if profiler == "ncu":
        stale = [
            Path(f"{public_stem}.details.csv"),
            Path(f"{public_stem}.raw.csv"),
        ]
    elif profiler == "nsys":
        stale = list(public_stem.parent.glob(f"{public_stem.name}_cuda_*.csv"))
    else:
        stale = []
    for path in stale:
        if path not in published:
            path.unlink(missing_ok=True)

    metadata_path = Path(f"{public_stem}.json")
    os.replace(staged_metadata, metadata_path)
    return metadata_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "problem", choices=ka_common.available_problems(PROBLEMS_HOME),
        help="problem to profile"
    )
    parser.add_argument(
        "--case",
        type=int,
        default=0,
        help="benchmark case index from the problem's bench.py (default: 0)",
    )
    parser.add_argument(
        "--impl",
        choices=[*IMPL_ARTIFACTS, "ref"],
        default="kernel",
        help="profile the generated kernel, the optimized kernel, or the "
        "PyTorch baseline (default: kernel)",
    )
    parser.add_argument(
        "--profiler",
        choices=ka_common.SUPPORTED_PROFILERS,
        default="proton",
        help="profiling tool (default: proton)",
    )
    parser.add_argument(
        "--iters", type=int, default=10, help="measured iterations (default: 10)"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--ncu-set",
        dest="ncu_set",
        default="basic",
        help="ncu --set value (default: basic; use detailed/full for more)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="stem prefix for the report files "
        "(default: the --impl value); must be a safe filename "
        "component, e.g. 'rounds_5' when profiling one experiment "
        "variant among several that share an output directory",
    )
    parser.add_argument(
        "--kernel",
        type=Path,
        default=None,
        help="kernel to profile (default: results/<problem>/final_kernel.py)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="exact directory for this problem's captures "
        "(default: results/<problem>/profiles)",
    )
    parser.add_argument(
        "--internal-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--meta-out",
        dest="meta_out",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.iters < 1:
        raise SystemExit(f"ERROR: --iters must be >= 1 (got {args.iters})")
    if args.warmup < 0:
        raise SystemExit(f"ERROR: --warmup must be >= 0 (got {args.warmup})")
    if args.label is not None:
        # The shared rules: a safe filename component that does not
        # collide with the default impl-based stems.
        validate_label(args.label)
    if args.impl == "ref" and args.kernel is not None:
        # The Makefile layer already rejects this combination at parse
        # time; direct CLI users deserve the same refusal instead of a
        # silently ignored --kernel that profiles the PyTorch baseline.
        raise SystemExit(
            "ERROR: --kernel cannot be combined with --impl ref — the "
            "PyTorch baseline has no kernel source"
        )

    if args.internal_child:
        _run_measured(
            args.problem,
            args.case,
            args.impl,
            args.iters,
            args.warmup,
            args.kernel,
            meta_out=args.meta_out,
        )
        return 0

    spec_mod = ka_common._load_spec(PROBLEMS_HOME, args.problem)
    cases = list(spec_mod.BENCH_CASES)
    if not 0 <= args.case < len(cases):
        raise SystemExit(f"ERROR: --case must be in 0..{len(cases) - 1}")

    out_dir = (
        args.out
        if args.out is not None
        else RESULTS_MOUNT / args.problem / "profiles"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{args.label or args.impl}_{args.profiler}_case{args.case}"

    # The original source is resolved — and its existence checked — once,
    # before any capture path can import or re-exec it.
    original_kernel = _resolve_kernel(args.problem, args.kernel, args.impl)

    child_cmd = [
        sys.executable,
        os.path.abspath(__file__),
        args.problem,
        "--case",
        str(args.case),
        "--impl",
        args.impl,
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--internal-child",
    ]

    ka_common.print_banner(
        f"KernelAgent — kernel profiling: {args.problem} case {args.case}"
    )
    print(f"Case spec:  {cases[args.case]}")
    print(f"Impl:       {args.impl}  Profiler: {args.profiler}  Iters: {args.iters}")

    with tempfile.TemporaryDirectory(
        prefix=f".{stem.name}.capture_", dir=out_dir
    ) as capture_dir_name:
        capture_dir = Path(capture_dir_name)
        staging_stem = capture_dir / stem.name
        # The measured child records its runtime facts here (one torch
        # import, in the process that pays it anyway).
        runtime_fragment = capture_dir / f"{stem.name}.runtime.json"
        with _capture_source(original_kernel) as (snapshot, kernel_sha256):
            # Both capture paths execute the snapshot rather than the
            # original file: the nsys/ncu child always receives it via
            # --kernel (also when the user selected the default kernel), and
            # the in-process proton path imports it — so a replacement
            # published while the profiler runs can change neither what is
            # measured nor what the metadata records.
            if snapshot is not None:
                child_cmd += ["--kernel", str(snapshot)]
            if args.profiler == "proton":
                report_ext = "hatchet"
                _run_measured(
                    args.problem,
                    args.case,
                    args.impl,
                    args.iters,
                    args.warmup,
                    snapshot,
                    proton_stem=staging_stem,
                    case=cases[args.case],
                    spec_mod=spec_mod,
                    meta_out=runtime_fragment,
                )
                report = Path(f"{staging_stem}.{report_ext}")
                if not report.is_file():
                    print(
                        f"ERROR: proton exited 0 and no report was generated "
                        f"({report})",
                        file=sys.stderr,
                    )
                    return 1
                exports = [_export_result("proton_hatchet", True, [report])]
                _print_proton_tree(report)
            else:
                prefix, report_ext, env_override = _profiler_invocation(
                    args.profiler, staging_stem, args.ncu_set
                )
                # The child records its runtime facts for the metadata
                # alongside the capture it runs.
                child_cmd += ["--meta-out", str(runtime_fragment)]
                cmd = prefix + child_cmd
                print(f"==> {' '.join(cmd)}")
                run_env = _merged_env(env_override)
                result = subprocess.run(cmd, env=run_env)
                report = Path(f"{staging_stem}.{report_ext}")
                if result.returncode != 0:
                    return _profiler_failed(args.profiler, result.returncode, report)
                # A profiler that exits 0 without producing a report (a silent
                # capture-window miss, for example) must fail explicitly.
                # Since this path is fresh, an earlier report cannot pass.
                if not report.is_file():
                    print(
                        f"ERROR: {args.profiler} exited 0 and no report was "
                        f"generated ({report})",
                        file=sys.stderr,
                    )
                    return 1
                if args.profiler == "ncu":
                    exports = _export_ncu(staging_stem, env_override)
                else:
                    exports = _export_nsys(staging_stem)

            published_exports, export_moves = _prepare_published_exports(
                exports, staging_stem, stem
            )
            staged_metadata = capture_dir / f"{stem.name}.json"
            _write_metadata(
                staging_stem,
                args,
                cases[args.case],
                original_kernel,
                kernel_sha256,
                exports=published_exports,
                destination=staged_metadata,
                report_path=Path(f"{stem}.{report_ext}"),
                runtime_fragment=runtime_fragment,
            )
            meta_path = _publish_capture(
                staging_stem,
                stem,
                report_ext,
                export_moves,
                args.profiler,
                staged_metadata,
            )

    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
