#!/usr/bin/env python3
"""Validate the installed KernelAgent package inside the finished image.

Runs in the final stage of containerfiles/Dockerfile.cuda, from outside
any KernelAgent source checkout (there is none in the image), against
the regular wheel installed into the locked environment. Fails the
image build on any regression that would otherwise surface only at
generation or optimization time on a GPU.

Checked, in order:
  - the running interpreter is the workspace venv's Python
  - the KernelAgent distribution is a regular (non-editable) install
    under the venv's site-packages
  - the modules generate-kernel.py and the entrypoint touch import
    cleanly (agent, verification worker, optimization manager, NVIDIA
    platform components, LLM providers, GPU specs catalog, torch, triton)
  - the optimizer's per-test timeout knob (0007 compatibility patch) is
    present on OptimizationWorker and NvidiaVerifier
  - the KA_REASONING_EFFORT reasoning-level knob (0005 compatibility
    patch) gates and validates in the installed providers: unset
    preserves the model-name gating, set overrides it for any model id,
    and an invalid value raises
  - the prompt templates PromptManager loads by name all ship and parse
    as Jinja2
  - the NCU wrapper template ships and NCUWrapperFactory loads it
  - the RAG optimization database resolves through the installed
    kernel_perf_agent package and initializes from its code samples
  - the shipped catalog covers this repository's target laptop GPU (0006
    compatibility patch) and the desktop RTX 5090 (0008 compatibility
    patch) with real peak figures, and the KA_GPU_SPECS_FILE
    override (0003 compatibility patch) works as installed: a supplemental
    catalog entry written to a JSON file round-trips through get_gpu_specs()
    (filling gaps and overriding shipped entries), a malformed file degrades
    to a warning, and its figures render into the optimization and
    bottleneck-analysis prompts
  - the benchmarking subprocess script ships beside its module
  - no editable-install hooks or builder-source paths leaked into
    site-packages
  - the image's CLI helpers, profilers, and problem specs are in place

Deterministic beyond the image's own environment: no GPU, no LLM
credentials, no network calls (resource loading that would phone home
is exercised only down to construction-free paths). Dependency
consistency is checked separately by `uv pip check` in the same build
step. Run with the venv Python:

    /workspace/bin/python check_install.py --workspace /workspace \
        [--cli-dir DIR]
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import shutil
import subprocess
import sys
import sysconfig
from importlib import resources
from pathlib import Path

# The templates PromptManager loads by name (prompt_manager.py) and the
# code samples OptHierarchy.hard_initialize reads by name — the exact
# resource set the generation and optimization paths need at runtime.
TEMPLATE_NAMES = (
    "kernel_generation.j2",
    "kernel_optimization.j2",
    "kernel_refinement.j2",
    "reflexion_prompt.j2",
    "test_generation.j2",
    "triton_guidelines.j2",
)
CODE_SAMPLE_COUNT = 7

# A synthetic catalog entry driven through the 0003 patch's
# KA_GPU_SPECS_FILE mechanism: operators point the env at a JSON file
# mapping GPU names to specs dicts to give the optimizer real peaks for
# GPUs the shipped catalog does not cover (the Makefile mounts such a
# file for this repository's target laptop GPU). The figures below are
# deliberately synthetic — these checks pin the mechanism, not any
# particular GPU's numbers, which live in the operator's file.
SUPPLEMENTAL_GPU_NAME = "NVIDIA Synthetic Check GPU"
SUPPLEMENTAL_GPU_SPECS = {
    "name": SUPPLEMENTAL_GPU_NAME,
    "architecture": "Synthetic",
    "peak_fp32_tflops": 12.5,
    "peak_fp16_tflops": 25.0,
    "peak_bf16_tflops": 25.0,
    "peak_memory_bw_gbps": 300,
    "sm_count": 42,
    "max_threads_per_sm": 1024,
    "l1_cache_kb": 64,
    "l2_cache_mb": 6,
    "memory_gb": 8,
    "memory_type": "GDDR6",
}
# Any name absent from both the shipped catalog and the supplemental
# file (checked to keep upstream's contract)
UNKNOWN_GPU_NAME = "NVIDIA Not A Real GPU"

# The 0006 patch adds this repository's target laptop GPU to the shipped
# catalog. Its key is the exact CUDA/nvidia-smi device name, because
# get_gpu_specs() matches names exactly.
LAPTOP_GPU_NAME = "NVIDIA RTX 5000 Ada Generation Laptop GPU"

# The 0008 patch adds the desktop RTX 5090 to the shipped catalog. Its key
# includes "GeForce" because that is the exact CUDA/nvidia-smi device name
# and get_gpu_specs() matches names exactly.
DESKTOP_GPU_NAME = "NVIDIA GeForce RTX 5090"

# Loads one experiment CLI in a fresh interpreter. argv: <cli_dir> <cli_path>.
# -I ignores PYTHONPATH, the cwd, and the user site, while the venv's own
# site-packages stay visible. The import must happen in this child: the
# parent's core_imports check has already loaded torch and triton into
# sys.modules, so an in-process snapshot cannot prove stdlib-only.
_CLI_ISOLATION_PROBE = r"""
import importlib.util
import sys

cli_dir, cli_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, cli_dir)
spec = importlib.util.spec_from_file_location("check_cli", cli_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if not hasattr(module, "main"):
    raise SystemExit(f"{cli_path} has no main()")
bad = sorted(
    name for name in sys.modules
    if name == "torch" or name.startswith(("torch.", "triton"))
)
if bad:
    raise SystemExit(f"{cli_path} imports torch/triton: {bad}")
"""


def _publish_supplemental_specs() -> str:
    """Write SUPPLEMENTAL_GPU_SPECS (plus one entry overriding a shipped
    catalog GPU, to pin the documented precedence) to a temp file and
    point KA_GPU_SPECS_FILE at it, the way the Makefile does for a real
    operator file. Returns the overridden catalog GPU's name."""
    import json
    import os
    import tempfile

    from kernel_perf_agent.kernel_opt.diagnose_prompt import gpu_specs_database

    catalog = gpu_specs_database.GPU_SPECS_DATABASE
    catalog_name = next(iter(catalog))
    override = dict(catalog[catalog_name])
    override["sm_count"] = 12345
    fd, path = tempfile.mkstemp(suffix=".json", prefix="ka_gpu_specs_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(
            {SUPPLEMENTAL_GPU_NAME: SUPPLEMENTAL_GPU_SPECS,
             catalog_name: override},
            fh,
        )
    os.environ["KA_GPU_SPECS_FILE"] = path
    return catalog_name


def _clear_supplemental_specs() -> None:
    import os

    path = os.environ.pop("KA_GPU_SPECS_FILE", None)
    if path is not None:
        try:
            os.unlink(path)
        except OSError:
            pass


def _captured_gpu_specs_warnings(call):
    """Run call() with the gpu_specs logger's records captured, so checks
    can assert on warnings without dirtying the build log."""
    import logging

    logger = logging.getLogger(
        "kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs")
    records: list = []
    handler = logging.Handler()
    handler.emit = records.append
    logger.addHandler(handler)
    try:
        result = call()
    finally:
        logger.removeHandler(handler)
    return result, records


FAILURES: list[str] = []


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"PASS {name}")
    except Exception as exc:  # noqa: BLE001 - report and keep checking
        FAILURES.append(name)
        print(f"FAIL {name}: {type(exc).__name__}: {exc}", file=sys.stderr)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"),
                        help="the image's venv/workspace directory")
    parser.add_argument("--cli-dir", type=Path, default=Path("/usr/local/bin"),
                        help="directory holding the image's CLI helpers")
    parser.add_argument("--problems-dir", type=Path,
                        default=Path("/workspace/user/problems"),
                        help="baked problem specs directory")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    site_packages = Path(sysconfig.get_paths()["purelib"])

    def venv_interpreter() -> None:
        require(sys.prefix == str(workspace),
                f"running under {sys.prefix}, expected the {workspace} venv")
        require(Path(sys.base_prefix) != Path(sys.prefix),
                "not running inside a virtual environment")

    check("workspace venv interpreter", venv_interpreter)

    def dist_metadata() -> None:
        dist = importlib.metadata.distribution("KernelAgent")
        site = Path(dist.locate_file("")).resolve()
        require(site == site_packages,
                f"KernelAgent installed at {site}, expected {site_packages}")
        names = [str(f) for f in (dist.files or [])]
        leaked = [n for n in names
                  if "editable" in n.lower() or n.endswith(".egg-link")]
        require(not leaked, f"editable-install artifacts shipped: {leaked}")

    check("regular install under site-packages", dist_metadata)

    def core_imports() -> None:
        import torch  # noqa: F401
        import triton  # noqa: F401
        import triton_kernel_agent
        from triton_kernel_agent import (  # noqa: F401
            TritonKernelAgent,
            VerificationWorker,
        )
        from triton_kernel_agent.worker import VerificationWorker  # noqa: F401
        from triton_kernel_agent.opt_manager import OptimizationManager  # noqa: F401
        from triton_kernel_agent.platform.nvidia import (  # noqa: F401
            NvidiaBenchmarker,
            NvidiaBottleneckAnalyzer,
            NvidiaRAGPrescriber,
            NvidiaVerifier,
            NvidiaWorkerRunner,
        )
        from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import (  # noqa: F401
            get_gpu_specs,
        )
        from utils.providers import get_model_provider  # noqa: F401

    check("core imports (agent, optimizer, providers, torch, triton)",
          core_imports)

    def optimizer_timeout_knob() -> None:
        import inspect

        from triton_kernel_agent.opt_worker import OptimizationWorker
        from triton_kernel_agent.platform.nvidia import NvidiaVerifier

        require(
            "test_timeout_s"
            in inspect.signature(OptimizationWorker.__init__).parameters,
            "0007 patch is missing or stale: OptimizationWorker has no "
            "test_timeout_s argument",
        )
        require(
            "test_timeout_s"
            in inspect.signature(NvidiaVerifier.__init__).parameters,
            "0007 patch is missing or stale: NvidiaVerifier has no "
            "test_timeout_s argument",
        )

    check("optimizer test-timeout knob (0007 patch)", optimizer_timeout_knob)

    def reasoning_effort_knob() -> None:
        import os

        from utils.providers.base import reasoning_effort_from_env
        from utils.providers.openai_base import OpenAICompatibleProvider

        class _Stub(OpenAICompatibleProvider):
            name = "stub"

            def _initialize_client(self) -> None:
                pass

            def get_response(self, *args, **kwargs):
                raise NotImplementedError

        provider = _Stub(api_key_env="UNUSED")
        messages = [{"role": "user", "content": "ping"}]
        saved = os.environ.pop("KA_REASONING_EFFORT", None)
        try:
            # Unset preserves the model-name gating exactly (0005 patch):
            # gpt-6.1-sol gets no reasoning_effort, gpt-5* keeps auto-high
            params = provider._build_api_params("gpt-6.1-sol", messages)
            require("reasoning_effort" not in params,
                    f"gpt-6.1-sol unexpectedly gated: {params}")
            params = provider._build_api_params("gpt-5.2", messages)
            require(params.get("reasoning_effort") == "high",
                    f"gpt-5.2 lost its auto-high: {params}")
            # Set overrides the gating for any model id, and the shared
            # helper (also used by the Anthropic provider) agrees
            os.environ["KA_REASONING_EFFORT"] = "low"
            params = provider._build_api_params("gpt-6.1-sol", messages)
            require(params.get("reasoning_effort") == "low", f"{params}")
            require(reasoning_effort_from_env() == "low",
                    "helper disagrees with the provider")
            # Invalid values fail loudly instead of silently misspelling
            os.environ["KA_REASONING_EFFORT"] = "ultra"
            try:
                provider._build_api_params("gpt-6.1-sol", messages)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid KA_REASONING_EFFORT accepted")
        finally:
            os.environ.pop("KA_REASONING_EFFORT", None)
            if saved is not None:
                os.environ["KA_REASONING_EFFORT"] = saved

    check("KA_REASONING_EFFORT reasoning-level knob (0005 patch)",
          reasoning_effort_knob)

    def templates() -> None:
        import jinja2

        base = resources.files("triton_kernel_agent") / "templates"
        found = sorted(p.name for p in base.iterdir()
                       if p.name.endswith(".j2"))
        for name in TEMPLATE_NAMES:
            require(name in found, f"prompt template {name} not shipped "
                                   f"(found {found})")
        for name in TEMPLATE_NAMES:
            jinja2.Environment().parse((base / name).read_text())

    check("prompt templates ship and parse", templates)

    def ncu_wrapper() -> None:
        import logging

        import jinja2
        from triton_kernel_agent.opt_worker_component.profiling import (
            NCUWrapperFactory,
        )

        template_dir = Path(
            importlib.import_module(
                "triton_kernel_agent.opt_worker_component.profiling"
            ).__file__
        ).parent
        template = template_dir / "ncu_wrapper_template.j2"
        require(template.is_file(), f"missing {template}")
        jinja2.Environment().parse(template.read_text())
        factory = NCUWrapperFactory(logging.getLogger("check_install"))
        _ = factory.template  # cached_property: reads and parses the file

    check("NCU wrapper template ships and loads", ncu_wrapper)

    def rag_database() -> None:
        import kernel_perf_agent
        from kernel_perf_agent.kernel_opt.database.base import OptHierarchy

        # Same package-relative resolution the patched RAGPrescriber uses
        db = (Path(kernel_perf_agent.__file__).resolve().parent
              / "kernel_opt" / "database" / "code_samples")
        require(db.is_dir(), f"RAG database directory missing: {db}")
        samples = sorted(p.name for p in db.glob("*.py"))
        require(len(samples) == CODE_SAMPLE_COUNT,
                f"expected {CODE_SAMPLE_COUNT} code samples, found {samples}")
        hierarchy = OptHierarchy()
        hierarchy.hard_initialize(db)
        require(hierarchy.root is not None and hierarchy.root.opt_children,
                "optimization hierarchy did not initialize")

    check("RAG optimization database resolves and initializes", rag_database)

    def gpu_specs_catalog() -> None:
        from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import (
            get_gpu_specs,
        )

        # The shipped catalog must cover this repository's target laptop
        # GPU (0006 patch) before any supplemental file is published, so
        # a missing, misspelled, or incomplete entry fails the build.
        shipped, records = _captured_gpu_specs_warnings(
            lambda: get_gpu_specs(LAPTOP_GPU_NAME))
        require(shipped is not None and not records,
                f"{LAPTOP_GPU_NAME!r} missing from the shipped catalog; "
                "the 0006 patch is missing or stale")
        require(
            shipped.get("peak_memory_bw_gbps") == 576
            and shipped.get("sm_count") == 76,
            "shipped laptop GPU specs are incomplete or unexpected",
        )

        catalog_name = _publish_supplemental_specs()
        try:
            # The supplemental entry resolves through KA_GPU_SPECS_FILE
            # with every figure intact and no warning — a warning would
            # mean the name-only fallback took over.
            specs, records = _captured_gpu_specs_warnings(
                lambda: get_gpu_specs(SUPPLEMENTAL_GPU_NAME))
            require(specs is not None,
                    f"{SUPPLEMENTAL_GPU_NAME!r} did not resolve through "
                    "KA_GPU_SPECS_FILE; the 0003 patch is missing or stale")
            for key, value in SUPPLEMENTAL_GPU_SPECS.items():
                require(specs.get(key) == value,
                        f"supplemental {key} is {specs.get(key)!r}, "
                        f"expected {value!r}")
            require(not records,
                    f"supplemental lookup warned: "
                    f"{[r.getMessage() for r in records]}")

            # Supplemental entries take precedence over the shipped
            # catalog (correcting figures as well as filling gaps)
            require(
                get_gpu_specs(catalog_name)["sm_count"] == 12345,
                "supplemental entry did not override the shipped catalog",
            )

            # get_gpu_specs returns a mutable copy, not the file's dict
            sm_count = SUPPLEMENTAL_GPU_SPECS["sm_count"]
            specs["sm_count"] = 0
            require(
                get_gpu_specs(SUPPLEMENTAL_GPU_NAME)["sm_count"] == sm_count,
                "mutating a lookup result changed the supplemental entry")

            # A malformed operator file must degrade to a warning, not
            # crash optimization: the entry no longer resolves and the
            # mechanism says why.
            import os

            Path(os.environ["KA_GPU_SPECS_FILE"]).write_text(
                "{not json", encoding="utf-8")
            missing, records = _captured_gpu_specs_warnings(
                lambda: get_gpu_specs(SUPPLEMENTAL_GPU_NAME))
            require(missing is None,
                    "malformed KA_GPU_SPECS_FILE still resolved the entry")
            require(
                any("KA_GPU_SPECS_FILE" in r.getMessage() for r in records),
                "malformed KA_GPU_SPECS_FILE stopped warning",
            )
        finally:
            _clear_supplemental_specs()

        # Unknown GPUs keep upstream's None contract and still warn
        missing, records = _captured_gpu_specs_warnings(
            lambda: get_gpu_specs(UNKNOWN_GPU_NAME))
        require(missing is None,
                "unknown GPU lookup must return None (0003 fallback)")
        require(any("Unknown GPU" in r.getMessage() for r in records),
                "unknown GPU lookup stopped warning")

    check("KA_GPU_SPECS_FILE supplemental specs resolve (0003 patch)",
          gpu_specs_catalog)

    def gpu_specs_desktop_5090() -> None:
        from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import (
            get_gpu_specs,
        )

        # The shipped catalog must cover the desktop RTX 5090 (0008 patch)
        # under its exact device name, so a missing, misspelled, or
        # incomplete entry fails the build.
        specs, records = _captured_gpu_specs_warnings(
            lambda: get_gpu_specs(DESKTOP_GPU_NAME))
        require(specs is not None and not records,
                f"{DESKTOP_GPU_NAME!r} missing from the shipped catalog; "
                "the 0008 patch is missing or stale")
        require(specs.get("name") == DESKTOP_GPU_NAME,
                f"shipped RTX 5090 name is {specs.get('name')!r}")
        require(
            specs.get("architecture") == "Blackwell"
            and specs.get("peak_memory_bw_gbps") == 1792
            and specs.get("sm_count") == 170,
            "shipped RTX 5090 specs are incomplete or unexpected",
        )

    check("shipped desktop RTX 5090 specs (0008 patch)",
          gpu_specs_desktop_5090)

    def gpu_specs_prompts() -> None:
        from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import (
            get_gpu_specs,
        )
        from kernel_perf_agent.kernel_opt.diagnose_prompt.judger_prompt import (
            RooflineResult,
            build_bottleneck_prompt,
        )
        from triton_kernel_agent.prompt_manager import PromptManager

        _publish_supplemental_specs()
        try:
            specs = get_gpu_specs(SUPPLEMENTAL_GPU_NAME)
            require(specs is not None,
                    "supplemental lookup failed (prior check)")

            # The optimization prompt's TARGET GPU section carries the
            # specs (the 0003-patched template renders "unknown" for
            # absent keys)
            prompt = PromptManager().render_kernel_optimization_prompt(
                problem_description="check problem",
                kernel_code="@triton.jit\ndef kernel_function(): pass",
                gpu_specs=specs,
                roofline={
                    "bottleneck": "memory",
                    "compute_sol_pct": 12.0,
                    "memory_sol_pct": 61.0,
                    "efficiency_pct": 61.0,
                    "headroom_pct": 39.0,
                    "at_roofline": False,
                    "uses_tensor_cores": False,
                    "warnings": [],
                },
                category="memory",
                summary="summary",
                reasoning="reasoning",
                root_cause={"cause": "cause",
                            "evidence": [{"metric": "m", "value": 1.0}]},
                recommended_fix={"fix": "fix", "rationale": "rationale"},
                pytorch_baseline_ms=1.0,
            )
            for line in (
                f"- GPU: {SUPPLEMENTAL_GPU_NAME}",
                f"- Architecture: {SUPPLEMENTAL_GPU_SPECS['architecture']}",
                f"- Peak Memory Bandwidth: "
                f"{SUPPLEMENTAL_GPU_SPECS['peak_memory_bw_gbps']} GB/s",
                f"- Peak FP32: "
                f"{SUPPLEMENTAL_GPU_SPECS['peak_fp32_tflops']} TFLOPS",
                f"- Peak FP16: "
                f"{SUPPLEMENTAL_GPU_SPECS['peak_fp16_tflops']} TFLOPS",
                f"- Peak BF16: "
                f"{SUPPLEMENTAL_GPU_SPECS['peak_bf16_tflops']} TFLOPS",
                f"- SM Count: {SUPPLEMENTAL_GPU_SPECS['sm_count']}",
                f"- Memory: {SUPPLEMENTAL_GPU_SPECS['memory_gb']} GB "
                f"{SUPPLEMENTAL_GPU_SPECS['memory_type']}",
            ):
                require(line in prompt, f"optimization prompt lost {line!r}")
            require("unknown" not in prompt,
                    "optimization prompt rendered 'unknown' for catalog specs")

            # The bottleneck-analysis prompt's specs section carries them
            # too
            bprompt = build_bottleneck_prompt(
                kernel_code="def kernel_function(): pass",
                ncu_metrics={"sm__cycles_active.avg": 1000.0},
                roofline=RooflineResult(
                    compute_sol_pct=12.0,
                    memory_sol_pct=61.0,
                    efficiency_pct=61.0,
                    at_roofline=False,
                    headroom_pct=39.0,
                    bottleneck="memory",
                    uses_tensor_cores=False,
                    warnings=[],
                ),
                gpu_specs=specs,
            )
            for line in (
                f"- Name: {SUPPLEMENTAL_GPU_NAME}",
                f"- Peak Memory Bandwidth: "
                f"{SUPPLEMENTAL_GPU_SPECS['peak_memory_bw_gbps']} GB/s",
                f"- Peak FP32 Performance: "
                f"{SUPPLEMENTAL_GPU_SPECS['peak_fp32_tflops']} TFLOPS",
                f"- SM Count: {SUPPLEMENTAL_GPU_SPECS['sm_count']}",
            ):
                require(line in bprompt, f"bottleneck prompt lost {line!r}")
        finally:
            _clear_supplemental_specs()

    check("supplemental GPU specs reach the optimization and bottleneck "
          "prompts", gpu_specs_prompts)

    def benchmarking() -> None:
        module = importlib.import_module(
            "triton_kernel_agent.opt_worker_component.benchmarking.benchmark"
        )
        script = Path(module.__file__).parent / "kernel_subprocess.py"
        require(script.is_file(), f"missing {script}")

    check("benchmarking subprocess script ships", benchmarking)

    def no_leaked_paths() -> None:
        for pth in site_packages.glob("*.pth"):
            text = pth.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                require("/tmp/" not in line and "/ka-src" not in line,
                        f"{pth.name} references a builder path: {line!r}")
        require(not list(site_packages.glob("__editable__*")),
                "editable-install .pth files present in site-packages")
        require(not list(site_packages.glob("*.egg-link")),
                "egg-link files present in site-packages")

    check("no editable hooks or builder paths in site-packages",
          no_leaked_paths)

    def clis() -> None:
        for name in ("generate-kernel.py", "profile-kernel.py",
                     "experiment-kernels.py", "report-experiment.py",
                     "verify-llm.py"):
            cli = args.cli_dir / name
            require(cli.is_file() and cli.stat().st_mode & 0o111,
                    f"missing or non-executable {cli}")
        # The shared helper module the CLIs import beside themselves; a
        # COPY-list regression must fail the build here, not at runtime.
        require((args.cli_dir / "ka_common.py").is_file(),
                "missing ka_common.py")
        for tool in ("ncu", "nsys"):
            require(shutil.which(tool) is not None, f"{tool} not on PATH")
        proton = workspace / "bin" / "proton"
        require(proton.is_file(), f"missing {proton}")

    check("CLI helpers and profilers in place", clis)

    def experiment_clis() -> None:
        for name in ("experiment-kernels.py", "report-experiment.py"):
            path = args.cli_dir / name
            # Both must stay standard-library-only: the report runs on
            # hosts with no torch/triton installed, and the runner must
            # never initialize CUDA ahead of the optimizer's forks. The
            # child imports the CLI and fails on torch/triton; the parent
            # only turns that into a build failure.
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    _CLI_ISOLATION_PROBE,
                    str(args.cli_dir),
                    str(path),
                ],
                capture_output=True,
                text=True,
            )
            require(
                proc.returncode == 0,
                f"{name} is not stdlib-only/importable: "
                f"{(proc.stderr or proc.stdout).strip()}",
            )

    check("experiment and report CLIs import stdlib-only", experiment_clis)

    def problems() -> None:
        import importlib.util

        # Discovery and completeness rule from the installed ka_common
        # (single-sourced with the CLIs and the Makefile's KA_PROBLEMS
        # derivation): every problems/ subdirectory holding spec.py must
        # carry the full PROBLEM_FILES set, so a partially baked problem
        # fails the build here.
        spec = importlib.util.spec_from_file_location(
            "check_ka_common", args.cli_dir / "ka_common.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        discovered = module.problem_dirs(args.problems_dir)
        require(discovered,
                f"no problems discovered under {args.problems_dir}")
        for problem in discovered:
            for fname in module.PROBLEM_FILES:
                path = args.problems_dir / problem / fname
                require(path.is_file(), f"missing problem file {path}")

    check("problem specs baked", problems)

    if FAILURES:
        print(f"ERROR: {len(FAILURES)} check(s) failed: "
              f"{', '.join(FAILURES)}", file=sys.stderr)
        return 1
    print("All installed-package checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
