"""Helpers shared by the KernelAgent CLIs.

The CLIs (generate-kernel.py, profile-kernel.py,
experiment-kernels.py, report-experiment.py, enrich-code-diffs.py)
ship together in one directory — workspace/KernelAgent/scripts/ in the
repository, /usr/local/bin in the image — so a plain ``import ka_common``
resolves via sys.path[0] in every real invocation. They previously
duplicated these helpers with drifting semantics (two _atomic_write_json
implementations that differed on mode preservation and parent creation,
two CUDA probe scripts kept in sync by comment only); this module holds
the superset implementation of each, parameterized where a caller's
needs genuinely differ.

Stdlib-only by contract: the image's check_install gate asserts the
experiment and report CLIs import nothing beyond the standard library.
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

# Per-problem files (required in every problem directory) and the shared
# harness module every one of them imports (problems/_harness.py). Kept as
# one constant so every copy site — publication, isolated verification
# runs, experiment snapshots — agrees on what makes up "a problem".
PROBLEM_FILES = ("problem.txt", "test_fixed.py", "bench.py", "spec.py")
SHARED_HARNESS = "_harness.py"

# Default filesystem locations. Each CLI binds its
# PROBLEMS_HOME/RESULTS_MOUNT module attribute at import time, but the
# container's canonical paths are defined once here.
PROBLEMS_HOME_DEFAULT = "/workspace/user/problems"
RESULTS_MOUNT_DEFAULT = "/workspace/user/results"

# Saved kernel artifacts, keyed by impl. The optimized kernel is
# deliberately separate from final_kernel.py: generation replaces
# final_kernel.py, optimization replaces optimized_kernel.py, and each
# verification record only ever describes its own artifact. Kept as one
# table so a new impl (or a renamed record) cannot drift between the CLI
# that writes an artifact and the one that reads it.
IMPL_ARTIFACTS = {
    "kernel": {
        "kernel": "final_kernel.py",
        "record": "verify.json",
        "summary": "generate_result.json",
    },
    "optimized": {
        "kernel": "optimized_kernel.py",
        "record": "verify_optimized.json",
        "summary": "optimize_result.json",
    },
}

# Profilers the experiment runner and report understand, in the order the
# report presents them.
SUPPORTED_PROFILERS = ("ncu", "nsys", "proton")

# Report selection sentinel: render the newest capture of each profiler
# across every capture batch instead of one whole batch.
LATEST_PER_PROFILER = "latest-per-profiler"

# Record stems a user-supplied --label must never collide with: the
# impl-based names the CLIs themselves publish (verify.json /
# verify_optimized.json, profile report stems).
RESERVED_RECORD_STEMS = ("kernel", "optimized", "ref")

# Variant keys inside one problem of an experiment manifest: the
# generated baseline plus one "rounds_<N>" key per optimization budget.
# The scheme has one owner here — the experiment runner writes the keys,
# the report sorts and validates them, and add-baseline refuses labels
# that would collide with them.
GENERATED_VARIANT = "generated"

# The experiment manifest's filename and schema version, checked by
# writer and reader alike (a bump must fail at mutation time, not after
# all artifacts are written).
MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1

# LLM backends the CLIs accept, the model the anthropic backend falls
# back to when OPENAI_MODEL is unset, and the reasoning-effort levels
# the run-wide and per-stage knobs accept (forwarded to the container
# as KA_REASONING_EFFORT).
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-6"
REASONING_EFFORT_LEVELS = ("low", "medium", "high")


def infer_llm_backend(model: str | None) -> str:
    """The backend a model id routes to: claude-* ids mean the Anthropic
    API unless a user-set OPENAI_BASE_URL points at an OpenAI-compatible
    server (which serves every id, claude included); everything else is
    OpenAI-compatible. The entrypoint marks the public URL it supplies
    itself (KA_OPENAI_BASE_URL_DEFAULTED=1), which does not reroute a
    claude-* id. The entrypoint derives the run-wide LLM_BACKEND this way,
    and the experiment runner derives each stage's."""
    url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    defaulted = (
        os.environ.get("KA_OPENAI_BASE_URL_DEFAULTED") or ""
    ).strip() == "1"
    if (model or "").strip().startswith("claude-") and (
        not url or defaulted
    ):
        return "anthropic"
    return "openai"


def validate_reasoning_effort(value: str | None, source: str) -> str | None:
    """Normalize an optional reasoning-effort knob: the lowercase level,
    or None when empty. Anything else exits with a clear error naming
    the knob (source), so a typo fails the run instead of silently
    running at the wrong level."""
    level = (value or "").strip().lower() or None
    if level is not None and level not in REASONING_EFFORT_LEVELS:
        raise SystemExit(
            f"ERROR: {source} must be one of "
            f"{' / '.join(REASONING_EFFORT_LEVELS)} (got {value!r})"
        )
    return level


def safe_name_component(value: str, what: str) -> None:
    """The shared path-safety rule for every user-supplied name that
    becomes a filename component (a --label, an experiment id): a short
    string starting with a letter or digit, then letters, digits, '.',
    '_', '-'. Callers layer their own collision rules on top."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value):
        raise SystemExit(
            f"ERROR: {what} must be a safe filename component "
            f"(got {value!r})"
        )


def validate_label(label: str) -> None:
    """A user-supplied --label: a safe filename component (see
    safe_name_component) that never collides with the reserved
    impl-based record stems."""
    safe_name_component(label, "--label")
    if label in RESERVED_RECORD_STEMS:
        raise SystemExit(
            f"ERROR: --label {label!r} collides with a reserved impl-based "
            "record name; pass a distinguishing label"
        )


def print_banner(title: str, fields=()) -> None:
    """The ===/title banner shared by the CLIs; optional fields render as
    aligned ``Label: value`` lines."""
    print("=" * 72)
    print(title)
    print("=" * 72)
    if fields:
        width = max(len(label) for label, _ in fields) + 2
        for label, value in fields:
            print(f"{label + ':':<{width}}{value}")


def variant_budget(variant: str) -> int | None:
    """The optimization round budget encoded in a variant key
    (rounds_<N>); None for every other key, the generated baseline
    included."""
    match = re.fullmatch(r"rounds_([0-9]+)", variant)
    return int(match.group(1)) if match else None


def problem_dirs(home: Path) -> list[str]:
    """Problem directory names: every home subdirectory containing
    spec.py — the marker the Makefile derives KA_PROBLEMS from and the
    build gate checks. In directory order."""
    if not home.is_dir():
        return []
    return sorted(
        entry.name
        for entry in home.iterdir()
        if entry.is_dir() and (entry / "spec.py").is_file()
    )


def available_problems(home: Path) -> list[str]:
    """Problem names: every home subdirectory with a complete set of
    problem files, in directory order."""
    return [
        name
        for name in problem_dirs(home)
        if all((home / name / fname).is_file() for fname in PROBLEM_FILES)
    ]


def experiments_home(results_root: Path) -> Path:
    """Experiments root under a results mount."""
    return Path(results_root) / "experiments"


def resolve_experiment(results_root: Path, exp_id: str) -> Path:
    """An experiment by id (under the results mount) or by directory; the
    caller decides whether to resolve() the returned path."""
    candidate = Path(exp_id)
    if candidate.is_dir() and (candidate / MANIFEST_FILENAME).is_file():
        return candidate
    resolved = experiments_home(results_root) / exp_id
    if (resolved / MANIFEST_FILENAME).is_file():
        return resolved
    raise SystemExit(
        f"ERROR: no experiment {exp_id!r} under {experiments_home(results_root)}"
    )


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json_object(path: Path, default=None):
    """A JSON object read from disk, or default when the file is missing,
    unreadable, malformed, or not an object."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return document if isinstance(document, dict) else default


def find_bench_case(bench_data: dict | None, case_index: int) -> dict | None:
    """The selected case's entry from a parsed BENCH_JSON payload."""
    for entry in (bench_data or {}).get("cases") or []:
        if entry.get("case") == case_index:
            return entry
    return None


def _load_manifest(exp_dir: Path) -> dict:
    """manifest.json of an experiment, schema-checked. The check lives in
    this one loader so the writer (experiment-kernels.py) and the reader
    (report-experiment.py) agree: a schema bump fails at mutation time,
    not after all artifacts have been written."""
    path = exp_dir / MANIFEST_FILENAME
    if not path.is_file():
        raise SystemExit(f"ERROR: {path} not found (not an experiment?)")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise SystemExit(
            f"ERROR: {path} has schema version "
            f"{manifest.get('schema_version')!r}, expected "
            f"{MANIFEST_SCHEMA_VERSION}"
        )
    return manifest


@contextlib.contextmanager
def _exclusive_lock(path: Path):
    """Serialize writers behind an exclusive flock on path.

    flock() locks per open file description, so this blocks across
    processes sharing the filesystem. Callers wrap it under their own
    name and lock-file convention (per-problem results, per-experiment
    manifest)."""
    with open(path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_write_text(path: Path, text: str) -> None:
    """Publish text by atomic replacement.

    The replacement is written beneath the destination directory and
    renamed over it, so concurrent readers never observe a truncated file
    and a failed write leaves the previous one in place. Keeping the
    temporary file on the same filesystem makes the rename atomic.
    """
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        mode = None

    with tempfile.TemporaryDirectory(
        prefix=f".{path.name}.", dir=path.parent
    ) as tmp_dir:
        tmp_path = Path(tmp_dir) / path.name
        tmp_path.write_text(text, encoding="utf-8")
        if mode is not None:
            tmp_path.chmod(mode)
        os.replace(tmp_path, path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Publish a JSON document by atomic replacement.

    Creates missing parent directories (the experiment runner writes
    into freshly made experiment trees) and preserves an existing file's
    mode (generate-kernel.py's results artifacts keep theirs across
    rewrites); both behaviors come from the callers this unifies.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    _atomic_write_text(path, text)


def _load_module(name: str, path: Path, register: bool = False):
    """Load a Python module from an arbitrary file path.

    register also enters it in sys.modules under name — the experiment
    runner's report-module loading relies on the registration; the
    profiling CLIs keep their specs unregistered.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"ERROR: cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_spec(
    problems_home: Path,
    name: str,
    prefix: str = "spec_",
    register: bool = False,
):
    """A problem's spec module (case data, input construction, reference
    op) — see problems/<name>/spec.py. Torch-free at module level by the
    problem contract, so loading it never imports torch (let alone
    initializes CUDA) in this process.

    prefix namespaces the synthesized module name per CLI, so several
    CLIs loaded into one process never collide on a spec.
    """
    path = problems_home / name / "spec.py"
    if not path.is_file():
        raise SystemExit(f"ERROR: {path} not found")
    return _load_module(f"{prefix}{name}", path, register=register)


def _env_int(name: str, default: int) -> int:
    """Argparse default from an env var; unset or empty falls back.

    The Makefile forwards KA_* variables unquoted, so an empty override
    (`make ... KA_WORKERS=`) reaches the script as "" — that must mean
    "unset", not a ValueError at parser-construction time.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"ERROR: {name}='{raw}' is not an integer")


# Runs as `sys.executable -c <script>` inside a probe runner. Kept as a
# constant so the parent never evaluates it: only the child process
# imports torch and initializes CUDA.
_CUDA_PROBE_SCRIPT = """\
import json
import sys
import uuid as uuidlib


def _uuid_text(raw):
    # CUDA's 16-byte device UUID in nvidia-smi's GPU-<uuid> form (None
    # when torch reports it in a form we cannot parse). torch releases
    # have variously exposed it as raw or hex-encoded bytes, a
    # hyphenated string, or a torch._C._CUuuid object whose .bytes
    # attribute holds the 16 raw bytes as a list of ints.
    try:
        raw = getattr(raw, "bytes", raw)
        if isinstance(raw, (bytes, bytearray, memoryview, list, tuple)):
            data = bytes(raw)
            if len(data) == 16:
                return "GPU-" + str(uuidlib.UUID(bytes=data))
            raw = data.decode("ascii")
        if isinstance(raw, str):
            if raw.startswith("GPU-"):
                raw = raw[4:]
            return "GPU-" + str(uuidlib.UUID(raw))
    except (TypeError, ValueError, UnicodeDecodeError):
        pass
    return None


try:
    import torch

    devices = []
    if torch.cuda.is_available():
        for ordinal in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(ordinal)
            devices.append(
                {
                    "ordinal": ordinal,
                    "uuid": _uuid_text(getattr(props, "uuid", None)),
                    "name": props.name,
                    "total_memory": getattr(props, "total_memory", None),
                    "major": getattr(props, "major", None),
                    "minor": getattr(props, "minor", None),
                }
            )
    print("CUDA_JSON " + json.dumps(devices))
    # The versions ride the same child so a caller needing both the
    # device inventory and torch/triton versions pays one torch import,
    # not two. triton is guarded separately so a broken triton install
    # still records the torch and CUDA versions.
    versions = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": None,
    }
    try:
        import triton
        versions["triton"] = triton.__version__
    except Exception:
        pass
    print("VERSIONS_JSON " + json.dumps(versions))
except Exception as exc:
    print("CUDA probe failed: %s" % exc, file=sys.stderr)
    sys.exit(1)
"""


def last_marker_json(stdout: str, marker: str):
    """The JSON payload of the last stdout line starting with marker, or
    None when the line is missing or malformed. The one parser for the
    subprocess-to-parent record channel (CUDA_JSON, VERSIONS_JSON,
    BENCH_JSON)."""
    line = next(
        (
            line
            for line in reversed(stdout.splitlines())
            if line.startswith(marker)
        ),
        None,
    )
    if line is None:
        return None
    try:
        return json.loads(line[len(marker) :])
    except json.JSONDecodeError:
        return None


def _parse_cuda_probe_stdout(stdout: str) -> tuple[list | None, dict | None]:
    """(devices, versions) from a CUDA probe child's stdout: the last
    CUDA_JSON / VERSIONS_JSON marker lines, JSON-decoded. Either value is
    None when its marker line is missing or malformed; callers validate
    the payloads against their own shapes (the full inventory, or just
    logical device 0)."""
    devices = last_marker_json(stdout, "CUDA_JSON ")
    versions = last_marker_json(stdout, "VERSIONS_JSON ")
    if not isinstance(devices, list):
        devices = None
    if not isinstance(versions, dict):
        versions = None
    return devices, versions


def _smi_csv_rows(stdout: str, field_count: int) -> list[list[str]]:
    """Stripped CSV rows of exactly field_count fields from an
    nvidia-smi --format=csv,noheader query. csv handles nvidia-smi's
    quoting of GPU names containing commas; rows of any other width are
    dropped, and callers apply their own per-field validation."""
    return [
        row
        for row in (
            [field.strip() for field in row]
            for row in csv.reader(io.StringIO(stdout), skipinitialspace=True)
        )
        if len(row) == field_count
    ]


def _run_cuda_probe(
    env: dict | None = None, timeout: int = 120
) -> tuple[list | None, dict | None]:
    """(devices, versions) from one CUDA probe child: spawns
    sys.executable -c _CUDA_PROBE_SCRIPT with stdin devnulled and env
    (a copy of this process's environment by default — pass a stripped
    copy to see CUDA's unfiltered device ordering), then parses its
    stdout. (None, None) when the child cannot run or exits nonzero;
    either half is independently None when its marker line is missing or
    malformed.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", _CUDA_PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(os.environ) if env is None else env,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    if result.returncode != 0:
        return None, None
    return _parse_cuda_probe_stdout(result.stdout or "")


def _probe_device_entry(entry) -> dict | None:
    """One validated device entry from a parsed CUDA probe payload, as
    {ordinal, uuid, name, total_memory, major, minor}, or None when any
    field the probe contract defines is missing or malformed. The probe
    script fills every field on real payloads (older torch builds leave
    the optional ones None), so a failure here means a malformed child,
    not an unusual host."""
    if not isinstance(entry, dict):
        return None
    name = entry.get("name")
    uuid_text = entry.get("uuid")
    memory = entry.get("total_memory")
    major = entry.get("major")
    minor = entry.get("minor")
    ordinal = entry.get("ordinal")
    if not (
        isinstance(ordinal, int)
        and isinstance(name, str)
        and name
        and (uuid_text is None or isinstance(uuid_text, str))
        and (memory is None or isinstance(memory, int))
        and (major is None or isinstance(major, int))
        and (minor is None or isinstance(minor, int))
    ):
        return None
    return {
        "ordinal": ordinal,
        "uuid": uuid_text,
        "name": name,
        "total_memory": memory,
        "major": major,
        "minor": minor,
    }


def _tool_version_text(stdout: str) -> str | None:
    """The version string to record for a profiler CLI, from its
    ``--version`` stdout: prefers the line carrying the version number
    (ncu's first line is a banner). The parse lives here so the
    experiment runner — which probes each tool once per capture batch
    and hands the result to profile-kernel.py via KA_TOOL_VERSION_<TOOL>
    — records the same string a standalone profile run would."""
    lines = [line.strip() for line in stdout.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None
    for line in lines:
        match = re.search(r"[Vv]ersion\s*[=: ]\s*(\S+)", line)
        if match:
            if line == lines[0]:
                return line
            return f"{lines[0]} — {match.group(1)}"
    return lines[0]


def probe_tool_version(
    tool: str, env: dict[str, str] | None = None
) -> str | None:
    """A profiler CLI's own version string, parsed with the same
    line-preference rule everywhere. Best effort: any failure yields None
    instead of failing the profile run, which already proves the tool
    works. env overrides the child environment (ncu needs the system
    interpreter on PATH)."""
    try:
        result = subprocess.run(
            [tool, "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return _tool_version_text(result.stdout or "")


def _smi_inventory(
    fields: tuple[str, ...],
    timeout: int = 60,
) -> list[list[str]] | None:
    """CSV rows of an nvidia-smi --query-gpu query for the given fields
    (the callers want different columns), or None when nvidia-smi cannot
    run or reports no rows. nvidia-smi ignores CUDA_VISIBLE_DEVICES, so
    the rows always describe the whole host; callers apply their own
    per-field validation."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return _smi_csv_rows(result.stdout or "", len(fields)) or None
