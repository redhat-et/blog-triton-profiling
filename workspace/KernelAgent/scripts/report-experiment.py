#!/usr/bin/env python3
"""Render a Markdown (and JSON) comparison report for a saved experiment.

Works entirely offline from the experiment's manifest and saved
artifacts — benchmark records, optimizer summaries, profiler captures
and their CSV exports — with no GPU, torch, triton, or LLM access, so it
runs on the host with a plain Python 3 standard library:

  - benchmark tables come from the verification records (the fixed
    bench.py timing; the optimizer's own acceptance timings are shown
    separately where they exist)
  - Nsight Compute tables come from the .details.csv export
    profile-kernel.py writes beside each capture
  - Nsight Systems tables come from the nsys stats CSV exports
  - Proton tables come from the JSON .hatchet capture
  - kernel sources are diffed pairwise into reports/source-diffs/
  - external baselines added with 'experiment-kernels.py add-baseline'
    render beside the generated and optimized variants, marked as
    external with the source they were archived from, and diff against
    the other variants like any kernel

Profiler timings are never merged into one interchangeable latency
column: each tool keeps its own section, with its measurement settings
and aggregation stated beside the table. Missing metrics render as "—".

Usage:
  report-experiment.py <experiment> [--results-root DIR] [--out PATH]
                       [--capture-batch ID] [--format markdown|json]

  <experiment>  experiment id under <results-root>/experiments, or an
                experiment directory directly
  --results-root  overrides RESULTS_MOUNT (default from the environment
                or /workspace/user/results)
  --out         write the Markdown somewhere other than
                <experiment>/reports/comparison.md (links are computed
                relative to the destination)
  --capture-batch  report a specific capture batch (default: the latest)
  --format json  print the normalized report data as JSON and exit

Examples:
  report-experiment.py webinar-v1
  report-experiment.py webinar-v1 --out slides/kernel-comparison.md
  report-experiment.py webinar-v1 --format json
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
from datetime import datetime
from pathlib import Path

from ka_common import (
    GENERATED_VARIANT,
    RESULTS_MOUNT_DEFAULT,
    SUPPORTED_PROFILERS,
    _atomic_write_json,
    _atomic_write_text,
    _load_manifest,
    resolve_experiment,
    variant_budget,
)

# The comparison.json document's own schema version — a separate contract
# from the manifest's (ka_common.MANIFEST_SCHEMA_VERSION, checked by the
# shared _load_manifest both CLIs call).
SCHEMA_VERSION = 1

RESULTS_MOUNT = Path(
    os.environ.get("RESULTS_MOUNT", RESULTS_MOUNT_DEFAULT)
)

# NCU details-page metric names (as they appear in --csv --page details)
# mapped to normalized keys. Duration units are resolved per row from the
# CSV's unit column, never assumed.
NCU_METRICS = {
    "Duration": "duration",
    "Compute (SM) Throughput": "sm_pct",
    "Memory Throughput": "mem_pct",
    "DRAM Throughput": "dram_pct",
    "Achieved Occupancy": "occupancy_pct",
    "Registers Per Thread": "regs",
    "Elapsed Cycles": "cycles",
    "Grid Size": "grid",
    "Block Size": "block",
}

_DURATION_UNITS = {
    "ns": 1e-3,
    "nsecond": 1e-3,
    "nanosecond": 1e-3,
    "nsecond(s)": 1e-3,
    "us": 1.0,
    "usecond": 1.0,
    "microsecond": 1.0,
    "ms": 1e3,
    "msecond": 1e3,
    "millisecond": 1e3,
    "s": 1e6,
    "second": 1e6,
}


def _num(value, digits=3):
    """Format a number for a table cell; None or unparseable becomes —."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _effort_suffix(effort):
    """A stage's reasoning effort as a table-cell suffix ('' when unset,
    so reports from before the knob exist render unchanged)."""
    return f" / {effort}" if effort else ""


def _x(value):
    """Format a speedup ratio like 3.48x; None becomes —."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}x"
    except (TypeError, ValueError):
        return "—"


def _norm_header(header: str) -> str:
    return re.sub(r"\s+", "", header or "").lower()


def _to_float(text):
    if text is None:
        return None
    try:
        return float(str(text).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _to_int(text):
    number = _to_float(text)
    return int(number) if number is not None else None


def _duration_us(value, unit):
    """A duration metric converted to microseconds via its unit column."""
    number = _to_float(value)
    if number is None:
        return None
    factor = _DURATION_UNITS.get((unit or "").strip().lower())
    if factor is None:
        # Unknown unit: assume the ncu default (ns) rather than guess a
        # different scale silently.
        factor = 1e-3
    return number * factor


# ---------------------------------------------------------------------------
# NCU CSV parsing


def _read_csv_rows(path: Path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        return [row for row in csv.reader(fh) if row]


def _parse_ncu_details(path: Path) -> list[dict]:
    """Per-launch records from --csv --page details: one row per
    (launch, section, metric), grouped here into one record per launch."""
    rows = _read_csv_rows(path)
    header_idx = None
    for i, row in enumerate(rows):
        if "metricname" in {_norm_header(c) for c in row}:
            header_idx = i
            break
    if header_idx is None:
        return []
    header = {_norm_header(c): j for j, c in enumerate(rows[header_idx])}
    id_col = header.get("id")
    kernel_col = header.get("kernelname")
    metric_col = header.get("metricname")
    unit_col = header.get("metricunit")
    value_col = header.get("metricvalue")
    if metric_col is None or value_col is None or kernel_col is None:
        return []
    launches: dict[str, dict] = {}
    for row in rows[header_idx + 1 :]:
        try:
            metric = row[metric_col]
            value = row[value_col]
            kernel = row[kernel_col]
        except IndexError:
            continue
        key = (
            row[id_col]
            if id_col is not None and id_col < len(row)
            else f"{kernel}:{len(launches)}"
        )
        if key not in launches:
            launches[key] = {"kernel": kernel, "metrics": {}}
        unit = (
            row[unit_col]
            if unit_col is not None and unit_col < len(row)
            else None
        )
        normalized = NCU_METRICS.get(metric)
        if normalized is None:
            continue
        if normalized == "duration":
            launches[key]["metrics"][normalized] = _duration_us(value, unit)
        elif normalized == "regs":
            launches[key]["metrics"][normalized] = _to_int(value)
        else:
            launches[key]["metrics"][normalized] = _to_float(value)
    return list(launches.values())


def _mean_metric(group: list[dict], key: str):
    """Mean of one metric across a kernel's launches, ignoring launches
    that did not report it (None when none did)."""
    values = [
        g["metrics"].get(key)
        for g in group
        if g["metrics"].get(key) is not None
    ]
    return sum(values) / len(values) if values else None


def _aggregate_launches(launches: list[dict]) -> list[dict]:
    """Group per-launch records by kernel name and aggregate: counts,
    mean and total duration, means for the percentage metrics, and the
    first launch's launch-shape metrics."""
    if not launches:
        return []
    grouped: dict[str, list[dict]] = {}
    for launch in launches:
        grouped.setdefault(launch["kernel"], []).append(launch)
    summaries = []
    for kernel, group in grouped.items():
        durations = [
            g["metrics"].get("duration")
            for g in group
            if g["metrics"].get("duration") is not None
        ]
        summaries.append(
            {
                "kernel": kernel,
                "launches": len(group),
                "duration_avg_us": (
                    sum(durations) / len(durations) if durations else None
                ),
                "duration_total_us": sum(durations) if durations else None,
                "sm_pct": _mean_metric(group, "sm_pct"),
                "mem_pct": _mean_metric(group, "mem_pct"),
                "dram_pct": _mean_metric(group, "dram_pct"),
                "occupancy_pct": _mean_metric(group, "occupancy_pct"),
                "regs": group[0]["metrics"].get("regs"),
                "grid": group[0]["metrics"].get("grid"),
                "block": group[0]["metrics"].get("block"),
                "cycles": _mean_metric(group, "cycles"),
            }
        )
    summaries.sort(key=lambda s: -(s["duration_total_us"] or 0.0))
    return summaries


# ---------------------------------------------------------------------------
# nsys stats CSV parsing


def _parse_nsys_table(path: Path) -> list[dict]:
    """Rows of an nsys stats CSV export (cuda_gpu_kern_sum, cuda_api_sum):
    the header row is located by its Name column, and columns are matched
    by normalized header text so minor version differences in spacing or
    units inside the header do not break the parse."""
    rows = _read_csv_rows(path)
    header_idx = None
    for i, row in enumerate(rows):
        headers = {_norm_header(c) for c in row}
        if "name" in headers and (
            "instances" in headers or "numcalls" in headers
        ):
            header_idx = i
            break
    if header_idx is None:
        return []
    header = rows[header_idx]
    mapping = {}
    for j, cell in enumerate(header):
        norm = _norm_header(cell)
        if norm == "name":
            mapping["name"] = j
        elif norm == "instances" or norm == "numcalls":
            mapping["instances"] = j
        elif norm.startswith("totaltime"):
            mapping["total_ns"] = j
        elif norm.startswith("avg"):
            mapping["avg_ns"] = j
        elif norm.startswith("time"):
            mapping["time_pct"] = j
    if "name" not in mapping:
        return []
    parsed = []
    for row in rows[header_idx + 1 :]:
        try:
            name = row[mapping["name"]]
        except IndexError:
            continue
        if not name.strip():
            continue

        def get(key, row=row):
            j = mapping.get(key)
            if j is None or j >= len(row):
                return None
            return row[j].strip()

        total_ns = _to_float(get("total_ns"))
        avg_ns = _to_float(get("avg_ns"))
        parsed.append(
            {
                "name": name,
                "instances": _to_float(get("instances")),
                "total_ms": total_ns / 1e6 if total_ns is not None else None,
                "avg_us": avg_ns / 1e3 if avg_ns is not None else None,
                "time_pct": _to_float(get("time_pct")),
            }
        )
    return parsed


# ---------------------------------------------------------------------------
# Proton hatchet parsing


def _walk_hatchet(node, rows):
    if not isinstance(node, dict):
        return
    frame = node.get("frame")
    if not isinstance(frame, dict):
        return
    entry = {
        "name": frame.get("name"),
        "type": frame.get("type"),
        "metrics": {},
    }
    metrics = node.get("metrics")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                entry["metrics"][str(key)] = value
    rows.append(entry)
    children = node.get("children")
    if isinstance(children, list):
        for child in children:
            _walk_hatchet(child, rows)


def _is_measured_kernel(row: dict) -> bool:
    """Whether an entry looks like a measured kernel rather than a scope
    node (ROOT/device/context/stream aggregations, and the trailing
    device-info object) that carries summed or zero times."""
    if not row["name"] or row["name"] == "ROOT":
        return False
    if row["metrics"].get("count", 1) == 0:
        return False
    kind = str(row["type"]).lower() if row["type"] is not None else None
    return (
        kind is None
        or "kernel" in kind
        or not kind.startswith(("device", "context", "stream"))
    )


def _parse_hatchet(path: Path) -> list[dict]:
    """Entries from a Proton .hatchet capture. The exact schema is
    version-dependent, so the parser is deliberately tolerant: a JSON
    document (list of trees, or an object with a tree/list field) whose
    nodes carry frame names and numeric metrics. Anything else yields an
    empty list and the report shows —."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data.get("tree") or data.get("graph") or []
    if not isinstance(data, list):
        return []
    rows: list[dict] = []
    for node in data:
        _walk_hatchet(node, rows)
    return [row for row in rows if _is_measured_kernel(row)]


# ---------------------------------------------------------------------------
# Report data assembly


def _export_paths(capture: dict, name: str) -> list[Path]:
    """Paths recorded for one named export of a capture (details CSV,
    nsys stats, hatchet)."""
    paths = []
    for export in capture.get("exports") or []:
        if export.get("name") == name and export.get("status") == "ok":
            paths.extend(Path(p) for p in export.get("paths", []))
    return paths


def _variant_sort_key(variant: str):
    budget = variant_budget(variant)
    if variant == GENERATED_VARIANT:
        return (0, 0)
    if budget is not None:
        return (1, budget)
    return (2, 0)


def _capture_summary(exp_dir: Path, capture: dict, profiler: str) -> dict:
    """Normalized per-capture data: parsed metric tables for the
    profiler's exports. Parse failures degrade to empty tables (rendered
    as —), never to a report error — the native capture stays linked."""
    summary: dict = {}
    if profiler == "ncu":
        launches = [
            launch
            for path in _export_paths(capture, "ncu_details_csv")
            for launch in _parse_ncu_details(exp_dir / path)
        ]
        summary["kernels"] = _aggregate_launches(launches)
        summary["metric_source"] = "details" if launches else None
    elif profiler == "nsys":
        kernels = []
        api = []
        for path in _export_paths(capture, "nsys_stats"):
            rows = _parse_nsys_table(exp_dir / path)
            name = Path(path).name
            if "api_sum" in name:
                api = rows
            else:
                kernels = rows
        summary["kernels"] = kernels
        summary["api"] = api
    elif profiler == "proton":
        path = capture.get("report_path")
        summary["entries"] = (
            _parse_hatchet(exp_dir / path)
            if path and capture.get("status") == "ok"
            else []
        )
    return summary


def _find_capture(
    batch: dict, problem: str, variant_key: str, profiler: str
):
    """The capture for one (variant, profiler): the profiler-suffixed
    entry, else a whole-variant refusal. A refusal (a missing or replaced
    archived kernel) is recorded once without a profiler suffix; it
    applies to every profiler in the batch."""
    captures = batch.get("captures") or {}
    capture = captures.get(f"{problem}/{variant_key}/{profiler}")
    if capture is not None:
        return capture
    refusal = captures.get(f"{problem}/{variant_key}")
    if refusal is not None and refusal.get("profiler") is None:
        return refusal
    return None


def _is_superseded(capture: dict, record: dict) -> bool:
    """Whether the capture describes a kernel this variant no longer has
    (the archived kernel changed after the batch was taken): never render
    its metrics against the new record."""
    expected = record.get("kernel_sha256")
    return bool(
        capture.get("status") == "ok"
        and expected
        and capture.get("kernel_sha256") not in (None, expected)
    )


def _variant_captures(
    exp_dir: Path,
    problem: str,
    variant_key: str,
    record: dict,
    batch: dict | None,
    profilers: list[str],
) -> dict:
    if batch is None:
        return {
            profiler: {"status": "no_capture_batch"} for profiler in profilers
        }
    captures = {}
    for profiler in profilers:
        capture = _find_capture(batch, problem, variant_key, profiler)
        if capture is None:
            captures[profiler] = {"status": "not_captured"}
        elif _is_superseded(capture, record):
            captures[profiler] = {
                "status": "superseded",
                "kernel_sha256": capture.get("kernel_sha256"),
            }
        else:
            captures[profiler] = {
                **capture,
                **_capture_summary(exp_dir, capture, profiler),
            }
    return captures


_VARIANT_FIELDS = (
    "status",
    "verify_status",
    "kind",
    "rounds_requested",
    "rounds_completed",
    "kernel_path",
    "kernel_sha256",
    "source_kernel_sha256",
    "source_path",
    "summary_path",
    "record_path",
    "speedup",
    "acceptance",
    "bench",
    "failure",
)


def _variant_data(
    exp_dir: Path,
    problem: str,
    variant_key: str,
    record: dict,
    batch: dict | None,
    profilers: list[str],
) -> dict:
    data = {"variant": variant_key}
    data.update({field: record.get(field) for field in _VARIANT_FIELDS})
    data["captures"] = _variant_captures(
        exp_dir, problem, variant_key, record, batch, profilers
    )
    return data


def _ordered_variant_records(problem_rec: dict) -> list[tuple[str, dict]]:
    """(key, record) pairs in report order: the generated baseline, then
    rounds_<N> ascending, then external baselines alphabetically."""
    records = []
    generated = problem_rec.get("generated") or {}
    if generated:
        records.append((GENERATED_VARIANT, generated))
    records.extend(
        sorted(
            (problem_rec.get("variants") or {}).items(),
            key=lambda item: _variant_sort_key(item[0]),
        )
    )
    records.extend(sorted((problem_rec.get("baselines") or {}).items()))
    return records


def _problem_data(
    exp_dir: Path,
    name: str,
    problem_rec: dict,
    batch: dict | None,
    profilers: list[str],
) -> dict:
    return {
        "problem": name,
        "case": problem_rec.get("case"),
        "case_spec": problem_rec.get("case_spec"),
        "status": problem_rec.get("status"),
        "variants": [
            _variant_data(exp_dir, name, key, record, batch, profilers)
            for key, record in _ordered_variant_records(problem_rec)
        ],
    }


def build_report_data(
    exp_dir: Path, manifest: dict, capture_id: str | None = None
) -> dict:
    config = manifest.get("configuration") or {}
    batches = manifest.get("capture_batches") or []
    batch = None
    if capture_id is not None:
        batch = next(
            (b for b in batches if b.get("capture_id") == capture_id), None
        )
        if batch is None:
            known = [b.get("capture_id") for b in batches]
            raise SystemExit(
                f"ERROR: no capture batch {capture_id!r} in "
                f"{exp_dir} (known: {', '.join(map(str, known))})"
            )
    elif batches:
        batch = batches[-1]

    # A later capture batch may use profilers the creating run did not
    # (and vice versa), so the report covers the union; order preserved.
    profilers = list(
        dict.fromkeys(
            (config.get("profilers") or [])
            + ((batch.get("profilers") or []) if batch else [])
        )
    )
    if not profilers:
        profilers = list(SUPPORTED_PROFILERS)

    problems = [
        _problem_data(
            exp_dir,
            name,
            manifest.get("problems", {}).get(name) or {},
            batch,
            profilers,
        )
        for name in config.get("problems", [])
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": manifest.get("experiment_id"),
        "status": manifest.get("status"),
        "created_at": manifest.get("created_at"),
        "report_generated_at": datetime.now().isoformat(),
        "configuration": config,
        "environment": manifest.get("environment"),
        "capture_batch": (
            {
                "capture_id": batch.get("capture_id"),
                "created_at": batch.get("created_at"),
                "profilers": batch.get("profilers"),
                "iters": batch.get("iters"),
                "warmup": batch.get("warmup"),
                "ncu_set": batch.get("ncu_set"),
                "status": batch.get("status"),
            }
            if batch
            else None
        ),
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Source diffs


def _source_diff_records(exp_dir: Path, data: dict, *, with_text: bool = True):
    """(record, destination, text) for each pairwise unified diff of a
    problem's archived kernel sources — generated, optimized variants,
    and external baselines — in the report's variant order."""
    for problem in data["problems"]:
        sources = [
            (v["variant"], exp_dir / v["kernel_path"])
            for v in problem["variants"]
            if v.get("kernel_path")
        ]
        # Each source is read once, however many pairs it appears in —
        # the pair loop is quadratic in the variant count.
        lines = {
            path: path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines(keepends=True)
            for path in dict.fromkeys(path for _, path in sources)
            if path.is_file()
        }
        for i, (a_key, a_path) in enumerate(sources):
            for b_key, b_path in sources[i + 1 :]:
                if a_path not in lines or b_path not in lines:
                    continue
                diff = difflib.unified_diff(
                    lines[a_path],
                    lines[b_path],
                    fromfile=f"{problem['problem']}/{a_key}",
                    tofile=f"{problem['problem']}/{b_key}",
                )
                rel = (
                    f"source-diffs/{problem['problem']}__{a_key}_to_{b_key}.diff"
                )
                yield (
                    {
                        "problem": problem["problem"],
                        "from": a_key,
                        "to": b_key,
                        "path": f"reports/{rel}",
                    },
                    exp_dir / "reports" / rel,
                    "".join(diff) if with_text else "",
                )


def _write_source_diffs(exp_dir: Path, data: dict) -> list[dict]:
    """Persist the pairwise source diffs under reports/source-diffs/ and
    return their records (the markdown report links to them)."""
    records = []
    for record, dest, text in _source_diff_records(exp_dir, data):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Markdown rendering


def _link(target: Path | None, from_dir: Path) -> str:
    if target is None:
        return "—"
    try:
        rel = os.path.relpath(target, from_dir)
    except ValueError:
        return f"`{target}`"
    return f"[{Path(rel).name}]({rel})"


def _git_label(git: dict | None):
    """The repository cell: a short commit id, marked when dirty."""
    if not git.get("commit"):
        return None
    return f"{git['commit'][:12]}" + (" (dirty)" if git.get("dirty") else "")


def _llm_label(config: dict, stage: str) -> str:
    """One stage's backend / model cell, falling back to the run-wide
    llm_backend/model of older manifests, plus its reasoning effort."""
    backend = config.get(f"{stage}_backend") or config.get("llm_backend") or "—"
    model = config.get(f"{stage}_model") or "—"
    return f"{backend} / {model}" + _effort_suffix(
        config.get(f"{stage}_reasoning_effort")
    )


def _env_rows(environment: dict | None) -> list[tuple[str, str]]:
    env = environment or {}
    gpu = env.get("gpu") or {}
    rows = [
        ("GPU", gpu.get("name")),
        ("GPU memory", gpu.get("memory_total")),
        ("Driver", gpu.get("driver_version")),
        ("CUDA", env.get("cuda")),
        ("PyTorch", env.get("torch")),
        ("Triton", env.get("triton")),
        ("KernelAgent", env.get("kernelagent")),
        ("Nsight Compute", env.get("ncu_version")),
        ("Nsight Systems", env.get("nsys_version")),
        ("Python", env.get("python")),
        ("Repository", _git_label(env.get("git") or {})),
    ]
    return rows


def _profilers_present(variants: list[dict]) -> list[str]:
    """Profilers with at least one successful capture on this problem."""
    return sorted(
        {
            profiler
            for variant in variants
            for profiler, capture in variant["captures"].items()
            if capture.get("status") == "ok"
        }
    )


def _batch_iters(batch: dict):
    """A capture batch's measured iterations."""
    iters = batch.get("iters") or {}
    return iters.get("default")


def _benchmark_section(variants: list[dict]) -> list[str]:
    generated_ms = None
    for variant in variants:
        if variant["variant"] == GENERATED_VARIANT:
            generated_ms = (variant["bench"] or {}).get("kernel_ms")
            break

    lines = [
        "### Benchmark (fixed harness, `do_bench` mean)",
        "",
        (
            "| Variant | Rounds used/budget | Fixed test | Status | "
            "Kernel (ms) | PyTorch (ms) | vs PyTorch | vs generated |"
        ),
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for variant in variants:
        bench = variant["bench"] or {}
        kernel_ms = bench.get("kernel_ms")
        ref_ms = bench.get("ref_ms")
        if generated_ms and kernel_ms:
            vs_generated = generated_ms / kernel_ms
        else:
            vs_generated = None
        rounds = "—"
        if variant["rounds_requested"] is not None:
            used = (
                str(variant["rounds_completed"])
                if variant["rounds_completed"] is not None
                else "—"
            )
            rounds = f"{used}/{variant['rounds_requested']}"
        # The fixed test passed: optimized variants carry verify_status;
        # the generated record and external baselines carry status.
        test = "PASS" if (
            variant.get("verify_status") == "verified"
            or variant.get("status") == "verified"
        ) else "—"
        variant_cell = f"`{variant['variant']}`"
        if variant.get("kind") == "external":
            variant_cell += " *(external baseline)*"
        lines.append(
            f"| {variant_cell} | {rounds} | {test} "
            f"| {variant['status'] or '—'} "
            f"| {_num(kernel_ms, 4)} | {_num(ref_ms, 4)} "
            f"| {_x(bench.get('speedup'))} | {_x(vs_generated)} |"
        )
    lines.append("")
    return lines


def _acceptance_section(variants: list[dict], data: dict) -> list[str]:
    rows = [
        v
        for v in variants
        if v.get("acceptance") and v["acceptance"].get("source_case_ms")
    ]
    if not rows:
        return []
    lines = [
        (
            "Optimizer acceptance timings (the comparison each "
            "optimization session gated on, case "
            f"{data['configuration'].get('case')}):"
        ),
        "",
        "| Variant | Source (ms) | Candidate (ms) | Accepted speedup |",
        "|---|---:|---:|---:|",
    ]
    for variant in rows:
        acc = variant["acceptance"]
        lines.append(
            f"| `{variant['variant']}` | {_num(acc.get('source_case_ms'), 4)} "
            f"| {_num(acc.get('candidate_case_ms'), 4)} "
            f"| {_x(variant.get('speedup'))} |"
        )
    lines.append("")
    return lines


def _ncu_section(variants: list[dict], batch: dict | None) -> list[str]:
    lines = ["### Nsight Compute (`ncu`)", ""]
    if batch:
        lines.append(
            f"`ncu --set {batch.get('ncu_set')}`, "
            f"{_batch_iters(batch)} measured "
            "iterations; duration is the mean over profiled "
            "launches, throughput and occupancy are per-launch "
            "means (% of peak)."
        )
        lines.append("")
    lines.append(
        "| Variant | Kernel | Launches | Duration avg (µs) | "
        "SM % | Memory % | DRAM % | Occupancy % | Reg/thread |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for variant in variants:
        capture = variant["captures"].get("ncu") or {}
        kernels = capture.get("kernels") or []
        if capture.get("status") != "ok" or not kernels:
            lines.append(_note_row(variant, capture, 7))
            continue
        for kernel in kernels:
            lines.append(
                f"| `{variant['variant']}` "
                f"| `{(kernel['kernel'] or '')[:48]}` "
                f"| {kernel['launches']} "
                f"| {_num(kernel['duration_avg_us'], 2)} "
                f"| {_num(kernel['sm_pct'], 1)} "
                f"| {_num(kernel['mem_pct'], 1)} "
                f"| {_num(kernel['dram_pct'], 1)} "
                f"| {_num(kernel['occupancy_pct'], 1)} "
                f"| {_num(kernel['regs'], 0)} |"
            )
    lines.append("")
    return lines


def _nsys_section(variants: list[dict], batch: dict | None) -> list[str]:
    lines = ["### Nsight Systems (`nsys`)", ""]
    if batch:
        lines.append(
            f"{_batch_iters(batch)} measured iterations; GPU "
            "kernel times exclude warmup and JIT (cudaProfilerApi window)."
        )
        lines.append("")
    lines.append(
        "| Variant | Kernel | Instances | Total (ms) | Avg (µs) | Time % |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    for variant in variants:
        capture = variant["captures"].get("nsys") or {}
        kernels = capture.get("kernels") or []
        if capture.get("status") != "ok" or not kernels:
            lines.append(_note_row(variant, capture, 4))
            continue
        for kernel in kernels:
            lines.append(
                f"| `{variant['variant']}` "
                f"| `{(kernel['name'] or '')[:48]}` "
                f"| {_num(kernel['instances'], 0)} "
                f"| {_num(kernel['total_ms'], 3)} "
                f"| {_num(kernel['avg_us'], 2)} "
                f"| {_num(kernel['time_pct'], 1)} |"
            )
    api_rows = []
    for variant in variants:
        capture = variant["captures"].get("nsys") or {}
        for row in capture.get("api") or []:
            api_rows.append((variant["variant"], row))
    if api_rows:
        api_rows.sort(key=lambda r: -(r[1].get("total_ms") or 0))
        lines.append("")
        lines.append("CUDA API calls (top by total time):")
        lines.append("")
        lines.append("| Variant | API call | Calls | Total (ms) |")
        lines.append("|---|---|---:|---:|")
        for variant_key, row in api_rows[:8]:
            lines.append(
                f"| `{variant_key}` | `{(row['name'] or '')[:48]}` "
                f"| {_num(row['instances'], 0)} "
                f"| {_num(row['total_ms'], 3)} |"
            )
    lines.append("")
    return lines


def _proton_section(variants: list[dict]) -> list[str]:
    lines = [
        "### Proton",
        "",
        "| Variant | Entry | Type | Time (ns) |",
        "|---|---|---|---:|",
    ]
    for variant in variants:
        capture = variant["captures"].get("proton") or {}
        entries = capture.get("entries") or []
        if capture.get("status") != "ok" or not entries:
            lines.append(_note_row(variant, capture, 2))
            continue
        shown = 0
        for entry in entries:
            time_value = next(
                (
                    v
                    for k, v in entry["metrics"].items()
                    if "time" in k.lower()
                ),
                None,
            )
            if time_value is None:
                continue
            lines.append(
                f"| `{variant['variant']}` "
                f"| `{(entry['name'] or '')[:48]}` "
                f"| {entry['type'] or '—'} "
                f"| {_num(time_value, 0)} |"
            )
            shown += 1
        if not shown:
            lines.append(
                f"| `{variant['variant']}` | no timing entries | | |"
            )
    lines.append("")
    return lines


def _artifacts_section(
    variants: list[dict],
    profilers_present: list[str],
    exp_dir: Path,
    out_dir: Path,
) -> list[str]:
    lines = ["#### Artifacts", ""]
    for variant in variants:
        parts = [f"`{variant['variant']}`"]
        if variant.get("kernel_path"):
            parts.append(
                "kernel: "
                + _artifact_link(exp_dir, variant["kernel_path"], out_dir)
            )
            if variant.get("kernel_sha256"):
                parts.append(f"sha256 `{variant['kernel_sha256'][:12]}…`")
        if variant.get("kind") == "external" and variant.get("source_path"):
            # Provenance of an archived external baseline: where it was
            # added from (outside the experiment tree, so plain code).
            parts.append(f"source: `{variant['source_path']}`")
        if variant.get("record_path"):
            parts.append(
                "record: "
                + _artifact_link(exp_dir, variant["record_path"], out_dir)
            )
        if variant.get("summary_path"):
            parts.append(
                "summary: "
                + _artifact_link(exp_dir, variant["summary_path"], out_dir)
            )
        for profiler in profilers_present:
            capture = variant["captures"].get(profiler) or {}
            if capture.get("report_path"):
                parts.append(
                    f"{profiler}: "
                    + _artifact_link(
                        exp_dir, capture["report_path"], out_dir
                    )
                )
        lines.append("- " + " · ".join(parts))
    lines.append("")
    return lines


def _failures_section(variants: list[dict]) -> list[str]:
    failures = [v for v in variants if v.get("failure")]
    if not failures:
        return []
    lines = ["#### Failures", ""]
    for variant in failures:
        failure = variant["failure"]
        lines.append(
            f"- `{variant['variant']}`: {variant['status']} at "
            f"{failure.get('stage')} — {failure.get('reason')}"
        )
    lines.append("")
    return lines


def _render_problem(
    problem: dict, data: dict, exp_dir: Path, out_dir: Path
) -> list[str]:
    lines: list[str] = [
        f"## {problem['problem']} — benchmark case {problem['case']}",
        "",
        f"Case: `{problem['case_spec']}`",
        "",
    ]
    variants = problem["variants"]
    lines.extend(_benchmark_section(variants))
    lines.extend(_acceptance_section(variants, data))

    batch = data.get("capture_batch")
    profilers_present = _profilers_present(variants)
    if not profilers_present:
        lines.append("_No profiler captures available for this problem._")
        lines.append("")
    for profiler in profilers_present:
        if profiler == "ncu":
            lines.extend(_ncu_section(variants, batch))
        elif profiler == "nsys":
            lines.extend(_nsys_section(variants, batch))
        elif profiler == "proton":
            lines.extend(_proton_section(variants))

    lines.extend(
        _artifacts_section(variants, profilers_present, exp_dir, out_dir)
    )
    lines.extend(_failures_section(variants))
    return lines


# How a capture with no metric rows is explained in the tables.
_CAPTURE_NOTES = {
    "missing_kernel": "archived kernel missing — refused",
    "hash_mismatch": "kernel hash mismatch — refused",
    "failed": "capture failed",
    "not_captured": "not captured",
    "superseded": "capture describes a previous kernel",
}


def _capture_note(capture: dict) -> str:
    return _CAPTURE_NOTES.get(capture.get("status"), "no metrics")


def _note_row(variant: dict, capture: dict, empty_cells: int) -> str:
    """The table row for a variant whose capture produced no metric rows:
    the variant cell, then the capture's note (why), then the table's
    remaining columns. empty_cells counts those remaining columns, so
    1 + 1 + empty_cells spans the table."""
    return (
        f"| `{variant['variant']}` | {_capture_note(capture)} "
        + "|" + " |" * empty_cells
    )


def _artifact_link(exp_dir: Path, rel: str | None, out_dir: Path) -> str:
    """A Markdown link to a manifest-relative artifact, resolved against
    the report's own location (the default report sits at
    <experiment>/reports/, a custom --out anywhere)."""
    if not rel:
        return "—"
    return _link(exp_dir / rel, out_dir)


def render_markdown(
    data: dict, exp_dir: Path, out_path: Path, diffs: list[dict]
) -> str:
    out_dir = out_path.parent
    lines: list[str] = []
    lines.append(f"# Kernel comparison experiment: `{data['experiment_id']}`")
    lines.append("")
    lines.append(f"- Status: **{data['status'] or '—'}**")
    lines.append(f"- Created: {data['created_at'] or '—'}")
    lines.append(f"- Report generated: {data['report_generated_at']}")
    batch = data.get("capture_batch")
    if batch:
        lines.append(
            f"- Capture batch: `{batch['capture_id']}` "
            f"({', '.join(batch.get('profilers') or [])}, {batch['status']})"
        )
    else:
        lines.append("- Capture batch: none")
    lines.append("")

    config = data.get("configuration") or {}
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Key | Value |")
    lines.append("|---|---|")
    for key, value in (
        ("Problems", ", ".join(config.get("problems") or [])),
        ("Benchmark case", config.get("case")),
        ("Generation rounds (budget)", config.get("generation_rounds")),
        (
            "Optimization budgets",
            ", ".join(str(r) for r in config.get("optimization_rounds") or []),
        ),
        ("Workers", config.get("workers")),
        ("Generation LLM", _llm_label(config, "generation")),
        ("Optimization LLM", _llm_label(config, "optimization")),
        ("Profilers", ", ".join(config.get("profilers") or [])),
    ):
        lines.append(f"| {key} | {value} |")
    lines.append("")

    lines.append("## Environment")
    lines.append("")
    lines.append("| Component | Version |")
    lines.append("|---|---|")
    for key, value in _env_rows(data.get("environment")):
        lines.append(f"| {key} | {value if value is not None else '—'} |")
    lines.append("")

    for problem in data["problems"]:
        lines.extend(_render_problem(problem, data, exp_dir, out_dir))

    if diffs:
        lines.append("## Source diffs")
        lines.append("")
        for diff in diffs:
            lines.append(
                f"- {diff['problem']}: `{diff['from']}` → `{diff['to']}`: "
                + _artifact_link(exp_dir, diff["path"], out_dir)
            )
        lines.append("")

    lines.append("## Interpreting the numbers")
    lines.append("")
    lines.append(
        "- Benchmark latencies come from the fixed `bench.py` harness "
        "(`triton.testing.do_bench` means over the exact case shapes and "
        "dtype); profiler durations come from each tool's own measurement "
        "of the captured launches and are not interchangeable with them."
    )
    lines.append(
        "- `rounds used/budget` reports the optimizer's completed rounds "
        "against the requested budget (`KA_ROUNDS` is a maximum; a run "
        "may stop earlier)."
    )
    lines.append(
        "- Every optimization budget started from the same generated "
        "kernel (hash-checked); rejected or failed attempts keep their "
        "outcome and are never replaced by another variant's kernel."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry points


def generate_report(
    exp_dir: Path, out_path: Path | None = None, capture_id: str | None = None
) -> Path:
    """Build comparison.json, source diffs, and comparison.md for one
    experiment; returns the Markdown path."""
    exp_dir = Path(exp_dir).resolve()
    manifest = _load_manifest(exp_dir)
    data = build_report_data(exp_dir, manifest, capture_id)
    diffs = _write_source_diffs(exp_dir, data)
    data["diffs"] = diffs

    reports_dir = exp_dir / "reports"
    _atomic_write_json(reports_dir / "comparison.json", data)
    out_path = Path(out_path) if out_path else reports_dir / "comparison.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        out_path, render_markdown(data, exp_dir, out_path, diffs)
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "experiment", help="experiment id or experiment directory"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_MOUNT,
        help="results mount holding experiments/ (default: RESULTS_MOUNT "
        "or /workspace/user/results)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the Markdown here instead of "
        "<experiment>/reports/comparison.md",
    )
    parser.add_argument(
        "--capture-batch",
        default=None,
        help="report this capture batch (default: the latest)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="output format (default: markdown)",
    )
    args = parser.parse_args()

    exp_dir = resolve_experiment(args.results_root, args.experiment).resolve()

    if args.format == "json":
        manifest = _load_manifest(exp_dir)
        data = build_report_data(exp_dir, manifest, args.capture_batch)
        # The diff records are computed, not written: this is a read-only
        # query, and a following markdown render (re)writes the files.
        data["diffs"] = [
            record
            for record, _, _ in _source_diff_records(
                exp_dir, data, with_text=False
            )
        ]
        print(json.dumps(data, indent=2))
        return 0
    out = generate_report(exp_dir, args.out, args.capture_batch)
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
