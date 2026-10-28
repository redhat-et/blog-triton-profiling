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
    profile-kernel.py writes beside each capture; achieved TFLOPS comes
    from the .raw.csv export's tensor-pipeline FLOP counters, and the
    Roofline and SOL section combines NCU's GPU Speed Of Light
    percentages with the raw page's FLOPs, DRAM bytes, and peak-sustained
    rates (one SVG roofline per problem under reports/)
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
                       [--capture-batch ID|latest-per-profiler]
                       [--format markdown|json]

  <experiment>  experiment id under <results-root>/experiments, or an
                experiment directory directly
  --results-root  overrides RESULTS_MOUNT (default from the environment
                or /workspace/user/results)
  --out         write the Markdown somewhere other than
                <experiment>/reports/comparison.md (links are computed
                relative to the destination)
  --capture-batch  report a specific capture batch (default: the latest),
                or latest-per-profiler to use the newest capture of each
                profiler across every batch
  --format json  print the normalized report data as JSON and exit

Examples:
  report-experiment.py webinar-v1
  report-experiment.py webinar-v1 --out slides/kernel-comparison.md
  report-experiment.py webinar-v1 --capture-batch latest-per-profiler
  report-experiment.py webinar-v1 --format json
"""

from __future__ import annotations

import argparse
import csv
import difflib
import html
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

from ka_common import (
    GENERATED_VARIANT,
    LATEST_PER_PROFILER,
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

# Percentage metrics must be matched with their unit: the details page
# reports "Memory Throughput" twice — the GPU Speed Of Light percentage
# and the Memory Workload Analysis GByte/s rate — and keying on the name
# alone silently keeps the rate.
_PCT_METRICS = {"sm_pct", "mem_pct", "dram_pct", "occupancy_pct"}

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


_BYTE_UNITS = {
    "byte": 1.0,
    "bytes": 1.0,
    "kbyte": 1e3,
    "kilobyte": 1e3,
    "mbyte": 1e6,
    "megabyte": 1e6,
    "gbyte": 1e9,
    "gigabyte": 1e9,
    "tbyte": 1e12,
    "terabyte": 1e12,
    # The peak DRAM rate is exported per DRAM cycle; scale those the same
    # way so a version that reports Kbyte/cycle still lands in bytes.
    "byte/cycle": 1.0,
    "kbyte/cycle": 1e3,
    "mbyte/cycle": 1e6,
    "gbyte/cycle": 1e9,
}

# Rate units for the raw page's peak-sustained FLOP column. NCU exports
# it as 1/ns (e.g. 104231.9, i.e. 104.2 TFLOPS); 1/s and 1/us cover
# other exports of the same counter.
_FLOPS_UNITS = {
    "1/s": 1e-12,
    "1/ns": 1e-3,
    "1/us": 1e-6,
    "1/ms": 1e-9,
}

_GHZ_UNITS = {
    "hz": 1e-9,
    "khz": 1e-6,
    "mhz": 1e-3,
    "ghz": 1.0,
}


def _bytes(value, unit):
    """A byte count converted to bytes; unknown units stay unscaled."""
    number = _to_float(value)
    if number is None:
        return None
    return number * _BYTE_UNITS.get((unit or "").strip().lower(), 1.0)


def _flops_per_second(value, unit):
    """A raw-page FLOP rate converted to TFLOPS; unknown units assume the
    raw page's 1/ns export."""
    number = _to_float(value)
    if number is None:
        return None
    factor = _FLOPS_UNITS.get((unit or "").strip().lower(), 1e-3)
    return number * factor


def _ghz(value, unit):
    """A clock rate converted to GHz; unknown units assume GHz."""
    number = _to_float(value)
    if number is None:
        return None
    return number * _GHZ_UNITS.get((unit or "").strip().lower(), 1.0)


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
        if (
            normalized in _PCT_METRICS
            and (unit or "").strip() not in ("", "%")
        ):
            # The Memory Workload Analysis page repeats "Memory
            # Throughput" in GByte/s; never let a rate overwrite the
            # GPU Speed Of Light percentage. A missing unit is accepted
            # for exports that omit the column.
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


# Raw-page metric families for the roofline. NCU reports executed FLOPs
# (an FMA counts as two operations) and its own peak-sustained rates;
# only the full metric set collects the tensor-pipeline counters.
_NCU_TENSOR_OPS_PREFIX = "sm__ops_path_tensor_"
_NCU_TENSOR_PEAK_SUFFIX = ".peak_sustained_elapsed.per_second"
_NCU_DURATION_METRIC = "gpu__time_duration.sum"
_NCU_DRAM_READ = "dram__bytes_read.sum"
_NCU_DRAM_WRITE = "dram__bytes_write.sum"
_NCU_DRAM_PEAK = "dram__bytes.sum.peak_sustained"
_NCU_DRAM_CLOCK = "dram__cycles_elapsed.avg.per_second"

# Per-kernel fields the raw page contributes to the report (and JSON).
_ROOFLINE_FIELDS = (
    "flops",
    "dram_bytes",
    "ai_flop_per_byte",
    "ridge_flop_per_byte",
    "peak_tflops",
    "peak_bw_gbps",
    "roofline_tflops",
    "achieved_bw_gbps",
)


def _parse_ncu_raw(path: Path) -> list[dict]:
    """Per-launch roofline inputs from --csv --page raw.

    The raw page is one column per metric, with a units row under the
    header. Response records carry the executed tensor FLOPs and the
    matching peak-sustained rate, the DRAM read+write bytes, and the
    capture's peak DRAM bandwidth. Only ``--set full`` collects the
    tensor counters, so a basic/detailed capture yields no records
    (rendered as —).
    """
    rows = _read_csv_rows(path)
    header_idx = None
    for i, row in enumerate(rows):
        headers = {_norm_header(c) for c in row}
        if {"id", "kernelname", _NCU_DURATION_METRIC} <= headers:
            header_idx = i
            break
    if header_idx is None:
        return []
    header = {_norm_header(c): j for j, c in enumerate(rows[header_idx])}
    id_col = header["id"]
    kernel_col = header["kernelname"]
    duration_col = header[_NCU_DURATION_METRIC]

    # (executed-FLOP column, its peak-rate column or None)
    tensor_cols = []
    for name, col in header.items():
        if not (
            name.startswith(_NCU_TENSOR_OPS_PREFIX)
            and name.endswith(".sum")
        ):
            continue
        tensor_cols.append((col, header.get(name + _NCU_TENSOR_PEAK_SUFFIX)))
    if not tensor_cols:
        return []

    start = header_idx + 1
    units: dict[int, str] = {}
    if start < len(rows):
        id_cell = rows[start][id_col] if id_col < len(rows[start]) else None
        if _to_int(id_cell) is None:
            # The units row sits directly under the header.
            units = dict(enumerate(rows[start]))
            start += 1

    def unit(col):
        return units.get(col)

    duration_unit = unit(duration_col) or "us"
    if (duration_unit or "").strip().lower() not in _DURATION_UNITS:
        duration_unit = "us"

    read_col = header.get(_NCU_DRAM_READ)
    write_col = header.get(_NCU_DRAM_WRITE)
    peak_col = header.get(_NCU_DRAM_PEAK)
    clock_col = header.get(_NCU_DRAM_CLOCK)

    launches = []
    for row in rows[start:]:
        try:
            kernel = row[kernel_col]
            duration_us = _duration_us(row[duration_col], duration_unit)
        except IndexError:
            continue
        if not kernel or duration_us is None:
            continue
        flops = 0.0
        tensor_peak_tflops = None
        for col, peak in tensor_cols:
            if col >= len(row):
                continue
            executed = _to_float(row[col]) or 0.0
            if executed <= 0:
                continue
            flops += executed
            if peak is not None and peak < len(row):
                candidate = _flops_per_second(row[peak], unit(peak))
                if candidate is not None and candidate > 0:
                    # Mixed-precision kernels sum FLOPs across active
                    # paths and divide by the fastest one's peak, a
                    # conservative denominator.
                    tensor_peak_tflops = max(
                        tensor_peak_tflops or 0.0, candidate
                    )
        dram_bytes = None
        if read_col is not None or write_col is not None:
            dram_bytes = 0.0
            for col in (read_col, write_col):
                if col is None:
                    continue
                value = row[col] if col < len(row) else None
                dram_bytes += _bytes(value, unit(col)) or 0.0
        dram_peak_gbps = None
        if (
            peak_col is not None
            and clock_col is not None
            and peak_col < len(row)
            and clock_col < len(row)
        ):
            per_cycle = _bytes(row[peak_col], unit(peak_col))
            clock_ghz = _ghz(row[clock_col], unit(clock_col))
            if per_cycle is not None and clock_ghz is not None:
                dram_peak_gbps = per_cycle * clock_ghz
        launches.append(
            {
                "kernel": kernel,
                "duration_us": duration_us,
                "flops": flops,
                "tensor_peak_tflops": tensor_peak_tflops,
                "dram_bytes": dram_bytes,
                "dram_peak_gbps": dram_peak_gbps,
            }
        )
    return launches


def _weighted_mean(pairs):
    """Duration-weighted mean of (value, duration) pairs, ignoring None
    values and non-positive durations."""
    total = 0.0
    weight = 0.0
    for value, duration in pairs:
        if value is None or duration is None or duration <= 0:
            continue
        total += value * duration
        weight += duration
    return total / weight if weight > 0 else None


def _ratio(numerator, denominator):
    if numerator is None or not denominator:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _aggregate_ncu_raw(launches: list[dict]) -> dict[str, dict]:
    """Per-kernel roofline geometry from the raw page.

    Achieved TFLOPS and DRAM bandwidth weight every launch by its cost
    (summed FLOPs/bytes over summed time). The peak-sustained
    denominators are duration-weighted means because NCU scales them
    with the clock it sampled during each replay. SOL percentages are
    not derived here: they are NCU's own details-page values.
    """
    grouped: dict[str, dict] = {}
    for launch in launches:
        acc = grouped.setdefault(
            launch["kernel"],
            {
                "launches": 0,
                "flops": 0.0,
                "dram_bytes": 0.0,
                "duration_us": 0.0,
                "bytes_complete": True,
                "peak_tflops": [],
                "peak_bw_gbps": [],
            },
        )
        acc["launches"] += 1
        acc["flops"] += launch["flops"]
        acc["duration_us"] += launch["duration_us"]
        if launch["dram_bytes"] is None:
            acc["bytes_complete"] = False
        else:
            acc["dram_bytes"] += launch["dram_bytes"]
        acc["peak_tflops"].append(
            (launch["tensor_peak_tflops"], launch["duration_us"])
        )
        acc["peak_bw_gbps"].append(
            (launch["dram_peak_gbps"], launch["duration_us"])
        )

    summaries = {}
    for kernel, acc in grouped.items():
        duration_s = acc["duration_us"] * 1e-6
        tflops = _ratio(acc["flops"], duration_s * 1e12)
        dram_bytes = acc["dram_bytes"] if acc["bytes_complete"] else None
        achieved_bw_gbps = _ratio(dram_bytes, duration_s * 1e9)
        ai = _ratio(acc["flops"], dram_bytes)
        if acc["flops"] <= 0:
            # A non-tensor kernel has a counter column but no FLOPs;
            # keep TFLOPS at 0 like before, but no roofline point.
            ai = None
        peak_tflops = _weighted_mean(acc["peak_tflops"])
        peak_bw_gbps = _weighted_mean(acc["peak_bw_gbps"])
        ridge = (
            _ratio(peak_tflops * 1000.0, peak_bw_gbps)
            if peak_tflops is not None and peak_bw_gbps
            else None
        )
        roofline = (
            min(peak_tflops, ai * peak_bw_gbps / 1000.0)
            if peak_tflops is not None
            and ai is not None
            and peak_bw_gbps
            else None
        )
        summaries[kernel] = {
            "launches": acc["launches"],
            "flops": acc["flops"],
            "dram_bytes": dram_bytes,
            "tflops": tflops,
            "achieved_bw_gbps": achieved_bw_gbps,
            "ai_flop_per_byte": ai,
            "peak_tflops": peak_tflops,
            "peak_bw_gbps": peak_bw_gbps,
            "ridge_flop_per_byte": ridge,
            "roofline_tflops": roofline,
        }
    return summaries


def _limiter_note(kernel: dict) -> str | None:
    """A short heuristic note about what limits a kernel.

    This is explanatory text layered over NCU's percentages, not a second
    model: the bound comes from NCU's GPU SOL (max of compute and
    memory), and the flags call out occupancy/register pressure that the
    same capture already reports.
    """
    bound = kernel.get("bound")
    compute = kernel.get("sm_pct")
    memory = kernel.get("mem_pct")
    dram = kernel.get("dram_pct")
    parts = []
    if bound == "compute" and compute is not None:
        parts.append(f"tensor pipelines ({compute:.1f}%)")
    elif bound == "memory" and memory is not None:
        parts.append(f"memory hierarchy ({memory:.1f}%)")
        if dram is not None and memory - dram >= 15:
            parts.append(f"cache-level pressure (DRAM {dram:.1f}%)")
    occupancy = kernel.get("occupancy_pct")
    if occupancy is not None and occupancy < 25:
        parts.append(f"low occupancy ({occupancy:.1f}%)")
    regs = kernel.get("regs")
    if regs is not None and regs >= 128:
        parts.append(f"register pressure ({regs}/thread)")
    return "; ".join(parts) if parts else None


def _classify_bottleneck(kernel: dict) -> None:
    """Attach NCU's GPU SOL and the compute/memory verdict to a kernel:
    SOL % is the larger of the Compute (SM) and Memory Throughput
    percentages, exactly as NCU's Speed Of Light rule reports it."""
    compute = kernel.get("sm_pct")
    memory = kernel.get("mem_pct")
    if compute is not None and memory is not None:
        kernel["sol_pct"] = max(compute, memory)
        kernel["bound"] = "compute" if compute >= memory else "memory"
    else:
        kernel["sol_pct"] = None
        kernel["bound"] = None
    kernel["limiter"] = _limiter_note(kernel)


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
        raw = _aggregate_ncu_raw(
            [
                launch
                for path in _export_paths(capture, "ncu_raw_csv")
                for launch in _parse_ncu_raw(exp_dir / path)
            ]
        )
        for kernel in summary["kernels"]:
            roofline = raw.get(kernel["kernel"]) or {}
            kernel["tflops"] = roofline.get("tflops")
            for field in _ROOFLINE_FIELDS:
                kernel[field] = roofline.get(field)
            _classify_bottleneck(kernel)
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


def _merge_capture_batches(
    batches: list[dict],
) -> tuple[dict, dict[str, list[str]]]:
    """The newest capture per (problem, variant, profiler) across batches.

    Batches are walked oldest to newest, so a later capture overrides an
    older one for the same key. A variant-level refusal (key
    "problem/variant", profiler None) describes the archived kernel at
    that batch, so a newer refusal drops older profiler captures and a
    newer successful capture drops an older refusal. Returns the merged
    captures plus, per profiler, the batch ids that supplied the selected
    records.
    """
    merged: dict = {}
    for batch in batches:
        captures = batch.get("captures") or {}
        for key, record in captures.items():
            if record.get("profiler") is None:
                for old_key in [k for k in merged if k.startswith(f"{key}/")]:
                    del merged[old_key]
                merged[key] = record
        for key, record in captures.items():
            if record.get("profiler") is not None:
                merged.pop(key.rsplit("/", 1)[0], None)
                merged[key] = record

    provenance: dict[str, list[str]] = {}
    for batch in batches:
        capture_id = batch.get("capture_id")
        for key, record in (batch.get("captures") or {}).items():
            # Identity, not equality: only the record currently held by
            # the merge is the selected newest capture for this key.
            if merged.get(key) is not record:
                continue
            profiler = record.get("profiler")
            if profiler is None:
                continue
            ids = provenance.setdefault(profiler, [])
            if capture_id not in ids:
                ids.append(capture_id)
    return merged, provenance


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


def _safe_problem_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name or "problem")


def _roofline_points(variants: list[dict]) -> list[dict]:
    """One chart point per ncu kernel with a usable roofline, in report
    order. A point needs the raw page's AI, achieved TFLOPS, and both
    peak rates."""
    points = []
    for variant in variants:
        capture = variant["captures"].get("ncu") or {}
        if capture.get("status") != "ok":
            continue
        for kernel in capture.get("kernels") or []:
            if (
                kernel.get("ai_flop_per_byte") is None
                or kernel.get("tflops") is None
                or kernel.get("peak_tflops") is None
                or kernel.get("peak_bw_gbps") is None
            ):
                continue
            points.append(
                {
                    "variant": variant["variant"],
                    "kernel": kernel["kernel"],
                    "ai": kernel["ai_flop_per_byte"],
                    "tflops": kernel["tflops"],
                    "peak_tflops": kernel["peak_tflops"],
                    "peak_bw_gbps": kernel["peak_bw_gbps"],
                    "sol_pct": kernel.get("sol_pct"),
                }
            )
    return points


def _problem_roofline(variants: list[dict]) -> dict | None:
    """The problem's shared roofline denominators: the highest
    peak-sustained rates observed across its kernels, and the ridge they
    imply."""
    points = _roofline_points(variants)
    if not points:
        return None
    peak_tflops = max(
        (point["peak_tflops"] for point in points), default=None
    )
    peak_bw_gbps = max(
        (point["peak_bw_gbps"] for point in points), default=None
    )
    return {
        "peak_tflops": peak_tflops,
        "peak_bw_gbps": peak_bw_gbps,
        "ridge_flop_per_byte": _ratio(
            (peak_tflops or 0.0) * 1000.0, peak_bw_gbps
        ),
        "points": len(points),
    }


def _problem_data(
    exp_dir: Path,
    name: str,
    problem_rec: dict,
    batch: dict | None,
    profilers: list[str],
) -> dict:
    variants = [
        _variant_data(exp_dir, name, key, record, batch, profilers)
        for key, record in _ordered_variant_records(problem_rec)
    ]
    data = {
        "problem": name,
        "case": problem_rec.get("case"),
        "case_spec": problem_rec.get("case_spec"),
        "status": problem_rec.get("status"),
        "variants": variants,
    }
    roofline = _problem_roofline(variants)
    if roofline:
        data["roofline"] = roofline
        data["roofline_chart"] = (
            f"reports/roofline-{_safe_problem_stem(name)}.svg"
        )
    return data


def build_report_data(
    exp_dir: Path, manifest: dict, capture_id: str | None = None
) -> dict:
    config = manifest.get("configuration") or {}
    batches = manifest.get("capture_batches") or []
    merge: dict | None = None
    batch = None
    if capture_id == LATEST_PER_PROFILER:
        merged, profiler_batches = _merge_capture_batches(batches)
        batch = batches[-1] if batches else None
        captures_batch = {"captures": merged} if batches else None
        merge = {
            "mode": LATEST_PER_PROFILER,
            "capture_ids": [b.get("capture_id") for b in batches],
            "profiler_batches": profiler_batches,
        }
    else:
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
        captures_batch = batch

    # A later capture batch may use profilers the creating run did not
    # (and vice versa), so the report covers the union; order preserved.
    profiler_sources = (
        [
            profiler
            for b in batches
            for profiler in (b.get("profilers") or [])
        ]
        if merge is not None
        else ((batch.get("profilers") or []) if batch else [])
    )
    profilers = list(
        dict.fromkeys((config.get("profilers") or []) + profiler_sources)
    )
    if not profilers:
        profilers = list(SUPPORTED_PROFILERS)

    problems = [
        _problem_data(
            exp_dir,
            name,
            manifest.get("problems", {}).get(name) or {},
            captures_batch,
            profilers,
        )
        for name in config.get("problems", [])
    ]

    data = {
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
    if merge is not None:
        data["capture_merge"] = merge
    return data


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


def _ncu_section(
    variants: list[dict], batch: dict | None, merge: dict | None = None
) -> list[str]:
    lines = ["### Nsight Compute (`ncu`)", ""]
    if merge:
        lines.append(
            "Each variant's newest `ncu` capture across capture batches "
            "(`latest-per-profiler`); `--set` and measured iterations come "
            "from the batch that produced each capture, so duration is the "
            "mean over profiled launches, and throughput and occupancy are "
            "per-launch means (% of peak)."
        )
        lines.append("")
    elif batch:
        lines.append(
            f"`ncu --set {batch.get('ncu_set')}`, "
            f"{_batch_iters(batch)} measured "
            "iterations; duration is the mean over profiled "
            "launches, throughput and occupancy are per-launch "
            "means (% of peak)."
        )
        lines.append("")
    lines.append(
        "TFLOPS is tensor-pipeline executed FLOPs "
        "(`sm__ops_path_tensor_*.sum`) over summed profiled launch "
        "time; collecting it requires `ncu --set full`."
    )
    lines.append("")
    lines.append(
        "| Variant | Kernel | Launches | Duration avg (µs) | TFLOPS | "
        "SM % | Memory % | DRAM % | Occupancy % | Reg/thread |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for variant in variants:
        capture = variant["captures"].get("ncu") or {}
        kernels = capture.get("kernels") or []
        if capture.get("status") != "ok" or not kernels:
            lines.append(_note_row(variant, capture, 8))
            continue
        for kernel in kernels:
            lines.append(
                f"| `{variant['variant']}` "
                f"| `{(kernel['kernel'] or '')[:48]}` "
                f"| {kernel['launches']} "
                f"| {_num(kernel['duration_avg_us'], 2)} "
                f"| {_num(kernel.get('tflops'), 1)} "
                f"| {_num(kernel['sm_pct'], 1)} "
                f"| {_num(kernel['mem_pct'], 1)} "
                f"| {_num(kernel['dram_pct'], 1)} "
                f"| {_num(kernel['occupancy_pct'], 1)} "
                f"| {_num(kernel['regs'], 0)} |"
            )
    lines.append("")
    return lines


def _nsys_section(
    variants: list[dict], batch: dict | None, merge: dict | None = None
) -> list[str]:
    lines = ["### Nsight Systems (`nsys`)", ""]
    if merge:
        lines.append(
            "Each variant's newest `nsys` capture across capture batches "
            "(`latest-per-profiler`); measured iterations come from the "
            "batch that produced each capture, and GPU kernel times exclude "
            "warmup and JIT (cudaProfilerApi window)."
        )
        lines.append("")
    elif batch:
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


def _proton_section(
    variants: list[dict], merge: dict | None = None
) -> list[str]:
    lines = ["### Proton", ""]
    if merge:
        lines.append(
            "Each variant's newest Proton capture across capture batches "
            "(`latest-per-profiler`); entries are the in-process Triton "
            "profiler's timing tree."
        )
        lines.append("")
    lines.extend(
        [
            "| Variant | Entry | Type | Time (ns) |",
            "|---|---|---|---:|",
        ]
    )
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


# ---------------------------------------------------------------------------
# Roofline and SOL rendering


_SVG_WIDTH = 760
_SVG_LEFT = 78
_SVG_RIGHT = 28
_SVG_PLOT_TOP = 72
_SVG_PLOT_HEIGHT = 300
# Color-blind-safe qualitative palette, assigned per variant in report order.
_SVG_POINT_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#999999",
)


def _scaled(value, divisor):
    if value is None:
        return None
    try:
        return float(value) / divisor
    except (TypeError, ValueError):
        return None


def _relative_href(target: Path, from_dir: Path) -> str:
    try:
        return Path(os.path.relpath(target, from_dir)).as_posix()
    except ValueError:
        return target.as_posix()


def _nice_ceiling(value: float) -> float:
    """A readable upper bound (1/1.5/2/2.5/3/4/5/6/8 x 10^n) above value."""
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10.0 ** exponent
    for step in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if value <= step * base:
            return step * base
    return 10.0 * base


def _log_ticks(lo: float, hi: float) -> list[float]:
    ticks = []
    exponent = math.floor(math.log10(lo))
    while 10.0 ** exponent <= hi * 1.0001:
        for step in (1.0, 2.0, 5.0):
            value = step * 10.0 ** exponent
            if lo <= value <= hi:
                ticks.append(value)
        exponent += 1
    return ticks or [lo, hi]


def _axis_label(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000.0:g}k"
    return f"{value:g}"


def _roofline_svg(
    problem: str,
    points: list[dict],
    peak_tflops: float,
    peak_bw_gbps: float,
    ridge: float,
) -> str:
    """A self-contained roofline chart: arithmetic intensity on a log x
    axis, achieved TFLOPS on a linear y axis, the compute and DRAM roofs,
    and one point per kernel with a legend."""
    width = _SVG_WIDTH
    left = _SVG_LEFT
    right = width - _SVG_RIGHT
    top = _SVG_PLOT_TOP
    bottom = top + _SVG_PLOT_HEIGHT
    plot_width = right - left

    intensities = [point["ai"] for point in points] + [ridge]
    intensities = [
        value for value in intensities if value and value > 0
    ] or [1.0]
    log_lo = math.log10(min(intensities)) - 0.12
    log_hi = math.log10(max(intensities)) + 0.12
    if log_hi - log_lo < 0.7:
        middle = (log_hi + log_lo) / 2.0
        log_lo, log_hi = middle - 0.35, middle + 0.35
    x_lo, x_hi = 10.0 ** log_lo, 10.0 ** log_hi

    y_max = _nice_ceiling(
        max([peak_tflops] + [point["tflops"] for point in points]) * 1.1
    )

    def px(value):
        fraction = (math.log10(value) - log_lo) / (log_hi - log_lo)
        return left + fraction * plot_width

    def py(value):
        return bottom - (value / y_max) * _SVG_PLOT_HEIGHT

    def esc(text):
        return html.escape(str(text), quote=False)

    legend_top = bottom + 64
    height = legend_top + len(points) * 18 + 16
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img">'
        ),
        f"<title>{esc('Roofline for ' + problem)}</title>",
        "<style>",
        ".bg{fill:#ffffff}.fg{fill:#1f2328}.muted{fill:#57606a}",
        ".grid{stroke:#d0d7de;stroke-width:1}",
        ".axis{stroke:#57606a;stroke-width:1}",
        ".roof{stroke:#8250df;stroke-width:2.5;fill:none}",
        ".ridge{stroke:#8250df;stroke-width:1;stroke-dasharray:5 4}",
        "@media (prefers-color-scheme: dark){",
        ".bg{fill:#0d1117}.fg{fill:#e6edf3}.muted{fill:#8b949e}",
        ".grid{stroke:#30363d}.axis{stroke:#8b949e}",
        ".roof{stroke:#a371f7}.ridge{stroke:#a371f7}",
        "}",
        "</style>",
        (
            f'<rect class="bg" x="0" y="0" width="{width}" '
            f'height="{height}"/>'
        ),
        (
            f'<text class="fg" x="{left}" y="30" font-size="18" '
            f'font-weight="600">Roofline — {esc(problem)}</text>'
        ),
        (
            f'<text class="muted" x="{left}" y="50" font-size="12">'
            f"peak {peak_tflops:.1f} TFLOPS · {peak_bw_gbps:.0f} GB/s · "
            f"ridge {ridge:.0f} FLOP/B</text>"
        ),
    ]

    for tick in _log_ticks(x_lo, x_hi):
        x = px(tick)
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top}" '
            f'x2="{x:.1f}" y2="{bottom}"/>'
        )
        parts.append(
            f'<text class="muted" x="{x:.1f}" y="{bottom + 18}" '
            f'font-size="11" text-anchor="middle">'
            f"{esc(_axis_label(tick))}</text>"
        )
    for index in range(5):
        value = y_max * index / 4.0
        y = py(value)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" '
            f'x2="{right}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="muted" x="{left - 10}" y="{y + 4:.1f}" '
            f'font-size="11" text-anchor="end">{value:.0f}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{bottom}" '
        f'x2="{right}" y2="{bottom}"/>'
    )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{top}" '
        f'x2="{left}" y2="{bottom}"/>'
    )
    parts.append(
        f'<text class="muted" x="{(left + right) / 2.0:.1f}" '
        f'y="{bottom + 40}" font-size="12" text-anchor="middle">'
        "Arithmetic intensity (FLOP/byte, log scale)</text>"
    )
    middle_y = (top + bottom) / 2.0
    parts.append(
        f'<text class="muted" x="18" y="{middle_y:.1f}" font-size="12" '
        f'text-anchor="middle" transform="rotate(-90 18 '
        f'{middle_y:.1f})">Achieved throughput (TFLOPS)</text>'
    )

    def roof_value(intensity):
        return min(peak_tflops, intensity * peak_bw_gbps / 1000.0)

    samples = []
    steps = 80
    for index in range(steps + 1):
        intensity = 10.0 ** (
            log_lo + (log_hi - log_lo) * index / steps
        )
        samples.append(
            f"{px(intensity):.1f},{py(roof_value(intensity)):.1f}"
        )
    parts.append(f'<polyline class="roof" points="{" ".join(samples)}"/>')

    parts.append(
        f'<text class="muted" x="{right - 8}" '
        f'y="{py(peak_tflops) - 8:.1f}" font-size="11" '
        'text-anchor="end">compute roof</text>'
    )
    if x_lo < ridge < x_hi:
        slope_end = 10.0 ** (
            math.log10(x_lo)
            + 0.35 * (math.log10(ridge) - math.log10(x_lo))
        )
    else:
        slope_end = x_lo * 1.2
    if slope_end < x_hi:
        parts.append(
            f'<text class="muted" x="{px(slope_end):.1f}" '
            f'y="{py(roof_value(slope_end)) - 8:.1f}" font-size="11">'
            "DRAM roof</text>"
        )
    if x_lo < ridge < x_hi:
        x = px(ridge)
        parts.append(
            f'<line class="ridge" x1="{x:.1f}" y1="{top}" '
            f'x2="{x:.1f}" y2="{bottom}"/>'
        )
        parts.append(
            f'<text class="muted" x="{x + 6:.1f}" y="{top + 14}" '
            f'font-size="11">ridge {ridge:.0f} FLOP/B</text>'
        )

    variants = list(
        dict.fromkeys(point["variant"] for point in points)
    )
    colors = {
        variant: _SVG_POINT_COLORS[index % len(_SVG_POINT_COLORS)]
        for index, variant in enumerate(variants)
    }
    for point in points:
        color = colors[point["variant"]]
        parts.append(
            f'<circle cx="{px(point["ai"]):.1f}" '
            f'cy="{py(point["tflops"]):.1f}" r="5" fill="{color}" '
            'stroke="#ffffff" stroke-width="1.5"/>'
        )
    for index, point in enumerate(points):
        color = colors[point["variant"]]
        y = legend_top + index * 18
        parts.append(
            f'<rect x="{left}" y="{y - 9}" width="10" height="10" '
            f'fill="{color}" stroke="#ffffff" stroke-width="1"/>'
        )
        label = point["variant"]
        if sum(
            1
            for other in points
            if other["variant"] == point["variant"]
        ) > 1:
            label += f" · {point['kernel']}"
        if point.get("sol_pct") is not None:
            label += f" — SOL {point['sol_pct']:.1f}%"
        parts.append(
            f'<text class="fg" x="{left + 16}" y="{y}" font-size="12">'
            f"{esc(label)}</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _roofline_section(
    problem: dict, exp_dir: Path, out_dir: Path
) -> list[str]:
    variants = problem["variants"]
    points = _roofline_points(variants)
    meta = problem.get("roofline") or {}
    has_sol = any(
        kernel.get("sol_pct") is not None
        for variant in variants
        for kernel in (
            (variant["captures"].get("ncu") or {}).get("kernels") or []
        )
    )
    if not points and not has_sol:
        return []
    lines = [
        "### Roofline and SOL",
        "",
        (
            "SOL percentages are Nsight Compute's own GPU Speed Of Light "
            "metrics: `Compute (SM) Throughput`, `Memory Throughput` (the "
            "busiest memory level, L1/L2 or DRAM), and `DRAM Throughput`. "
            "`SOL %` is the larger of compute and memory, as NCU's rule "
            "defines it, and `Bound` names that larger one."
        ),
        "",
        (
            "The roofline derives arithmetic intensity from executed tensor "
            "FLOPs (`sm__ops_path_tensor_*.sum`) over DRAM read+write bytes, "
            "and takes its peaks from the same `--set full` raw capture. "
            "FLOPs and DRAM bytes are totals over the profiled launches; "
            "the peaks are duration-weighted means because NCU scales them "
            "with the sampled clock. It is a classic DRAM roofline, so a "
            "kernel can sit on the compute side of the ridge while NCU's "
            "cache-level memory SOL is the larger percentage."
        ),
        "",
        (
            "`Limiter` adds heuristics from the same capture: occupancy "
            "below 25 % and register counts at or above 128 are secondary "
            "limits, not a second model."
        ),
        "",
        "#### Roofline",
        "",
        (
            "| Variant | Kernel | FLOPs (G) | DRAM (MB) | AI (FLOP/B) "
            "| Ridge (FLOP/B) | Peak TFLOPS | Peak DRAM GB/s "
            "| Roofline TFLOPS | Achieved TFLOPS | Achieved DRAM GB/s |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        capture = variant["captures"].get("ncu") or {}
        kernels = capture.get("kernels") or []
        if capture.get("status") != "ok" or not kernels:
            lines.append(_note_row(variant, capture, 9))
            continue
        for kernel in kernels:
            lines.append(
                f"| `{variant['variant']}` "
                f"| `{(kernel['kernel'] or '')[:48]}` "
                f"| {_num(_scaled(kernel.get('flops'), 1e9), 1)} "
                f"| {_num(_scaled(kernel.get('dram_bytes'), 1e6), 1)} "
                f"| {_num(kernel.get('ai_flop_per_byte'), 1)} "
                f"| {_num(kernel.get('ridge_flop_per_byte'), 1)} "
                f"| {_num(kernel.get('peak_tflops'), 1)} "
                f"| {_num(kernel.get('peak_bw_gbps'), 0)} "
                f"| {_num(kernel.get('roofline_tflops'), 1)} "
                f"| {_num(kernel.get('tflops'), 1)} "
                f"| {_num(kernel.get('achieved_bw_gbps'), 1)} |"
            )
    lines.append("")
    lines.extend(
        [
            "#### Bottleneck",
            "",
            (
                "| Variant | Kernel | Compute SOL % | Memory SOL % "
                "| DRAM SOL % | SOL % | Bound | Limiter |"
            ),
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for variant in variants:
        capture = variant["captures"].get("ncu") or {}
        kernels = capture.get("kernels") or []
        if capture.get("status") != "ok" or not kernels:
            lines.append(_note_row(variant, capture, 6))
            continue
        for kernel in kernels:
            lines.append(
                f"| `{variant['variant']}` "
                f"| `{(kernel['kernel'] or '')[:48]}` "
                f"| {_num(kernel.get('sm_pct'), 1)} "
                f"| {_num(kernel.get('mem_pct'), 1)} "
                f"| {_num(kernel.get('dram_pct'), 1)} "
                f"| {_num(kernel.get('sol_pct'), 1)} "
                f"| {kernel.get('bound') or '—'} "
                f"| {kernel.get('limiter') or '—'} |"
            )
    chart = problem.get("roofline_chart")
    if (
        chart
        and points
        and meta.get("peak_tflops") is not None
        and meta.get("peak_bw_gbps") is not None
        and meta.get("ridge_flop_per_byte") is not None
    ):
        href = _relative_href(exp_dir / chart, out_dir)
        lines.append("")
        lines.append(
            f"![Roofline for {problem['problem']}]({href})"
        )
    lines.append("")
    return lines


def _write_roofline_charts(exp_dir: Path, data: dict) -> None:
    """Render one SVG per problem with a roofline and remove stale
    roofline charts this report no longer produces."""
    reports_dir = exp_dir / "reports"
    expected = set()
    for problem in data.get("problems") or []:
        chart = problem.get("roofline_chart")
        meta = problem.get("roofline") or {}
        points = _roofline_points(problem["variants"])
        if (
            not chart
            or not points
            or meta.get("peak_tflops") is None
            or meta.get("peak_bw_gbps") is None
            or meta.get("ridge_flop_per_byte") is None
        ):
            continue
        target = exp_dir / chart
        expected.add(target.name)
        svg = _roofline_svg(
            problem["problem"],
            points,
            meta["peak_tflops"],
            meta["peak_bw_gbps"],
            meta["ridge_flop_per_byte"],
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(target, svg)
    for stale in reports_dir.glob("roofline-*.svg"):
        if stale.name not in expected:
            stale.unlink()


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
    merge = data.get("capture_merge")
    profilers_present = _profilers_present(variants)
    if not profilers_present:
        lines.append("_No profiler captures available for this problem._")
        lines.append("")
    for profiler in profilers_present:
        if profiler == "ncu":
            lines.extend(_ncu_section(variants, batch, merge))
            lines.extend(_roofline_section(problem, exp_dir, out_dir))
        elif profiler == "nsys":
            lines.extend(_nsys_section(variants, batch, merge))
        elif profiler == "proton":
            lines.extend(_proton_section(variants, merge))

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
    merge = data.get("capture_merge")
    batch = data.get("capture_batch")
    if merge:
        capture_ids = merge.get("capture_ids") or []
        lines.append(
            "- Capture selection: newest capture of each profiler across "
            f"{len(capture_ids)} batch(es) (`{merge.get('mode')}`)"
        )
        lines.append(
            "- Capture batches: "
            + (
                ", ".join(f"`{capture_id}`" for capture_id in capture_ids)
                if capture_ids
                else "none"
            )
        )
        sources = merge.get("profiler_batches") or {}
        if sources:
            lines.append(
                "- Profiler batches: "
                + "; ".join(
                    f"{profiler} ← "
                    + ", ".join(f"`{capture_id}`" for capture_id in ids)
                    for profiler, ids in sources.items()
                )
            )
    elif batch:
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
    lines.append(
        "- The Roofline and SOL section uses NCU's own Speed Of Light "
        "percentages; its arithmetic intensity and roofs come from the "
        "capture's executed tensor FLOPs, DRAM traffic, and peak-sustained "
        "rates, so they need `ncu --set full` and describe what the kernel "
        "actually executed, not an algorithmic FLOP model. DRAM counters "
        "exclude traffic served from L2, so a cache-friendly kernel shows a "
        "higher arithmetic intensity than its total memory work implies."
    )
    lines.append(
        "- Peak-sustained rates are the clocks NCU sampled during the "
        "capture; laptop power and thermal limits keep sustained "
        "performance below those peaks, and the limiter flags are "
        "heuristics over the same metrics."
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
    _write_roofline_charts(exp_dir, data)
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
        help="report this capture batch (default: the latest), or "
        "latest-per-profiler for the newest capture of each profiler "
        "across every batch",
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
