#!/usr/bin/env python3
"""Add optional LLM commentary to an experiment's code-diff analysis.

This CLI is deliberately separate from report-experiment.py: the report
stays host-side, offline, and standard-library-only. This command runs in
the KernelAgent container with the configured LLM provider, regenerates
the deterministic analysis in memory, and writes sidecar files:

  reports/code-diff-analysis.llm.md
  reports/code-diff-analysis.llm.json

The deterministic reports/code-diff-analysis.{md,json} files remain
authoritative and are never overwritten. Nothing is written to the
manifest, the kernels, the records, or the experiment status.

Usage:
  enrich-code-diffs.py <experiment> [--results-root DIR]
                       [--capture-batch ID|latest-per-profiler]
                       [--max-pairs N]

Exit codes: 0 all attempted pairs succeeded (or there was nothing to do),
1 at least one pair failed (the sidecar is partial), 2 the LLM
configuration/provider is unavailable (nothing was written).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import diff_analysis
from ka_common import (
    RESULTS_MOUNT_DEFAULT,
    _atomic_write_json,
    _atomic_write_text,
    _load_manifest,
    _load_module,
    infer_llm_backend,
    resolve_experiment,
)

PROMPT_VERSION = "code-diff-llm-v1"
MAX_PROMPT_CHARS = 32000
MAX_RESPONSE_CHARS = 4000
LLM_MARKDOWN = "reports/code-diff-analysis.llm.md"
LLM_JSON = "reports/code-diff-analysis.llm.json"


def _hint(backend: str) -> str:
    if backend == "anthropic":
        return "hint: export ANTHROPIC_API_KEY"
    return (
        "hint: export OPENAI_API_KEY (any non-empty value works for local\n"
        "      OpenAI-compatible servers, e.g. OPENAI_API_KEY=local)"
    )


def _report_module():
    path = Path(__file__).with_name("report-experiment.py")
    return _load_module("report_experiment", path, register=True)


def _load_provider():
    from utils.providers import get_model_provider
    from utils.providers.base import reasoning_effort_from_env

    return get_model_provider, reasoning_effort_from_env


def _flatten_pairs(doc):
    for problem in doc.get("problems") or []:
        for pair in problem.get("pairs") or []:
            yield problem, pair


def _diff_text(exp_dir, pair):
    path = pair.get("diff_path")
    if path:
        full = exp_dir / path
        if full.is_file():
            return full.read_text(encoding="utf-8", errors="replace")
    # The report normally wrote the diff already; reconstruct only if it
    # is missing so a read-only experiment still gets a prompt.
    source = pair.get("source") or {}
    from_path = (source.get("from") or {}).get("path")
    to_path = (source.get("to") or {}).get("path")
    if not from_path or not to_path:
        return ""
    from_file = exp_dir / from_path
    to_file = exp_dir / to_path
    if not from_file.is_file() or not to_file.is_file():
        return ""
    from_lines = from_file.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines(keepends=True)
    to_lines = to_file.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            from_lines,
            to_lines,
            fromfile=f"{pair.get('problem')}/{pair.get('from')}",
            tofile=f"{pair.get('problem')}/{pair.get('to')}",
        )
    )


def _prompt(problem, pair, diff_text):
    evidence = pair.get("evidence") or {}
    bench = evidence.get("bench") or {}
    ncu = evidence.get("ncu") or {}
    changes = []
    for change in pair.get("changes") or []:
        changes.append(
            {
                "category": change.get("category"),
                "title": change.get("title"),
                "why_it_matters": change.get("why_it_matters"),
                "excerpt": change.get("excerpt"),
            }
        )
    context = problem.get("case_context") or {}
    payload = {
        "problem": problem.get("problem"),
        "case": problem.get("case"),
        "case_spec": problem.get("case_spec"),
        "case_context": {
            "fields": context.get("fields"),
            "causal": context.get("causal"),
            "divisibility": context.get("divisibility"),
        },
        "from": pair.get("from"),
        "to": pair.get("to"),
        "from_role": pair.get("from_role"),
        "to_role": pair.get("to_role"),
        "sources": pair.get("source"),
        "bench": bench,
        "ncu": ncu,
        "deterministic_changes": changes,
        "deterministic_invariants": pair.get("invariants"),
        "deterministic_caveats": pair.get("caveats"),
        "deterministic_takeaway": pair.get("takeaway"),
    }
    instructions = (
        "You are reviewing a pairwise diff between Triton kernels from a "
        "KernelAgent experiment. Use only the evidence provided. Distinguish "
        "a mechanism from a measured cause: never say a change caused a "
        "speedup. Never invent numbers. If bench and NCU disagree, say so and "
        "do not claim a fixed-harness win for that pair. If a side has no "
        "usable NCU capture, say the attribution is inferred. Return only a "
        "JSON object with exactly these keys:\n"
        '{"pair_summary": "2-4 sentences", '
        '"changes": [{"category": "...", "title": "...", '
        '"why_it_matters": "...", "evidence": "...", "fair_note": "..."}], '
        '"invariants": ["..."], "caveats": ["..."], '
        '"takeaway": "1-2 sentences"}\n'
        "Stay under 800 words. Do not wrap the JSON in a Markdown fence.\n\n"
    )
    prefix = instructions + json.dumps(payload, indent=2) + "\n\nFull diff:\n"
    remaining = MAX_PROMPT_CHARS - len(prefix) - 100
    truncated = False
    if diff_text and len(diff_text) > remaining:
        cut = diff_text[: max(0, remaining)]
        newline = cut.rfind("\n")
        if newline > 0:
            cut = cut[:newline]
        diff_text = cut + "\n... diff truncated ...\n"
        truncated = True
    return prefix + (diff_text or "(diff unavailable)\n"), truncated


def _string_list(value):
    if not isinstance(value, list):
        return []
    return [str(item)[:500] for item in value[:10] if item]


def _parse_response(text):
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response is not a JSON object")
        stripped = stripped[start : end + 1]
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("response JSON is not an object")
    if not isinstance(data.get("pair_summary"), str):
        raise ValueError("missing pair_summary")
    if not isinstance(data.get("takeaway"), str):
        raise ValueError("missing takeaway")

    changes = data.get("changes")
    if not isinstance(changes, list):
        raise ValueError("changes is not a list")
    normalized = []
    for change in changes[:10]:
        if not isinstance(change, dict):
            continue
        title = change.get("title") or change.get("category")
        why = change.get("why_it_matters")
        if not isinstance(title, str) or not isinstance(why, str):
            continue
        normalized.append(
            {
                "category": str(change.get("category") or "other_code_change"),
                "title": title[:300],
                "why_it_matters": why[:2000],
                "evidence": str(change.get("evidence") or "")[:2000],
                "fair_note": str(change.get("fair_note") or "")[:2000],
            }
        )
    invariants = _string_list(data.get("invariants"))
    caveats = _string_list(data.get("caveats"))
    return {
        "pair_summary": data["pair_summary"][:2000],
        "changes": normalized,
        "invariants": invariants,
        "caveats": caveats,
        "takeaway": data["takeaway"][:1000],
    }


def _source_hashes(doc):
    hashes = {}
    for problem in doc.get("problems") or []:
        for pair in problem.get("pairs") or []:
            key = (
                f"{pair.get('problem')}/{pair.get('from')}/{pair.get('to')}"
            )
            source = pair.get("source") or {}
            hashes[key] = {
                "from": (source.get("from") or {}).get("sha256"),
                "to": (source.get("to") or {}).get("sha256"),
            }
    return hashes


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
        default=Path(
            os.environ.get("RESULTS_MOUNT", RESULTS_MOUNT_DEFAULT)
        ),
        help="results mount holding experiments/",
    )
    parser.add_argument(
        "--capture-batch",
        default=None,
        help="capture batch the deterministic analysis uses (default: "
        "latest), or latest-per-profiler",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="enrich at most this many pairs (cost control)",
    )
    args = parser.parse_args()

    exp_dir = resolve_experiment(args.results_root, args.experiment).resolve()

    model = os.environ.get("OPENAI_MODEL", "").strip()
    if not model:
        print(
            "ERROR: OPENAI_MODEL is not set (pass it on the make command "
            "line)",
            file=sys.stderr,
        )
        return 2
    backend = os.environ.get("LLM_BACKEND") or infer_llm_backend(model)

    try:
        get_model_provider, reasoning_effort_from_env = _load_provider()
        provider = get_model_provider(model)
    except (ImportError, ValueError) as exc:
        print(
            f"ERROR: no provider for model '{model}': {exc}",
            file=sys.stderr,
        )
        print(_hint(backend), file=sys.stderr)
        return 2
    if provider.name != backend:
        print(
            f"ERROR: expected the {backend} provider for model '{model}', "
            f"got '{provider.name}'",
            file=sys.stderr,
        )
        print(_hint(backend), file=sys.stderr)
        return 2
    if not provider.is_available():
        print(
            f"ERROR: the {provider.name} provider is not available",
            file=sys.stderr,
        )
        print(_hint(backend), file=sys.stderr)
        return 2
    try:
        effort = reasoning_effort_from_env()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report = _report_module()
    manifest = _load_manifest(exp_dir)
    data = report.build_report_data(exp_dir, manifest, args.capture_batch)
    records = report.source_diff_records(exp_dir, data)
    doc = diff_analysis.build_analysis(exp_dir, data, records)

    deterministic = exp_dir / diff_analysis.REPORT_JSON
    if deterministic.is_file():
        try:
            old = json.loads(deterministic.read_text(encoding="utf-8"))
            if old.get("fingerprint") != doc.get("fingerprint"):
                print(
                    "WARNING: reports/code-diff-analysis.json has a "
                    "different fingerprint; the existing deterministic "
                    "report is stale. The LLM sidecar uses the current "
                    "sources.",
                    file=sys.stderr,
                )
        except (OSError, ValueError):
            print(
                "WARNING: could not read the existing deterministic "
                "analysis JSON",
                file=sys.stderr,
            )

    entries = []
    lookup = {}
    attempted = 0
    failures = 0
    analysis_failures = 0
    skipped = 0
    for problem, pair in _flatten_pairs(doc):
        key = (pair.get("problem"), pair.get("from"), pair.get("to"))
        if pair.get("error"):
            entry = {
                "problem": pair.get("problem"),
                "from": pair.get("from"),
                "to": pair.get("to"),
                "status": "skipped",
                "error": f"deterministic analysis failed: {pair['error']}",
                "response": None,
            }
            skipped += 1
            analysis_failures += 1
            entries.append(entry)
            lookup[key] = entry
            continue
        if args.max_pairs is not None and attempted >= args.max_pairs:
            entry = {
                "problem": pair.get("problem"),
                "from": pair.get("from"),
                "to": pair.get("to"),
                "status": "skipped",
                "error": "max-pairs limit",
                "response": None,
            }
            skipped += 1
            entries.append(entry)
            lookup[key] = entry
            continue

        attempted += 1
        diff_text = _diff_text(exp_dir, pair)
        prompt, truncated = _prompt(problem, pair, diff_text)
        raw = ""
        try:
            response = provider.get_response(
                model, [{"role": "user", "content": prompt}]
            )
            raw = response.content or ""
            parsed = _parse_response(raw)
            entry = {
                "problem": pair.get("problem"),
                "from": pair.get("from"),
                "to": pair.get("to"),
                "status": "ok",
                "error": None,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "prompt_chars": len(prompt),
                "diff_truncated": truncated,
                "response": parsed,
                "raw_response": None,
            }
        except Exception as exc:  # noqa: BLE001 - one failed pair must
            # not discard the rest of the sidecar
            failures += 1
            entry = {
                "problem": pair.get("problem"),
                "from": pair.get("from"),
                "to": pair.get("to"),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "prompt_chars": len(prompt),
                "diff_truncated": truncated,
                "response": None,
                "raw_response": raw[:MAX_RESPONSE_CHARS],
            }
        entries.append(entry)
        lookup[key] = entry

    if not entries:
        status = "empty"
    elif failures or analysis_failures:
        status = "partial"
    else:
        status = "ok"

    provenance = {
        "model": model,
        "backend": backend,
        "provider": provider.name,
        "reasoning_effort": effort,
        "prompt_version": PROMPT_VERSION,
        "max_prompt_chars": MAX_PROMPT_CHARS,
        "deterministic_fingerprint": doc.get("fingerprint"),
        "deterministic_generated_at": doc.get("generated_at"),
        "capture_selection": doc.get("capture_selection"),
        "source_sha256": _source_hashes(doc),
    }
    llm_doc = {
        "schema_version": 1,
        "experiment_id": doc.get("experiment_id"),
        "generated_at": datetime.now().isoformat(),
        "status": status,
        "provenance": provenance,
        "pairs": entries,
        "summary": (
            f"LLM commentary for {attempted} of "
            f"{len(entries)} pairs; {failures} failed, "
            f"{analysis_failures} analysis-failed, "
            f"{skipped - analysis_failures} skipped."
        ),
        "paths": {"markdown": LLM_MARKDOWN, "json": LLM_JSON},
    }

    reports_dir = exp_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(reports_dir / "code-diff-analysis.llm.json", llm_doc)
    llm_render = {
        "provenance": provenance,
        "_lookup": lookup,
    }
    _atomic_write_text(
        reports_dir / "code-diff-analysis.llm.md",
        diff_analysis.render_markdown(doc, llm=llm_render),
    )
    print(f"LLM analysis: {reports_dir / 'code-diff-analysis.llm.md'}")
    if status == "partial":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
