#!/usr/bin/env python3
"""Deterministic analysis of an experiment's pairwise kernel diffs.

This module is imported by report-experiment.py and by the optional
enrich-code-diffs.py CLI. It is deliberately standard-library-only: the
host report must run without torch, triton, a GPU, or an LLM.

The analysis turns the pairwise unified diffs the report already writes
into a human-readable companion document:

  - changed source regions are found with difflib.SequenceMatcher (not by
    parsing the written .diff hunks, which merge unrelated edits)
  - recognized Triton constructs get a mechanism explanation and a small
    curated excerpt
  - benchmark and Nsight Compute evidence is reported separately, with
    explicit "inferred, not measured on both sides" caveats
  - invariants present on both sides ("what did not change") are listed

The document is generated from the report's normalized data plus the
diff records. It never modifies kernels, records, or the manifest.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import io
import json
import os
import re
import tokenize
from datetime import datetime
from pathlib import Path

from ka_common import _atomic_write_json, _atomic_write_text

# Separate contract from comparison.json: this document has its own
# version so the report and the analysis can evolve independently.
ANALYSIS_SCHEMA_VERSION = 1

MAX_EXCERPTS_PER_PAIR = 6
MAX_EXCERPT_CHANGED_LINES = 24
MAX_INVARIANTS = 6

REPORT_MARKDOWN = "reports/code-diff-analysis.md"
REPORT_JSON = "reports/code-diff-analysis.json"

# Capture statuses the report already uses, repeated here so this module
# can explain a missing/refused capture without importing the report.
_CAPTURE_NOTES = {
    "missing_kernel": "archived kernel missing — refused",
    "hash_mismatch": "kernel hash mismatch — refused",
    "failed": "capture failed",
    "not_captured": "not captured",
    "superseded": "capture describes a previous kernel",
}

# Known problem case_spec layouts. Unknown problems fall back to
# positional names (arg0, arg1, ...), so the analysis still runs.
_CASE_FIELDS = {
    "matmul_bias": ("M", "N", "K", "dtype"),
    "flash_attention": ("B", "H", "S", "D", "causal", "dtype"),
}

# Case dimensions that a `SYMBOL % TILE == 0` guard can refer to.
_DIM_ALIASES = {
    "M": "M",
    "N": "N",
    "K": "K",
    "B": "B",
    "H": "H",
    "S": "S",
    "SEQ": "S",
    "SEQUENCE": "S",
    "SEQLEN": "S",
    "D": "D",
    "HEAD_DIM": "D",
    "HEAD_SIZE": "D",
}


def _rx(*patterns: str):
    return tuple(re.compile(pattern) for pattern in patterns)


# Ordered detection rules. Priority expresses how directly a change can
# affect the hot loop, not a measured share of any speedup. The first
# rule in a region wins; all matches are kept for transparency.
_RULES = (
    {
        "key": "mask_elision",
        "priority": 10,
        "label": "Masking and predicated accesses",
        "patterns": _rx(
            r"\bmask\s*=",
            r"\bother\s*=",
            r"\btl\.where\s*\(",
            r"%\s*[A-Za-z_]\w*\s*==\s*0",
            r"\bif\s+[^\n]*%[^\n]*==\s*0",
        ),
        "why": (
            "Predicated loads/stores or tensor selects are added or removed. "
            "When the case dimensions divide the tile, a `% == 0` guard can "
            "resolve at compile time and the hot loop issues unconditional "
            "accesses; otherwise the masked path remains necessary."
        ),
    },
    {
        "key": "autotune_space",
        "priority": 9,
        "label": "Autotune candidate space",
        "patterns": _rx(
            r"@triton\.autotune",
            r"triton\.Config\s*\(",
            r"\bconfigs\s*=",
        ),
        "why": (
            "The candidate tile/warp/stage space changes. A different "
            "candidate set can select a better register/shared-memory/"
            "occupancy balance. The source alone does not prove which "
            "configuration ran; NCU grid/block/occupancy is the evidence."
        ),
    },
    {
        "key": "pipelining",
        "priority": 8,
        "label": "Software pipelining",
        "patterns": _rx(
            r"\btl\.range\s*\(",
            r"\bnum_stages\s*=\s*\d",
        ),
        "why": (
            "Pipeline depth or loop form changes: more stages overlap global "
            "loads with tensor-core work at higher shared-memory/register "
            "cost; fewer stages reduce pressure but expose load latency."
        ),
    },
    {
        "key": "causal_specialization",
        "priority": 8,
        "label": "Causal loop specialization",
        "patterns": _rx(
            r"\bIS_CAUSAL\b",
            r"\bMASK_CAUSAL\b",
            r"\bprefix_end\b",
            r"\bdiagonal_end\b",
            r"\bpos_q\b",
            r"\bpos_k\b",
            r"\bend_n\b",
        ),
        "why": (
            "The causal loop is split into an unmasked prefix and a masked "
            "diagonal, or fully masked blocks are pruned. Only causal cases "
            "are affected."
        ),
    },
    {
        "key": "grid_addressing",
        "priority": 7,
        "label": "Grid and program addressing",
        "patterns": _rx(
            r"\btl\.program_id\s*\(",
            r"\bgrid\s*=",
            r"\bbase\s*=",
            r"\bB_Seqlen\b",
            r"\bB_Start_Loc\b",
            r"\bbatch_head\b",
            r"\bcur_batch\b",
            r"\bcur_head\b",
        ),
        "why": (
            "Program-ID mapping, launch geometry, or address construction "
            "changes. Flattening axes or replacing per-program metadata loads "
            "with direct offsets removes fixed integer/global-load work per "
            "CTA and changes how tiles map to L2."
        ),
    },
    {
        "key": "constexpr_addressing",
        "priority": 7,
        "label": "Compile-time strides and index arithmetic",
        "patterns": _rx(
            r":\s*tl\.constexpr",
            r"\bstride_[A-Za-z_]\w*",
            r"\bSTRIDE[A-Z0-9_]*\b",
            r"%\s*(?:M|N|K|SEQ|S)\b",
            r"\btl\.multiple_of\s*\(",
            r"\btl\.assume\s*\(",
        ),
        "why": (
            "Runtime dimensions/strides become `tl.constexpr` and/or modulo "
            "wrapping is removed. Compile-time constants let Triton fold "
            "address arithmetic, remove integer divide/modulo from every "
            "element's index path, and prove alignment."
        ),
    },
    {
        "key": "grouped_swizzle",
        "priority": 8,
        "label": "Grouped CTA order",
        "patterns": _rx(
            r"\bGROUP_SIZE\w*\b",
            r"\bgroup_id\b",
            r"\bpid_m\b",
            r"\bpid_n\b",
        ),
        "why": (
            "Grouped CTA ordering changes which output tiles run "
            "concurrently. This is an L2-reuse heuristic for the A/B "
            "panels, not an arithmetic change."
        ),
    },
    {
        "key": "online_softmax",
        "priority": 6,
        "label": "Online-softmax recurrence",
        "patterns": _rx(
            r"\bm_i\b",
            r"\bl_i\b",
            r"\bnext_m\b",
            r"\balpha\b",
            r"\bexp2\s*\(",
            r"\btl\.dot\s*\(\s*p",
        ),
        "why": (
            "The online-softmax recurrence is touched. This is "
            "correctness-critical: changes affect numerical stability and "
            "rescale/dot work, not just scheduling; verify with the fixed "
            "test."
        ),
    },
    {
        "key": "dtype_math",
        "priority": 5,
        "label": "Dtype and math path",
        "patterns": _rx(
            r"\.to\s*\(",
            r"\btl\.float32\b",
            r"\bfloat16\b",
            r"\bbfloat16\b",
            r"\b(?:tl\.)?exp\s*\(",
            r"\b(?:tl\.)?exp2\s*\(",
            r"-1\.0e8",
            r"-float\([\"']inf[\"']\)",
        ),
        "why": (
            "Dtype conversion points or math functions change. fp32 "
            "accumulation and fewer conversions trade precision against "
            "conversion/instruction cost; `exp2` avoids a base conversion."
        ),
    },
    {
        "key": "host_launch",
        "priority": 4,
        "label": "Host wrapper and launch contract",
        "patterns": _rx(
            r"\bdef\s+kernel_function\b",
            r"@triton\.jit",
            r"\bgrid\s*=\s*lambda",
            r"\b[A-Za-z_]\w*\[grid\]\s*\(",
            r"\bBLOCK_[A-Z]+\s*=",
            r"\btorch\.empty\s*\(",
            r"\bassert\b",
            r"\braise\s+ValueError\b",
        ),
        "why": (
            "Host-side wrapper, launch geometry, or contract checks change. "
            "This can alter grid shape and argument passing without changing "
            "device code; assertions and validation are not performance-"
            "relevant."
        ),
    },
    {
        "key": "preamble",
        "priority": 1,
        "label": "Preamble and imports",
        "patterns": _rx(
            r"^\s*(?:import|from)\s",
            r"^\s*#",
            r"^\s*(?:\"\"\"|''')",
        ),
        "why": "Only imports, comments, docstrings, or license text differ.",
    },
    {
        "key": "unclassified",
        "priority": 0,
        "label": "Unclassified source change",
        "patterns": (),
        "why": (
            "No known Triton transform pattern matched. The excerpt is shown "
            "for inspection; do not infer a performance cause from it."
        ),
    },
)
_RULE_BY_KEY = {rule["key"]: rule for rule in _RULES}

_INVARIANT_PROBES = (
    {
        "key": "grouped_cta_swizzle",
        "title": "Grouped CTA order preserved",
        "patterns": _rx(r"\bgroup_id\b", r"\bpid_m\b"),
        "detail": (
            "Both keep a grouped CTA order (`group_id`/`pid_m`) for L2 reuse."
        ),
    },
    {
        "key": "online_softmax",
        "title": "Online-softmax recurrence preserved",
        "patterns": _rx(
            r"\b(?:tl\.)?(?:math\.)?exp2?\s*\(",
            r"\btl\.dot\s*\(",
            r"\b(?:m_i|next_m|row_max)\b",
            r"\b(?:l_i|row_sum)\b",
        ),
        "detail": (
            "Both use the online-softmax recurrence (running max/denominator, "
            "`exp2`, and a P·V dot); no full score matrix is materialized."
        ),
    },
    {
        "key": "fp32_accumulation",
        "title": "fp32 accumulation preserved",
        "patterns": _rx(r"\btl\.float32\b"),
        "detail": "Both accumulate in fp32.",
    },
    {
        "key": "masked_store",
        "title": "Masked output store preserved",
        "patterns": _rx(r"tl\.store\s*\(", r"\bmask\s*="),
        "detail": "Both mask the output store.",
    },
    {
        "key": "autotune_key",
        "title": "Autotune key preserved",
        "compare": "autotune_key",
        "detail": "Both autotune on the same key list.",
    },
    {
        "key": "kernel_entry",
        "title": "Kernel entry signature preserved",
        "compare": "signature",
        "detail": "Both expose the same `kernel_function` signature.",
    },
)


# ---------------------------------------------------------------------------
# Small formatting helpers


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value, digits=3):
    number = _num(value)
    if number is None:
        return "—"
    return f"{number:.{digits}f}"


def _sha(value):
    return f"`{value[:12]}…`" if value else "—"


def _line_span(label, start, end):
    if start is None or end is None:
        return f"{label} —"
    if end <= start:
        return f"{label} after line {start}"
    return f"{label} {start + 1}-{end}"


def _first_line(lines, flags, pattern):
    for index, line in enumerate(lines):
        if flags[index] and pattern.search(line):
            return index + 1
    return None


def _has_all(text, patterns):
    return all(pattern.search(text) for pattern in patterns)


def _fence_for(text):
    run = best = 0
    for char in text:
        if char == "`":
            run += 1
            best = max(best, run)
        else:
            run = 0
    return "`" * max(3, best + 1)


# ---------------------------------------------------------------------------
# Source reading and code-line filtering


def _code_flags(lines):
    """True for lines carrying code; comments and docstrings are noise
    for the construct regexes. Falls back to a line-oriented state
    machine when the source is not tokenizable (draft or invalid code)."""
    flags = [False] * len(lines)
    code_lines = set()
    string_lines = set()
    try:
        tokens = tokenize.generate_tokens(
            io.StringIO("".join(lines)).readline
        )
        for token in tokens:
            if token.type in (
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.COMMENT,
            ):
                continue
            start, end = token.start[0] - 1, token.end[0] - 1
            target = (
                string_lines if token.type == tokenize.STRING else code_lines
            )
            for index in range(start, min(end, len(lines) - 1) + 1):
                target.add(index)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        in_doc = None
        for index, line in enumerate(lines):
            stripped = line.strip()
            if in_doc:
                string_lines.add(index)
                if in_doc in stripped:
                    in_doc = None
                continue
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith(('"""', "'''")):
                string_lines.add(index)
                quote = stripped[:3]
                if stripped.count(quote) == 1:
                    in_doc = quote
                continue
            code_lines.add(index)

    for index in range(len(lines)):
        if index in code_lines:
            flags[index] = True
        elif index in string_lines:
            flags[index] = False
        else:
            stripped = lines[index].strip()
            flags[index] = bool(stripped) and not stripped.startswith("#")
    return flags


def _read_source(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    flags = _code_flags(lines)
    code = "\n".join(line for line, flag in zip(lines, flags) if flag)
    return {"lines": lines, "flags": flags, "code": code}


# ---------------------------------------------------------------------------
# Case context and tile/divisibility extraction


def _extract_tiles(*texts):
    tiles = {}
    assign = re.compile(
        r"[\"']?(BLOCK[A-Z0-9_]*|BM|BN|BK)[\"']?\s*[:=]\s*(\d+)"
    )
    loop = re.compile(
        r"for\s+(BLOCK\w*|BM|BN|BK)\s+in\s+\[([0-9,\s]+)\]"
    )
    for text in texts:
        for match in assign.finditer(text):
            tiles.setdefault(match.group(1), set()).add(int(match.group(2)))
        for match in loop.finditer(text):
            values = {
                int(value)
                for value in re.findall(r"\d+", match.group(2))
            }
            tiles.setdefault(match.group(1), set()).update(values)
    return {name: sorted(values) for name, values in sorted(tiles.items())}


def _divisibility(int_fields, tiles, *texts):
    guards = set()
    guard = re.compile(
        r"\b([A-Za-z_]\w*)\s*%\s*([A-Za-z_]\w*)\s*==\s*0"
    )
    for text in texts:
        for match in guard.finditer(text):
            guards.add((match.group(1), match.group(2)))

    results = []
    for dim_symbol, tile_symbol in sorted(guards):
        dim_name = _DIM_ALIASES.get(dim_symbol.upper(), dim_symbol)
        dim = int_fields.get(dim_name)
        values = tiles.get(tile_symbol) or []
        if dim is None or not values:
            continue
        results.append(
            {
                "symbol": dim_symbol,
                "dim": dim,
                "tile_symbol": tile_symbol,
                "tile_values": values,
                "all_divide": all(dim % value == 0 for value in values),
            }
        )
    return results


def _case_context(problem, *code_texts):
    spec = problem.get("case_spec")
    fields = {}
    value = None
    try:
        value = ast.literal_eval(spec) if isinstance(spec, str) else spec
    except (ValueError, SyntaxError):
        value = None
    names = _CASE_FIELDS.get(problem.get("problem"), ())
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            name = names[index] if index < len(names) else f"arg{index}"
            fields[name] = item

    int_fields = {
        name: item
        for name, item in fields.items()
        if isinstance(item, int) and not isinstance(item, bool)
    }
    causal = next(
        (item for item in fields.values() if isinstance(item, bool)), None
    )
    tiles = _extract_tiles(*code_texts)
    return {
        "case_spec": spec,
        "fields": fields,
        "integer_fields": int_fields,
        "causal": causal,
        "tiles": tiles,
        "divisibility": _divisibility(int_fields, tiles, *code_texts),
    }


# ---------------------------------------------------------------------------
# Changed regions and excerpts

_SPLIT_HINT = re.compile(
    r"^\s*(?:def |class |@|for |if |elif |else:|while |return |#)"
)


def _chunk_ranges(lines):
    """Ranges of roughly 20-50 lines, split at blank lines or statement
    boundaries when possible."""
    ranges = []
    start = 0
    for index, line in enumerate(lines):
        if index == start:
            continue
        boundary = _SPLIT_HINT.match(line) or not line.strip()
        if boundary and index - start >= 20:
            ranges.append((start, index))
            start = index
        elif index - start >= 50:
            ranges.append((start, index))
            start = index
    if start < len(lines):
        ranges.append((start, len(lines)))
    return [(first, last) for first, last in ranges if last > first]


def _single_sided_chunk(region, removed, added, removed_offset, added_offset,
                        from_flags, to_flags):
    from_start = region["from_start"] + removed_offset
    to_start = region["to_start"] + added_offset
    return {
        "from_start": from_start,
        "from_end": from_start + len(removed),
        "to_start": to_start,
        "to_end": to_start + len(added),
        "removed": removed,
        "added": added,
        "removed_code": [
            line
            for offset, line in enumerate(removed)
            if from_flags[from_start + offset]
        ],
        "added_code": [
            line
            for offset, line in enumerate(added)
            if to_flags[to_start + offset]
        ],
        "context_before": [],
        "context_after": [],
    }


def _split_region(region, from_flags, to_flags):
    """Split an oversized replace block at logical boundaries.

    A whole-file rewrite often arrives as one SequenceMatcher replace
    opcode with no reliable line alignment. Showing a proportional
    pairing of unrelated lines would be misleading, so oversized regions
    become one-sided chunks: each added or removed block keeps its own
    line range and the full diff link carries the exact pairing.
    """
    total = len(region["removed"]) + len(region["added"])
    if total <= MAX_EXCERPT_CHANGED_LINES * 2:
        return [region]

    chunks = []
    for added_start, added_end in _chunk_ranges(region["added"]):
        chunks.append(
            _single_sided_chunk(
                region,
                [],
                region["added"][added_start:added_end],
                0,
                added_start,
                from_flags,
                to_flags,
            )
        )
    for removed_start, removed_end in _chunk_ranges(region["removed"]):
        chunks.append(
            _single_sided_chunk(
                region,
                region["removed"][removed_start:removed_end],
                [],
                removed_start,
                0,
                from_flags,
                to_flags,
            )
        )
    return chunks or [region]


def _regions(from_lines, to_lines, from_flags, to_flags):
    matcher = difflib.SequenceMatcher(
        None, from_lines, to_lines, autojunk=False
    )
    ops = matcher.get_opcodes()
    regions = []
    index = 0
    while index < len(ops):
        tag = ops[index][0]
        if tag == "equal":
            index += 1
            continue

        start = index
        # Merge edits separated by a single unchanged line, so one logical
        # change is not split into two excerpts.
        while (
            index + 2 < len(ops)
            and ops[index + 1][0] == "equal"
            and ops[index + 1][2] - ops[index + 1][1] <= 1
            and ops[index + 2][0] != "equal"
        ):
            index += 2
        end = index

        from_start, from_end = ops[start][1], ops[end][2]
        to_start, to_end = ops[start][3], ops[end][4]
        removed = from_lines[from_start:from_end]
        added = to_lines[to_start:to_end]
        if removed or added:
            before = ops[start - 1] if start > 0 else None
            after = ops[end + 1] if end + 1 < len(ops) else None
            if before is not None and before[0] == "equal":
                context_before = from_lines[before[1] : before[2]][-3:]
            else:
                context_before = from_lines[
                    max(0, from_start - 3) : from_start
                ]
            if after is not None and after[0] == "equal":
                context_after = from_lines[after[1] : after[2]][:3]
            else:
                context_after = from_lines[from_end : from_end + 3]

            regions.extend(
                _split_region(
                    {
                        "from_start": from_start,
                        "from_end": from_end,
                        "to_start": to_start,
                        "to_end": to_end,
                        "removed": removed,
                        "added": added,
                        "removed_code": [
                            line
                            for offset, line in enumerate(removed)
                            if from_flags[from_start + offset]
                        ],
                        "added_code": [
                            line
                            for offset, line in enumerate(added)
                            if to_flags[to_start + offset]
                        ],
                        "context_before": context_before,
                        "context_after": context_after,
                    },
                    from_flags,
                    to_flags,
                )
            )
        index += 1
    return regions


def _count(rule, lines):
    patterns = rule["patterns"]
    if not patterns:
        return 0
    return sum(
        1
        for line in lines
        if any(pattern.search(line) for pattern in patterns)
    )


def _classify(region):
    matches = []
    for rule in _RULES:
        removed = _count(rule, region["removed_code"])
        added = _count(rule, region["added_code"])
        if removed or added:
            matches.append((rule, removed, added))

    non_preamble = [
        match for match in matches if match[0]["key"] != "preamble"
    ]
    pool = non_preamble or matches
    if not pool:
        rule, removed, added = (
            _RULE_BY_KEY["unclassified"],
            0,
            0,
        )
    else:
        rule, removed, added = max(
            pool,
            key=lambda match: (
                match[0]["priority"],
                match[1] + match[2],
                len(region["removed_code"]) + len(region["added_code"]),
            ),
        )

    if not non_preamble and matches:
        # Only imports/comments/docstrings differ: not worth an excerpt.
        return None, None, [match[0]["key"] for match in matches]

    if added > removed:
        direction = "to_adds"
    elif removed > added:
        direction = "to_removes"
    else:
        direction = "mixed"
    return rule, direction, [match[0]["key"] for match in matches]


def _change_why(rule, case_context):
    why = rule["why"]
    context = case_context or {}
    if rule["key"] == "mask_elision":
        divisibility = context.get("divisibility") or []
        if divisibility:
            if all(item.get("all_divide") for item in divisibility):
                why += (
                    " For this case, every detected candidate tile divides "
                    "the guarded dimension, so the guard can resolve at "
                    "compile time."
                )
            else:
                why += (
                    " Not every detected candidate tile divides the guarded "
                    "dimension; only divisible configurations take the "
                    "unmasked path."
                )
    elif rule["key"] == "causal_specialization":
        causal = context.get("causal")
        if causal is False:
            why += (
                " This case is non-causal, so the change cannot affect its "
                "hot loop; it matters for causal cases."
            )
        elif causal is True:
            why += (
                " This case is causal, so the change can affect its hot loop."
            )
    return why


def _render_excerpt(region):
    changed = [("remove", line) for line in region["removed"]]
    changed.extend(("add", line) for line in region["added"])
    omitted = 0
    if len(changed) > MAX_EXCERPT_CHANGED_LINES:
        head = changed[: MAX_EXCERPT_CHANGED_LINES // 2]
        tail = changed[-MAX_EXCERPT_CHANGED_LINES // 2 :]
        omitted = len(changed) - len(head) - len(tail)
        changed = head + [("omit", omitted)] + tail

    rendered = []
    for line in region["context_before"]:
        rendered.append(" " + line.rstrip("\r\n"))
    for kind, line in changed:
        if kind == "omit":
            rendered.append(f"... {line} changed lines omitted ...")
        else:
            prefix = "-" if kind == "remove" else "+"
            rendered.append(prefix + line.rstrip("\r\n"))
    for line in region["context_after"]:
        rendered.append(" " + line.rstrip("\r\n"))
    return "\n".join(rendered)


def _normalized_signature(region):
    text = " ".join(region["removed_code"] + region["added_code"])
    return re.sub(r"\s+", " ", text).strip()[:2000]


def _similar(left, right):
    if not left or not right:
        return False
    return (
        difflib.SequenceMatcher(None, left, right).ratio() >= 0.9
    )


def _change_title(rule, direction):
    if rule["key"] == "mask_elision":
        return {
            "to_adds": "Mask-free fast path added",
            "to_removes": "Mask-free fast path removed",
            "mixed": "Masking or predication changed",
        }[direction]
    if rule["key"] == "autotune_space":
        return {
            "to_adds": "Autotune candidate space expanded or changed",
            "to_removes": "Autotune candidate space reduced or changed",
            "mixed": "Autotune candidate space changed",
        }[direction]
    direction_phrase = {
        "to_adds": "added in `to`",
        "to_removes": "removed in `to`",
        "mixed": "changed",
    }[direction]
    return f"{rule['label']} — {direction_phrase}"


def _select_changes(regions, case_context, seen, pair_label):
    candidates = []
    for region in regions:
        if not region["removed_code"] and not region["added_code"]:
            # Comment/docstring-only edit: not a performance mechanism.
            continue
        rule, direction, matched = _classify(region)
        if rule is None:
            continue
        candidates.append(
            {
                "region": region,
                "rule": rule,
                "direction": direction,
                "matched_categories": matched,
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["rule"]["priority"],
            item["region"]["from_start"],
        )
    )
    # Prefer one excerpt per category before spending the remaining
    # budget on repeated regions of the same kind.
    selected = []
    categories_seen = set()
    for item in candidates:
        category = item["rule"]["key"]
        if category in categories_seen or category == "unclassified":
            continue
        selected.append(item)
        categories_seen.add(category)
        if len(selected) >= MAX_EXCERPTS_PER_PAIR:
            break
    if len(selected) < MAX_EXCERPTS_PER_PAIR:
        for item in candidates:
            if item not in selected:
                selected.append(item)
            if len(selected) >= MAX_EXCERPTS_PER_PAIR:
                break
    selected.sort(
        key=lambda item: (
            -item["rule"]["priority"],
            item["region"]["from_start"],
        )
    )
    changes = []
    for item in selected:
        rule = item["rule"]
        region = item["region"]
        signature = _normalized_signature(region)
        repeated_from = None
        for previous_signature, previous_label in seen:
            if previous_label == pair_label:
                continue
            if _similar(previous_signature, signature):
                repeated_from = previous_label
                break
        excerpt = None
        if repeated_from is None:
            excerpt = _render_excerpt(region)
            seen.append((signature, pair_label))

        direction = item["direction"]
        changes.append(
            {
                "category": rule["key"],
                "priority": rule["priority"],
                "title": _change_title(rule, direction),
                "direction": direction,
                "matched_categories": item["matched_categories"],
                "line_ranges": {
                    "from_start": region["from_start"],
                    "from_end": region["from_end"],
                    "to_start": region["to_start"],
                    "to_end": region["to_end"],
                },
                "excerpt": excerpt,
                "why_it_matters": _change_why(rule, case_context),
                "repeated_from": repeated_from,
            }
        )
    return changes


# ---------------------------------------------------------------------------
# Evidence


def _dominant_kernel(capture):
    kernels = capture.get("kernels") or []
    if not kernels:
        return None
    return max(
        kernels,
        key=lambda kernel: _num(kernel.get("duration_avg_us")) or 0.0,
    )


def _ncu_side(variant):
    capture = (variant.get("captures") or {}).get("ncu") or {}
    status = capture.get("status") or "not_captured"
    if status != "ok":
        return {
            "status": status,
            "note": _CAPTURE_NOTES.get(status, "no metrics"),
        }
    kernel = _dominant_kernel(capture)
    if kernel is None:
        return {"status": "failed", "note": "no kernel rows"}
    return {
        "status": status,
        "kernels_count": len(capture.get("kernels") or []),
        "kernel": kernel.get("kernel"),
        "duration_avg_us": _num(kernel.get("duration_avg_us")),
        "sm_pct": _num(kernel.get("sm_pct")),
        "mem_pct": _num(kernel.get("mem_pct")),
        "dram_pct": _num(kernel.get("dram_pct")),
        "occupancy_pct": _num(kernel.get("occupancy_pct")),
        "regs": _num(kernel.get("regs")),
        "tflops": _num(kernel.get("tflops")),
        "sol_pct": _num(kernel.get("sol_pct")),
        "bound": kernel.get("bound"),
        "limiter": kernel.get("limiter"),
        "grid": _num(kernel.get("grid")),
        "block": _num(kernel.get("block")),
        "launches": _num(kernel.get("launches")),
    }


def _external_note(from_variant, to_variant, from_code, to_code):
    for variant, code in (
        (from_variant, from_code),
        (to_variant, to_code),
    ):
        if variant.get("kind") != "external":
            continue
        features = []
        for label, pattern in (
            ("variable sequence lengths", r"\bB_Seqlen\b"),
            ("GQA", r"\bkv_group_num\b|\bnum_queries_per_kv\b"),
            ("sliding windows", r"\bsliding_window\b|\bwindow_size\b"),
            ("attention sinks", r"\bsink\w*\b"),
        ):
            if re.search(pattern, code):
                features.append(label)
        note = (
            f"`{variant['variant']}` is an archived external baseline"
            f" (`{variant.get('source_path') or 'unknown source'}`). This "
            "is a source comparison, not a controlled one-change A/B"
        )
        if features:
            note += f"; it also supports {', '.join(features)}"
        note += "."
        return note
    return None


def _bench_evidence(from_variant, to_variant):
    from_bench = from_variant.get("bench") or {}
    to_bench = to_variant.get("bench") or {}
    from_ms = _num(from_bench.get("kernel_ms"))
    to_ms = _num(to_bench.get("kernel_ms"))
    from_tflops = _num(from_bench.get("kernel_tflops"))
    to_tflops = _num(to_bench.get("kernel_tflops"))
    from_ref = _num(from_bench.get("ref_ms"))
    to_ref = _num(to_bench.get("ref_ms"))

    measured = (
        from_ms is not None
        and to_ms is not None
        and from_ms > 0
        and to_ms > 0
    )
    data = {
        "measured_both": measured,
        "from_ms": from_ms,
        "to_ms": to_ms,
        "from_tflops": from_tflops,
        "to_tflops": to_tflops,
        "from_ref_ms": from_ref,
        "to_ref_ms": to_ref,
    }
    caveats = []
    if measured:
        ratio = from_ms / to_ms
        delta = (to_ms - from_ms) / from_ms * 100.0
        data.update(
            {
                "ratio_from_over_to": ratio,
                "delta_pct": delta,
                "faster_side": "to" if to_ms < from_ms else (
                    "from" if from_ms < to_ms else "tie"
                ),
                "material": abs(delta) >= 5.0,
            }
        )
    if from_ref and to_ref:
        difference = abs(from_ref - to_ref) / max(from_ref, to_ref)
        if difference > 0.10:
            caveats.append(
                "The PyTorch reference times differ between the two "
                f"records ({_fmt(from_ref)} ms vs {_fmt(to_ref)} ms); the "
                "kernel ratio is fixed-harness evidence, not a same-session "
                "controlled A/B."
            )
    return data, caveats


def _ncu_evidence(from_variant, to_variant):
    from_side = _ncu_side(from_variant)
    to_side = _ncu_side(to_variant)
    from_duration = from_side.get("duration_avg_us")
    to_duration = to_side.get("duration_avg_us")
    measured = (
        from_side.get("status") == "ok"
        and to_side.get("status") == "ok"
        and from_duration is not None
        and to_duration is not None
    )
    data = {
        "measured_both": measured,
        "from": from_side,
        "to": to_side,
    }
    if measured:
        ratio = (
            from_duration / to_duration
            if to_duration
            else None
        )
        delta = (
            (to_duration - from_duration) / from_duration * 100.0
            if from_duration
            else None
        )
        data.update(
            {
                "ratio_from_over_to": ratio,
                "delta_pct": delta,
                "faster_side": (
                    "to"
                    if to_duration < from_duration
                    else "from"
                    if from_duration < to_duration
                    else "tie"
                ),
                "material": delta is not None and abs(delta) >= 5.0,
                "kernels_compared": {
                    "from": from_side.get("kernels_count", 1),
                    "to": to_side.get("kernels_count", 1),
                },
                "kernel_names": {
                    "from": from_side.get("kernel"),
                    "to": to_side.get("kernel"),
                },
            }
        )
    return data


def _evidence(from_variant, to_variant, from_code, to_code):
    bench, caveats = _bench_evidence(from_variant, to_variant)
    ncu = _ncu_evidence(from_variant, to_variant)

    for side_name, side in (("from", ncu["from"]), ("to", ncu["to"])):
        side_variant = (
            from_variant if side_name == "from" else to_variant
        )
        if side.get("status") != "ok":
            caveats.append(
                f"`{side_variant['variant']}` has no usable NCU capture "
                f"({side.get('note', 'not captured')}); occupancy/register "
                "effects are inferred, not measured on both sides."
            )
        elif (side.get("kernels_count") or 0) > 1:
            caveats.append(
                f"`{side_variant['variant']}` has "
                f"{side['kernels_count']} kernels in the capture; the "
                f"table compares the dominant one (`{side.get('kernel')}`)."
            )

    methodology_conflict = None
    if (
        bench.get("measured_both")
        and bench.get("material")
        and ncu.get("measured_both")
        and ncu.get("material")
        and bench.get("faster_side") != ncu.get("faster_side")
    ):
        methodology_conflict = {
            "bench_faster_side": bench.get("faster_side"),
            "ncu_faster_side": ncu.get("faster_side"),
        }
        caveats.append(
            "The fixed harness and NCU disagree on direction. NCU locks "
            "clocks at or below boost, so its duration should be greater "
            "than or equal to the fixed-harness time; this pattern is most "
            "plausibly a different autotune configuration or a clock/power "
            "difference. Use NCU for code-level attribution and do not claim "
            "a fixed-harness win for this pair."
        )

    below = []
    for side_name, side, variant in (
        ("from", ncu["from"], from_variant),
        ("to", ncu["to"], to_variant),
    ):
        duration = side.get("duration_avg_us")
        kernel_ms = (
            bench.get("from_ms") if side_name == "from" else bench.get("to_ms")
        )
        if (
            duration is not None
            and kernel_ms is not None
            and duration < kernel_ms * 1000.0 * 0.95
        ):
            below.append(variant["variant"])
    if below:
        caveats.append(
            "NCU duration is below the fixed-harness time for "
            + ", ".join(f"`{name}`" for name in below)
            + "; the benchmark and profiler likely selected different "
            "autotune configurations or ran at different clocks."
        )

    external = _external_note(
        from_variant, to_variant, from_code, to_code
    )
    if external:
        caveats.append(external)

    return {
        "bench": bench,
        "ncu": ncu,
        "methodology_conflict": methodology_conflict,
        "ncu_below_fixed_harness": below,
        "caveats": caveats,
    }


# ---------------------------------------------------------------------------
# Invariants


def _autotune_key(code):
    match = re.search(r"\bkey\s*=\s*\[([^\]]*)\]", code)
    if not match:
        return None
    return re.sub(r"\s+", "", match.group(1))


def _kernel_signature(code):
    match = re.search(
        r"\bdef\s+kernel_function\s*\((.*?)\)", code, re.DOTALL
    )
    if not match:
        return None
    return re.sub(r"\s+", "", match.group(1))


def _invariants(from_source, to_source):
    results = []
    for probe in _INVARIANT_PROBES:
        compare = probe.get("compare")
        if compare == "autotune_key":
            if (
                _autotune_key(from_source["code"])
                and _autotune_key(from_source["code"])
                == _autotune_key(to_source["code"])
            ):
                results.append(
                    {
                        "key": probe["key"],
                        "title": probe["title"],
                        "detail": probe["detail"],
                        "from_line": _first_line(
                            from_source["lines"],
                            from_source["flags"],
                            re.compile(r"\bkey\s*=\s*\["),
                        ),
                        "to_line": _first_line(
                            to_source["lines"],
                            to_source["flags"],
                            re.compile(r"\bkey\s*=\s*\["),
                        ),
                    }
                )
        elif compare == "signature":
            if (
                _kernel_signature(from_source["code"])
                and _kernel_signature(from_source["code"])
                == _kernel_signature(to_source["code"])
            ):
                results.append(
                    {
                        "key": probe["key"],
                        "title": probe["title"],
                        "detail": probe["detail"],
                        "from_line": _first_line(
                            from_source["lines"],
                            from_source["flags"],
                            re.compile(r"\bdef\s+kernel_function\b"),
                        ),
                        "to_line": _first_line(
                            to_source["lines"],
                            to_source["flags"],
                            re.compile(r"\bdef\s+kernel_function\b"),
                        ),
                    }
                )
        else:
            patterns = probe["patterns"]
            if _has_all(from_source["code"], patterns) and _has_all(
                to_source["code"], patterns
            ):
                first = patterns[0]
                results.append(
                    {
                        "key": probe["key"],
                        "title": probe["title"],
                        "detail": probe["detail"],
                        "from_line": _first_line(
                            from_source["lines"],
                            from_source["flags"],
                            first,
                        ),
                        "to_line": _first_line(
                            to_source["lines"],
                            to_source["flags"],
                            first,
                        ),
                    }
                )
        if len(results) >= MAX_INVARIANTS:
            break
    return results


# ---------------------------------------------------------------------------
# Pair analysis


def _relative_link(exp_dir: Path, rel_path: str | None):
    if not rel_path:
        return None
    try:
        return os.path.relpath(
            exp_dir / rel_path, exp_dir / "reports"
        ).replace(os.sep, "/")
    except ValueError:
        return None


def _siblings(variants):
    groups = {}
    for variant in variants:
        sha = variant.get("source_kernel_sha256")
        if not sha and variant.get("variant") == "generated":
            sha = variant.get("kernel_sha256")
        if sha:
            groups.setdefault(sha, []).append(variant["variant"])
    siblings = []
    for sha, names in groups.items():
        internal = [name for name in names if name != "generated"]
        if len(internal) > 1:
            siblings.append(
                {
                    "source_kernel_sha256": sha,
                    "variants": internal,
                }
            )
    return siblings


def _sibling_note(siblings):
    if not siblings:
        return None
    parts = []
    for group in siblings:
        names = " and ".join(f"`{name}`" for name in group["variants"])
        parts.append(
            f"{names} optimize the same source "
            f"({_sha(group['source_kernel_sha256'])}); they are independent "
            "runs, not a chain."
        )
    return " ".join(parts)


def _source_record(exp_dir, variant):
    path = variant.get("kernel_path")
    return {
        "path": path,
        "sha256": variant.get("kernel_sha256"),
        "link": _relative_link(exp_dir, path),
    }


def _takeaway(changes, evidence):
    parts = []
    if changes:
        categories = []
        for change in changes:
            if change["category"] not in categories:
                categories.append(change["category"])
        parts.append("Detected " + ", ".join(categories))
    bench = evidence.get("bench") or {}
    if bench.get("measured_both"):
        parts.append(
            "fixed harness "
            f"{_fmt(bench.get('from_ms'))} → {_fmt(bench.get('to_ms'))} ms"
        )
    ncu = evidence.get("ncu") or {}
    if ncu.get("measured_both"):
        parts.append(
            "NCU "
            f"{_fmt((ncu.get('from') or {}).get('duration_avg_us'))} → "
            f"{_fmt((ncu.get('to') or {}).get('duration_avg_us'))} µs"
        )
    else:
        parts.append("NCU not measured on both sides")
    return "; ".join(parts) + "."


def _analyze_pair(
    exp_dir,
    problem_name,
    record,
    variants,
    sources,
    case_context,
    seen,
):
    from_key, to_key = record["from"], record["to"]
    from_variant = variants[from_key]
    to_variant = variants[to_key]
    from_source = sources[from_key]
    to_source = sources[to_key]
    pair_label = f"`{from_key}` → `{to_key}`"

    regions = _regions(
        from_source["lines"],
        to_source["lines"],
        from_source["flags"],
        to_source["flags"],
    )
    pair_tiles = _extract_tiles(from_source["code"], to_source["code"])
    pair_context = dict(case_context)
    pair_context["tiles"] = pair_tiles
    pair_context["divisibility"] = _divisibility(
        case_context.get("integer_fields") or {},
        pair_tiles,
        from_source["code"],
        to_source["code"],
    )
    changes = _select_changes(regions, pair_context, seen, pair_label)
    evidence = _evidence(
        from_variant,
        to_variant,
        from_source["code"],
        to_source["code"],
    )

    return {
        "problem": problem_name,
        "from": from_key,
        "to": to_key,
        "from_role": from_variant.get("kind") or "internal",
        "to_role": to_variant.get("kind") or "internal",
        "diff_path": record.get("path"),
        "diff_link": _relative_link(exp_dir, record.get("path")),
        "source": {
            "from": _source_record(exp_dir, from_variant),
            "to": _source_record(exp_dir, to_variant),
        },
        "changed_lines": {
            "removed": sum(len(region["removed"]) for region in regions),
            "added": sum(len(region["added"]) for region in regions),
        },
        "external": {
            "from": from_variant.get("kind") == "external",
            "to": to_variant.get("kind") == "external",
            "source_path": (
                from_variant.get("source_path")
                if from_variant.get("kind") == "external"
                else to_variant.get("source_path")
                if to_variant.get("kind") == "external"
                else None
            ),
        },
        "evidence": evidence,
        "changes": changes,
        "invariants": _invariants(from_source, to_source),
        "caveats": evidence["caveats"],
        "takeaway": _takeaway(changes, evidence),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Document assembly


def _fingerprint(data, pairs):
    payload = {
        "experiment_id": data.get("experiment_id"),
        "capture_batch": data.get("capture_batch"),
        "capture_merge": data.get("capture_merge"),
        "sources": sorted(
            [
                pair["problem"],
                pair["from"],
                pair["to"],
                (pair.get("source") or {}).get("from", {}).get("sha256"),
                (pair.get("source") or {}).get("to", {}).get("sha256"),
            ]
            for pair in pairs
        ),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_analysis(exp_dir, data, diff_records):
    exp_dir = Path(exp_dir)
    records_by_problem = {}
    for record in diff_records:
        records_by_problem.setdefault(record["problem"], []).append(record)

    problems_out = []
    all_pairs = []
    skipped = []
    seen_by_from = {}
    variants_with_source = 0
    pairs_expected = 0

    for problem in data.get("problems") or []:
        name = problem.get("problem")
        variants = {
            variant["variant"]: variant
            for variant in problem.get("variants") or []
        }
        sources = {}
        for key, variant in variants.items():
            path = variant.get("kernel_path")
            if not path:
                skipped.append(
                    {
                        "problem": name,
                        "variant": key,
                        "status": variant.get("status"),
                        "reason": "no kernel_path",
                    }
                )
                continue
            full_path = exp_dir / path
            if not full_path.is_file():
                skipped.append(
                    {
                        "problem": name,
                        "variant": key,
                        "status": variant.get("status"),
                        "reason": "source file missing",
                    }
                )
                continue
            try:
                sources[key] = _read_source(full_path)
                variants_with_source += 1
            except OSError as exc:
                skipped.append(
                    {
                        "problem": name,
                        "variant": key,
                        "status": variant.get("status"),
                        "reason": f"source unreadable: {exc}",
                    }
                )

        case_context = _case_context(
            problem, *(source["code"] for source in sources.values())
        )
        pairs_expected += len(sources) * (len(sources) - 1) // 2
        siblings = _siblings(list(variants.values()))
        pairs_out = []
        for record in records_by_problem.get(name, []):
            from_key, to_key = record["from"], record["to"]
            if from_key not in sources or to_key not in sources:
                pairs_out.append(
                    {
                        "problem": name,
                        "from": from_key,
                        "to": to_key,
                        "diff_path": record.get("path"),
                        "diff_link": _relative_link(
                            exp_dir, record.get("path")
                        ),
                        "source": {
                            "from": _source_record(
                                exp_dir, variants.get(from_key) or {}
                            ),
                            "to": _source_record(
                                exp_dir, variants.get(to_key) or {}
                            ),
                        },
                        "changes": [],
                        "invariants": [],
                        "caveats": [],
                        "takeaway": "Source unavailable; analysis skipped.",
                        "error": "source file missing",
                    }
                )
                continue
            try:
                pair = _analyze_pair(
                    exp_dir,
                    name,
                    record,
                    variants,
                    sources,
                    case_context,
                    seen_by_from.setdefault(from_key, []),
                )
            except Exception as exc:  # noqa: BLE001 - one pair must not
                # discard the rest of the analysis document
                pair = {
                    "problem": name,
                    "from": from_key,
                    "to": to_key,
                    "diff_path": record.get("path"),
                    "diff_link": _relative_link(
                        exp_dir, record.get("path")
                    ),
                    "source": {},
                    "changes": [],
                    "invariants": [],
                    "caveats": [],
                    "takeaway": "Analysis failed for this pair.",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            pairs_out.append(pair)
            all_pairs.append(pair)

        problems_out.append(
            {
                "problem": name,
                "case": problem.get("case"),
                "case_spec": problem.get("case_spec"),
                "case_context": case_context,
                "siblings": siblings,
                "pairs": pairs_out,
            }
        )

    categories = {}
    conflicts = 0
    changes_detected = 0
    for pair in all_pairs:
        for change in pair.get("changes") or []:
            changes_detected += 1
            categories[change["category"]] = (
                categories.get(change["category"], 0) + 1
            )
        if (pair.get("evidence") or {}).get("methodology_conflict"):
            conflicts += 1

    variants_total = sum(
        len(problem.get("variants") or [])
        for problem in data.get("problems") or []
    )

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "experiment_id": data.get("experiment_id"),
        "generated_at": datetime.now().isoformat(),
        "capture_selection": {
            "capture_batch": data.get("capture_batch"),
            "capture_merge": data.get("capture_merge"),
        },
        "fingerprint": _fingerprint(data, all_pairs),
        "coverage": {
            "variants_total": variants_total,
            "variants_with_source": variants_with_source,
            "pairs_expected": pairs_expected,
            "pairs_analyzed": len(all_pairs),
            "skipped": skipped,
        },
        "problems": problems_out,
        "summary": {
            "pairs_analyzed": len(all_pairs),
            "changes_detected": changes_detected,
            "changes_by_category": categories,
            "methodology_conflicts": conflicts,
        },
        "paths": {
            "markdown": REPORT_MARKDOWN,
            "json": REPORT_JSON,
        },
        "error": None,
    }


# ---------------------------------------------------------------------------
# Markdown rendering


def _source_cell(source):
    if not source or not source.get("path"):
        return "—"
    label = Path(source["path"]).name
    link = source.get("link")
    sha = source.get("sha256")
    text = (
        f"[{label}]({link})"
        if link
        else f"`{source['path']}`"
    )
    if sha:
        text += f" sha {_sha(sha)}"
    return text


def _delta_cell(left, right, digits=3):
    left_number = _num(left)
    right_number = _num(right)
    if left_number is None or right_number is None:
        return "—"
    return f"{right_number - left_number:+.{digits}f}"


def _evidence_table(evidence):
    bench = evidence.get("bench") or {}
    ncu = evidence.get("ncu") or {}
    from_ncu = ncu.get("from") or {}
    to_ncu = ncu.get("to") or {}

    rows = [
        (
            "Fixed-harness kernel (ms)",
            bench.get("from_ms"),
            bench.get("to_ms"),
            _delta_cell(bench.get("from_ms"), bench.get("to_ms")),
        ),
        (
            "Fixed-harness TFLOPS",
            bench.get("from_tflops"),
            bench.get("to_tflops"),
            _delta_cell(bench.get("from_tflops"), bench.get("to_tflops")),
        ),
        (
            "PyTorch reference (ms)",
            bench.get("from_ref_ms"),
            bench.get("to_ref_ms"),
            _delta_cell(bench.get("from_ref_ms"), bench.get("to_ref_ms")),
        ),
        (
            "NCU duration (µs)",
            from_ncu.get("duration_avg_us"),
            to_ncu.get("duration_avg_us"),
            _delta_cell(
                from_ncu.get("duration_avg_us"),
                to_ncu.get("duration_avg_us"),
            ),
        ),
        (
            "NCU SM %",
            from_ncu.get("sm_pct"),
            to_ncu.get("sm_pct"),
            _delta_cell(from_ncu.get("sm_pct"), to_ncu.get("sm_pct"), 1),
        ),
        (
            "NCU memory %",
            from_ncu.get("mem_pct"),
            to_ncu.get("mem_pct"),
            _delta_cell(from_ncu.get("mem_pct"), to_ncu.get("mem_pct"), 1),
        ),
        (
            "NCU DRAM %",
            from_ncu.get("dram_pct"),
            to_ncu.get("dram_pct"),
            _delta_cell(
                from_ncu.get("dram_pct"), to_ncu.get("dram_pct"), 1
            ),
        ),
        (
            "NCU occupancy %",
            from_ncu.get("occupancy_pct"),
            to_ncu.get("occupancy_pct"),
            _delta_cell(
                from_ncu.get("occupancy_pct"),
                to_ncu.get("occupancy_pct"),
                1,
            ),
        ),
        (
            "Registers/thread",
            from_ncu.get("regs"),
            to_ncu.get("regs"),
            _delta_cell(from_ncu.get("regs"), to_ncu.get("regs"), 0),
        ),
        (
            "NCU TFLOPS",
            from_ncu.get("tflops"),
            to_ncu.get("tflops"),
            _delta_cell(from_ncu.get("tflops"), to_ncu.get("tflops")),
        ),
        (
            "NCU grid / block",
            (
                f"{_fmt(from_ncu.get('grid'), 0)} / "
                f"{_fmt(from_ncu.get('block'), 0)}"
                if from_ncu.get("status") == "ok"
                else "—"
            ),
            (
                f"{_fmt(to_ncu.get('grid'), 0)} / "
                f"{_fmt(to_ncu.get('block'), 0)}"
                if to_ncu.get("status") == "ok"
                else "—"
            ),
            "—",
        ),
    ]

    lines = [
        "#### Evidence",
        "",
        "| Metric | from | to | Δ |",
        "|---|---:|---:|---:|",
    ]
    for label, left, right, delta in rows:
        if isinstance(left, str):
            left_cell = left
        else:
            left_cell = _fmt(left)
        if isinstance(right, str):
            right_cell = right
        else:
            right_cell = _fmt(right)
        lines.append(f"| {label} | {left_cell} | {right_cell} | {delta} |")
    lines.append("")
    for side_name, side in (("from", ncu.get("from")), ("to", ncu.get("to"))):
        if side and side.get("status") != "ok":
            lines.append(
                f"- {side_name}: {side.get('note', 'no capture')}"
            )
    if any(
        side and side.get("status") != "ok"
        for side in (ncu.get("from"), ncu.get("to"))
    ):
        lines.append("")
    return lines


def _render_llm(pair, llm):
    if not llm:
        return []
    lookup = llm.get("_lookup") or {}
    entry = lookup.get(
        (pair.get("problem"), pair.get("from"), pair.get("to"))
    )
    if not entry:
        return []
    lines = ["#### LLM commentary", ""]
    if entry.get("status") != "ok":
        lines.append(
            f"_LLM enrichment {entry.get('status')}: "
            f"{entry.get('error') or 'no response'}_"
        )
        lines.append("")
        return lines
    response = entry.get("response") or {}
    if response.get("pair_summary"):
        lines.append(response["pair_summary"])
        lines.append("")
    for change in response.get("changes") or []:
        title = change.get("title") or change.get("category") or "Change"
        why = change.get("why_it_matters") or ""
        lines.append(f"**{title}.** {why}")
        if change.get("evidence"):
            lines.append(f"Evidence: {change['evidence']}")
        if change.get("fair_note"):
            lines.append(f"Fair note: {change['fair_note']}")
        lines.append("")
    if response.get("invariants"):
        lines.append(
            "What did not change: "
            + "; ".join(response["invariants"])
        )
        lines.append("")
    if response.get("caveats"):
        lines.extend(f"- {caveat}" for caveat in response["caveats"])
        lines.append("")
    if response.get("takeaway"):
        lines.append(f"**LLM takeaway:** {response['takeaway']}")
        lines.append("")
    return lines


def _render_pair(pair, llm):
    lines = [
        f"### Pair: `{pair.get('from')}` → `{pair.get('to')}`",
        "",
    ]
    source = pair.get("source") or {}
    lines.append(
        f"Sources: `{pair.get('from')}` "
        f"{_source_cell(source.get('from'))} · `{pair.get('to')}` "
        f"{_source_cell(source.get('to'))} · full diff: "
        f"[{Path(pair.get('diff_path') or 'diff').name}]"
        f"({pair.get('diff_link') or '—'})"
    )
    lines.append("")

    if pair.get("error"):
        lines.append(f"**Analysis error:** {pair['error']}")
        lines.append("")
        return lines

    lines.extend(_evidence_table(pair.get("evidence") or {}))

    changes = pair.get("changes") or []
    if not changes:
        lines.append(
            "_No recognized Triton-level changes; see the full diff._"
        )
        lines.append("")
    for index, change in enumerate(changes, 1):
        lines.append(f"#### {index}. {change.get('title')}")
        ranges = change.get("line_ranges") or {}
        if ranges.get("from_start") is not None:
            from_start = ranges.get("from_start")
            from_end = ranges.get("from_end")
            to_start = ranges.get("to_start")
            to_end = ranges.get("to_end")
            from_text = (
                _line_span("from", from_start, from_end)
                if from_end is not None and from_end > from_start
                else "from —"
            )
            to_text = (
                _line_span("to", to_start, to_end)
                if to_end is not None and to_end > to_start
                else "to —"
            )
            lines.append(f"Source lines: {from_text} → {to_text}")
            lines.append("")
        excerpt = change.get("excerpt")
        if excerpt:
            fence = _fence_for(excerpt)
            lines.extend([f"{fence}diff", excerpt, fence, ""])
        if change.get("repeated_from"):
            lines.append(
                "**Repeated:** This change also appears in "
                f"{change['repeated_from']}; excerpt omitted."
            )
            lines.append("")
        lines.append(
            f"**Why it matters:** {change.get('why_it_matters', '—')}"
        )
        lines.append("")

    invariants = pair.get("invariants") or []
    if invariants:
        lines.extend(["#### What did *not* change", ""])
        for invariant in invariants:
            lines.append(
                f"- **{invariant['title']}.** {invariant['detail']}"
            )
        lines.append("")

    caveats = pair.get("caveats") or []
    if caveats:
        lines.extend(["#### Caveats", ""])
        lines.extend(f"- {caveat}" for caveat in caveats)
        lines.append("")

    lines.append(f"**Takeaway:** {pair.get('takeaway', '—')}")
    lines.append("")
    lines.extend(_render_llm(pair, llm))
    return lines


def render_markdown(doc, llm=None):
    lines = [
        f"# Code diff analysis: `{doc.get('experiment_id')}`",
        "",
        f"Generated {doc.get('generated_at')} by the offline report.",
        "",
        "## How to read this",
        "",
        "- `-`/`+` in each heading name the `from`/`to` variants; they do not",
        "  mean worse/better.",
        "- Excerpts omit unchanged or repeated lines; code text is "
        "verbatim.",
        "- A **Why it matters** paragraph explains a detected mechanism. "
        "It is not proof that the mechanism caused a measured difference.",
        "- Benchmark and NCU numbers come from different methodologies and "
        "are reported separately. \"inferred, not measured on both sides\" "
        "marks a claim where one side lacks a usable capture.",
        "",
    ]

    if doc.get("error"):
        lines.extend(
            [
                f"**Analysis error:** {doc['error']}",
                "",
            ]
        )

    if llm:
        provenance = llm.get("provenance") or {}
        lines.extend(
            [
                "## LLM commentary provenance",
                "",
                f"- Model: `{provenance.get('model') or '—'}`",
                f"- Backend/provider: {provenance.get('backend') or '—'} / "
                f"{provenance.get('provider') or '—'}",
                f"- Reasoning effort: "
                f"{provenance.get('reasoning_effort') or 'model default'}",
                f"- Prompt version: "
                f"`{provenance.get('prompt_version') or '—'}`",
                f"- Deterministic fingerprint: "
                f"`{provenance.get('deterministic_fingerprint') or '—'}`",
                "",
                "The deterministic sections remain authoritative; the LLM",
                "commentary is an optional reading aid.",
                "",
            ]
        )

    coverage = doc.get("coverage") or {}
    lines.extend(
        [
            "## Coverage",
            "",
            "| Problem | Pairs analyzed | Skipped variants |",
            "|---|---:|---:|",
        ]
    )
    for problem in doc.get("problems") or []:
        skipped = sum(
            1
            for item in coverage.get("skipped") or []
            if item.get("problem") == problem.get("problem")
        )
        lines.append(
            f"| `{problem.get('problem')}` | "
            f"{len(problem.get('pairs') or [])} | {skipped} |"
        )
    lines.extend(
        [
            "",
            f"Pairs expected: {coverage.get('pairs_expected', 0)}; "
            f"analyzed: {coverage.get('pairs_analyzed', 0)}.",
            "",
        ]
    )
    if coverage.get("skipped"):
        lines.append("Skipped:")
        for item in coverage["skipped"]:
            lines.append(
                f"- `{item.get('problem')}`/`{item.get('variant')}`: "
                f"{item.get('reason')} (status `{item.get('status')}`)"
            )
        lines.append("")

    summary = doc.get("summary") or {}
    lines.extend(["## Summary", ""])
    lines.append(
        f"- Pairs analyzed: {summary.get('pairs_analyzed', 0)}"
    )
    lines.append(
        f"- Change groups detected: {summary.get('changes_detected', 0)}"
    )
    if summary.get("changes_by_category"):
        categories = ", ".join(
            f"{key} ({value})"
            for key, value in sorted(summary["changes_by_category"].items())
        )
        lines.append(f"- Categories: {categories}")
    lines.append(
        f"- Methodology conflicts: {summary.get('methodology_conflicts', 0)}"
    )
    lines.append("")

    for problem in doc.get("problems") or []:
        lines.extend(
            [
                f"## `{problem.get('problem')}` — benchmark case "
                f"`{problem.get('case')}`",
                "",
                f"Case: `{problem.get('case_spec')}`",
                "",
            ]
        )
        context = problem.get("case_context") or {}
        if context.get("fields"):
            fields = ", ".join(
                f"`{key}={value}`" for key, value in context["fields"].items()
            )
            lines.extend([f"Case context: {fields}", ""])
        if context.get("divisibility"):
            parts = []
            for item in context["divisibility"]:
                division = (
                    "all divide"
                    if item["all_divide"]
                    else "not all divide"
                )
                parts.append(
                    f"`{item['symbol']} % {item['tile_symbol']} == 0` "
                    f"({item['dim']} / {item['tile_values']}: {division})"
                )
            lines.extend(["Divisibility: " + "; ".join(parts), ""])
        sibling_note = _sibling_note(problem.get("siblings") or [])
        if sibling_note:
            lines.extend([f"Sibling note: {sibling_note}", ""])
        if not problem.get("pairs"):
            lines.extend(["_No pairwise diffs with available sources._", ""])
        for pair in problem.get("pairs") or []:
            lines.extend(_render_pair(pair, llm))

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Writing and pointers


def pointer(doc, written: bool):
    summary = doc.get("summary") or {}
    return {
        "markdown": REPORT_MARKDOWN,
        "json": REPORT_JSON,
        "pairs_analyzed": summary.get("pairs_analyzed", 0),
        "changes_detected": summary.get("changes_detected", 0),
        "changes_by_category": summary.get("changes_by_category", {}),
        "methodology_conflicts": summary.get("methodology_conflicts", 0),
        "fingerprint": doc.get("fingerprint"),
        "generated_at": doc.get("generated_at"),
        "written": written,
        "error": doc.get("error"),
    }


def write(exp_dir, doc):
    reports_dir = Path(exp_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(reports_dir / "code-diff-analysis.json", doc)
    _atomic_write_text(
        reports_dir / "code-diff-analysis.md", render_markdown(doc)
    )
    return pointer(doc, written=True)


def error_document(exp_dir, experiment_id, exc):
    doc = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at": datetime.now().isoformat(),
        "capture_selection": {},
        "fingerprint": None,
        "coverage": {
            "variants_total": 0,
            "pairs_expected": 0,
            "pairs_analyzed": 0,
            "skipped": [],
        },
        "problems": [],
        "summary": {
            "pairs_analyzed": 0,
            "changes_detected": 0,
            "changes_by_category": {},
            "methodology_conflicts": 0,
        },
        "paths": {"markdown": REPORT_MARKDOWN, "json": REPORT_JSON},
        "error": f"{type(exc).__name__}: {exc}",
    }
    write(exp_dir, doc)
    return doc
