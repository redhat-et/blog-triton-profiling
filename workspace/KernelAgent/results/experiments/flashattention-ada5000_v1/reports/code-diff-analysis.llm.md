# Code diff analysis: `flashattention-ada5000_v1`

Generated 2026-10-28T22:12:27.821397 by the offline report.

## How to read this

- `-`/`+` in each heading name the `from`/`to` variants; they do not
  mean worse/better.
- Excerpts omit unchanged or repeated lines; code text is verbatim.
- A **Why it matters** paragraph explains a detected mechanism. It is not proof that the mechanism caused a measured difference.
- Benchmark and NCU numbers come from different methodologies and are reported separately. "inferred, not measured on both sides" marks a claim where one side lacks a usable capture.

## LLM commentary provenance

- Model: `gpt-6.1-sol`
- Backend/provider: openai / openai
- Reasoning effort: model default
- Prompt version: `code-diff-llm-v1`
- Deterministic fingerprint: `sha256:f86629691354120bb4be77ed2cb9746dd83028ddee2591b7bdd3c8efcb5b3d9c`

The deterministic sections remain authoritative; the LLM
commentary is an optional reading aid.

## Coverage

| Problem | Pairs analyzed | Skipped variants |
|---|---:|---:|
| `flash_attention` | 6 | 0 |

Pairs expected: 6; analyzed: 6.

## Summary

- Pairs analyzed: 6
- Change groups detected: 36
- Categories: autotune_space (5), causal_specialization (4), constexpr_addressing (3), grid_addressing (6), host_launch (2), mask_elision (8), online_softmax (3), pipelining (5)
- Methodology conflicts: 4

## `flash_attention` — benchmark case `0`

Case: `(2, 8, 1024, 64, False, 'float16')`

Case context: `B=2`, `H=8`, `S=1024`, `D=64`, `causal=False`, `dtype=float16`

Divisibility: `SEQ % BLOCK_M == 0` (1024 / [32, 64, 128]: all divide); `SEQ % BLOCK_N == 0` (1024 / [32, 64, 128]: all divide)

Sibling note: `rounds_5` and `rounds_10` optimize the same source (`d1c93f8eb483…`); they are independent runs, not a chain.

### Pair: `generated` → `rounds_5`

Sources: `generated` [final_kernel.py](../problems/flash_attention/generated/flash_attention/final_kernel.py) sha `d1c93f8eb483…` · `rounds_5` [optimized_kernel.py](../problems/flash_attention/case_0/rounds_5/flash_attention/optimized_kernel.py) sha `138f765d4df3…` · full diff: [flash_attention__generated_to_rounds_5.diff](source-diffs/flash_attention__generated_to_rounds_5.diff)

#### Evidence

| Metric | from | to | Δ |
|---|---:|---:|---:|
| Fixed-harness kernel (ms) | 0.134 | 0.437 | +0.304 |
| Fixed-harness TFLOPS | 32.100 | 9.820 | -22.280 |
| PyTorch reference (ms) | 0.068 | 0.330 | +0.262 |
| NCU duration (µs) | 185.454 | 72.624 | -112.830 |
| NCU SM % | 41.756 | 54.565 | +12.8 |
| NCU memory % | 51.616 | 20.845 | -30.8 |
| NCU DRAM % | 7.240 | 16.678 | +9.4 |
| NCU occupancy % | 16.162 | 12.984 | -3.2 |
| Registers/thread | 64.000 | 248.000 | +184 |
| NCU TFLOPS | 23.159 | 59.140 | +35.981 |
| NCU grid / block | 1024 / 128 | 128 / 128 | — |

#### 1. Mask-free fast path added
Source lines: from — → to 7-28

```diff
+def _attention_inner(
+    q, K, V, base, rows, dims,
+    acc, normalizer, maximum,
+    lo, hi,
+    SEQ: tl.constexpr,
+    HEAD_DIM: tl.constexpr,
+    BLOCK_M: tl.constexpr,
+    BLOCK_N: tl.constexpr,
+    MASK_CAUSAL: tl.constexpr,
+):
+    cols = tl.arange(0, BLOCK_N)
+    scale: tl.constexpr = 1.4426950408889634 * HEAD_DIM ** -0.5
+
+    k_ptrs = K + base + dims[:, None] + cols[None, :] * HEAD_DIM
+    v_ptrs = V + base + cols[:, None] * HEAD_DIM + dims[None, :]
+
+    for start in tl.range(lo, hi, BLOCK_N):
+        keys = start + cols
+
+        if SEQ % BLOCK_N == 0:
+            k = tl.load(k_ptrs + start * HEAD_DIM)
+            v = tl.load(v_ptrs + start * HEAD_DIM)
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 2. Mask-free fast path added
Source lines: from — → to 29-48

```diff
+        else:
+            k = tl.load(
+                k_ptrs + start * HEAD_DIM,
+                mask=keys[None, :] < SEQ,
+                other=0.0,
+            )
+            v = tl.load(
+                v_ptrs + start * HEAD_DIM,
+                mask=keys[:, None] < SEQ,
+                other=0.0,
+            )
+
+        scores = tl.dot(q, k)
+
+        if SEQ % BLOCK_N != 0:
+            scores = tl.where(keys[None, :] < SEQ, scores, -float("inf"))
+        if MASK_CAUSAL:
+            scores = tl.where(
+                rows[:, None] >= keys[None, :], scores, -float("inf")
+            )
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 3. Autotune candidate space expanded or changed
Source lines: from — → to 49-73

```diff
+
+        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1) * scale)
+        probabilities = tl.exp2(scores * scale - new_maximum[:, None])
+        correction = tl.exp2(maximum - new_maximum)
+        block_sum = tl.sum(probabilities, axis=1)
+
+        acc = acc * correction[:, None]
+        acc = tl.dot(probabilities.to(q.dtype), v, acc)
+        normalizer = normalizer * correction + block_sum
+        maximum = new_maximum
+
+    return acc, normalizer, maximum
... 1 changed lines omitted ...
+
+@triton.autotune(
+    configs=[
+        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
+        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
+        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
+        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
+        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
+        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
+    ],
+    key=["SEQ", "HEAD_DIM", "BATCH_HEADS", "CAUSAL"],
+)
```

**Why it matters:** The candidate tile/warp/stage space changes. A different candidate set can select a better register/shared-memory/occupancy balance. The source alone does not prove which configuration ran; NCU grid/block/occupancy is the evidence.

#### 4. Compile-time strides and index arithmetic — added in `to`
Source lines: from — → to 74-79

```diff
+@triton.jit
+def _flash_attention_kernel(
+    Q, K, V, O,
+    SEQ: tl.constexpr,
+    HEAD_DIM: tl.constexpr,
+    BATCH_HEADS: tl.constexpr,
```

**Why it matters:** Runtime dimensions/strides become `tl.constexpr` and/or modulo wrapping is removed. Compile-time constants let Triton fold address arithmetic, remove integer divide/modulo from every element's index path, and prove alignment.

#### 5. Grid and program addressing — added in `to`
Source lines: from — → to 133-147

```diff
+    batch, heads, seq, head_dim = Q.shape
+    output = torch.empty(Q.shape, device=Q.device, dtype=Q.dtype)
+    if seq == 0 or batch * heads == 0:
+        return output
+
+    grid = lambda meta: (
+        triton.cdiv(seq, meta["BLOCK_M"]),
+        batch * heads,
+    )
+    _flash_attention_kernel[grid](
+        Q, K, V, output,
+        SEQ=seq,
+        HEAD_DIM=head_dim,
+        BATCH_HEADS=batch * heads,
+        CAUSAL=bool(causal),
```

**Why it matters:** Program-ID mapping, launch geometry, or address construction changes. Flattening axes or replacing per-program metadata loads with direct offsets removes fixed integer/global-load work per CTA and changes how tiles map to L2.

#### 6. Host wrapper and launch contract — removed in `to`
Source lines: from 124-144 → to —

```diff
-    """Compute scaled dot-product attention with a single fused Triton kernel.
-
-    The kernel fuses QK^T, scaling, optional causal masking, numerically stable
-    online softmax, and the weighted V reduction. It does not materialize the
-    attention-score or probability matrices.
-    """
-    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
-        raise ValueError("Q, K, and V must have shape [batch, heads, seq, dim]")
-    if Q.shape != K.shape or Q.shape != V.shape:
-        raise ValueError("Q, K, and V must have identical shapes")
-    if Q.dtype != K.dtype or Q.dtype != V.dtype:
-        raise ValueError("Q, K, and V must have the same dtype")
-    if Q.device != K.device or Q.device != V.device:
-        raise ValueError("Q, K, and V must be on the same device")
-    if not Q.is_cuda:
-        raise ValueError("Q, K, and V must be CUDA tensors")
-
-    batch, heads, seq, head_dim = Q.shape
-    output = torch.empty(
-        (batch, heads, seq, head_dim), device=Q.device, dtype=Q.dtype
-    )
```

**Why it matters:** Host-side wrapper, launch geometry, or contract checks change. This can alter grid shape and argument passing without changing device code; assertions and validation are not performance-relevant.

#### What did *not* change

- **fp32 accumulation preserved.** Both accumulate in fp32.
- **Masked output store preserved.** Both mask the output store.
- **Kernel entry signature preserved.** Both expose the same `kernel_function` signature.

#### Caveats

- The PyTorch reference times differ between the two records (0.068 ms vs 0.330 ms); the kernel ratio is fixed-harness evidence, not a same-session controlled A/B.
- The fixed harness and NCU disagree on direction. NCU locks clocks at or below boost, so its duration should be greater than or equal to the fixed-harness time; this pattern is most plausibly a different autotune configuration or a clock/power difference. Use NCU for code-level attribution and do not claim a fixed-harness win for this pair.
- NCU duration is below the fixed-harness time for `rounds_5`; the benchmark and profiler likely selected different autotune configurations or ran at different clocks.

**Takeaway:** Detected mask_elision, autotune_space, constexpr_addressing, grid_addressing, host_launch; fixed harness 0.134 → 0.437 ms; NCU 185.454 → 72.624 µs.

#### LLM commentary

For noncausal fp16 attention with B=2, H=8, S=1024, D=64, rounds_5 replaces a fixed-tile, stride-aware kernel with autotuned contiguous addressing, divisibility-specialized accesses, and revised online-softmax arithmetic. Bench reports 0.1338 → 0.4374 ms, whereas NCU reports 185.454 → 72.624 µs: the measurements disagree on direction, so this pair does not establish a fixed-harness win. Both sides have usable NCU captures, but their resource differences do not isolate any individual change as a measured cause.

**Larger query tiles and configurable staging.** Larger query tiles can reuse K/V across more query rows while increasing per-program accumulator and register demands.
Evidence: generated fixes BLOCK_M=16, BLOCK_N=64 and four warps. rounds_5 offers six configurations with BLOCK_M=32/64/128, BLOCK_N=32/64 and two or three stages. NCU grid size falls from 1024 to 128 with block size 128 on both sides; the target grid is consistent with BLOCK_M=128. Registers rise from 64 to 248 per thread and occupancy falls from 16.162% to 12.984%.
Fair note: The captured grid does not identify BLOCK_N or staging, and the benchmark's selected configuration is not supplied.

**Divisibility-specialized loads, scores and stores.** Compile-time divisibility branches permit unconditional accesses and omit boundary score selection for this case.
Evidence: rounds_5 guards Q/output accesses with SEQ % BLOCK_M and K/V accesses plus score masking with SEQ % BLOCK_N. S=1024 divides every offered tile size. generated explicitly supplies load/store masks and score selections.
Fair note: Source-level mask removal does not establish an instruction-count reduction; compiler simplification of the original is not shown.

**Contiguous offsets replace runtime strides.** Direct contiguous offsets simplify address expressions and specialize sequence-dependent arithmetic.
Evidence: rounds_5 makes SEQ constexpr and uses base=head*SEQ*HEAD_DIM, replacing runtime Q/K/V stride arguments and batch/head decomposition. The grid still indexes query tiles and flattened batch-head pairs.
Fair note: This narrows the input-layout contract: the target assumes contiguous Q/K/V, whereas generated uses their supplied strides. Layout validation is not provided.

**Base-two softmax and explicit dot accumulation.** Equivalent softmax formulations and accumulator placement can change instruction selection, dependencies and rounding.
Evidence: rounds_5 uses exp2 with scale 1.4426950408889634*HEAD_DIM**-0.5, passes the rescaled accumulator into tl.dot, and normalizes through a reciprocal. generated uses tl.exp, adds a separate dot result, and divides by the row sum.
Fair note: No instruction-level comparison or numerical-error results are supplied; these mechanisms cannot be assigned an isolated timing effect.

**Validation removed; empty-input return added.** The wrapper changes accepted-input checks and launch behavior independently of device arithmetic.
Evidence: rounds_5 removes rank, shape, dtype, device and CUDA checks, adds an empty-input return, and uses an autotune-dependent grid.
Fair note: The empty-input branch is inactive for this case, and validation removal is not evidence of a device-kernel improvement.

What did not change: Both implement fused attention with online softmax and fp32 accumulation without materializing full attention matrices.; Both retain kernel_function(Q, K, V, causal).; This case is noncausal; the new causal loop split is inactive.

- Reference timings also differ: 0.0677 versus 0.3296 ms, so these records are not a same-session controlled A/B.
- Target NCU duration is below its benchmark duration. Different autotune selections or clock/power conditions are plausible explanations, not established facts.
- NCU reports memory-hierarchy limitation for generated and tensor-pipeline limitation for rounds_5; these characterize the captures, not isolated causal effects.
- The supplied masked-store invariant conflicts with the full diff: rounds_5 retains only a fallback masked store and uses an unconditional store here.

**LLM takeaway:** The target combines larger captured query tiles, contiguous specialization and revised softmax arithmetic with higher register use and lower occupancy. NCU favors it while bench favors generated; resolve that disagreement before asserting a reproducible performance advantage.

### Pair: `generated` → `rounds_10`

Sources: `generated` [final_kernel.py](../problems/flash_attention/generated/flash_attention/final_kernel.py) sha `d1c93f8eb483…` · `rounds_10` [optimized_kernel.py](../problems/flash_attention/case_0/rounds_10/flash_attention/optimized_kernel.py) sha `9b1431e00661…` · full diff: [flash_attention__generated_to_rounds_10.diff](source-diffs/flash_attention__generated_to_rounds_10.diff)

#### Evidence

| Metric | from | to | Δ |
|---|---:|---:|---:|
| Fixed-harness kernel (ms) | 0.134 | 0.307 | +0.173 |
| Fixed-harness TFLOPS | 32.100 | 13.980 | -18.120 |
| PyTorch reference (ms) | 0.068 | 0.354 | +0.286 |
| NCU duration (µs) | 185.454 | 68.597 | -116.857 |
| NCU SM % | 41.756 | 58.061 | +16.3 |
| NCU memory % | 51.616 | 22.142 | -29.5 |
| NCU DRAM % | 7.240 | 17.146 | +9.9 |
| NCU occupancy % | 16.162 | 13.308 | -2.9 |
| Registers/thread | 64.000 | 244.000 | +180 |
| NCU TFLOPS | 23.159 | 62.610 | +39.451 |
| NCU grid / block | 1024 / 128 | 128 / 128 | — |

#### 1. Mask-free fast path added
Source lines: from — → to 27-51

```diff
+        if SEQ % BLOCK_N == 0:
+            k = tl.load(k_ptrs + start * HEAD_DIM)
+            v = tl.load(v_ptrs + start * HEAD_DIM)
+        else:
+            k = tl.load(
+                k_ptrs + start * HEAD_DIM,
+                mask=keys[None, :] < SEQ,
+                other=0,
+            )
+            v = tl.load(
+                v_ptrs + start * HEAD_DIM,
+                mask=keys[:, None] < SEQ,
... 1 changed lines omitted ...
+            )
+
+        scores = tl.dot(q, k) * scale
+
+        if SEQ % BLOCK_N != 0:
+            scores = tl.where(keys[None, :] < SEQ, scores, -float("inf"))
+        if MASK_CAUSAL:
+            scores = tl.where(
+                keys[None, :] <= rows[:, None],
+                scores,
+                -float("inf"),
+            )
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 2. Mask-free fast path added
Source lines: from — → to 121-144

```diff
+    query_block = tl.program_id(0)
+    batch_head = tl.program_id(1)
+    base = batch_head * SEQ * HEAD_DIM
+
+    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
+    dims = tl.arange(0, HEAD_DIM)
+    q_ptrs = Q + base + rows[:, None] * HEAD_DIM + dims[None, :]
+
+    if SEQ % BLOCK_M == 0:
+        q = tl.load(q_ptrs)
+    else:
+        q = tl.load(q_ptrs, mask=rows[:, None] < SEQ, other=0)
+
+    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
+    l = tl.zeros((BLOCK_M,), tl.float32)
+    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
+
+    if CAUSAL:
+        prefix_end = (query_block * BLOCK_M // BLOCK_N) * BLOCK_N
+
+        m, l, acc = _attention_blocks(
+            q, K, V, base, rows, m, l, acc,
+            0, prefix_end,
+            SEQ, HEAD_DIM, BLOCK_M, BLOCK_N, False,
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 3. Autotune candidate space expanded or changed
Source lines: from — → to 52-101

```diff
+
+        next_m = tl.maximum(m, tl.max(scores, axis=1))
+        p = tl.exp2(scores - next_m[:, None])
+        alpha = tl.exp2(m - next_m)
+        block_l = tl.sum(p, axis=1)
+
+        acc = acc * alpha[:, None]
+        acc = tl.dot(p.to(q.dtype), v, acc)
+        l = l * alpha + block_l
+        m = next_m
+
+    return m, l, acc
... 26 changed lines omitted ...
+            num_warps=4,
+            num_stages=3,
+        ),
+        triton.Config(
+            {"BLOCK_M": 128, "BLOCK_N": 32},
+            num_warps=4,
+            num_stages=3,
+        ),
+        triton.Config(
+            {"BLOCK_M": 128, "BLOCK_N": 64},
+            num_warps=4,
+            num_stages=3,
```

**Why it matters:** The candidate tile/warp/stage space changes. A different candidate set can select a better register/shared-memory/occupancy balance. The source alone does not prove which configuration ran; NCU grid/block/occupancy is the evidence.

#### 4. Software pipelining — added in `to`
Source lines: from — → to 7-26

```diff
+def _attention_blocks(
+    q, K, V, base, rows, m, l, acc,
+    begin, end,
+    SEQ: tl.constexpr,
+    HEAD_DIM: tl.constexpr,
+    BLOCK_M: tl.constexpr,
+    BLOCK_N: tl.constexpr,
+    MASK_CAUSAL: tl.constexpr,
+):
+    cols = tl.arange(0, BLOCK_N)
+    dims = tl.arange(0, HEAD_DIM)
+    scale: tl.constexpr = 1.4426950408889634 * HEAD_DIM ** -0.5
+
+    k_ptrs = K + base + dims[:, None] + cols[None, :] * HEAD_DIM
+    v_ptrs = V + base + cols[:, None] * HEAD_DIM + dims[None, :]
+
+    for start in tl.range(begin, end, BLOCK_N):
+        start = tl.multiple_of(start, BLOCK_N)
+        keys = start + cols
+
```

**Why it matters:** Pipeline depth or loop form changes: more stages overlap global loads with tensor-core work at higher shared-memory/register cost; fewer stages reduce pressure but expose load latency.

#### 5. Compile-time strides and index arithmetic — removed in `to`
Source lines: from 9-28 → to —

```diff
-def _flash_attention_kernel(
-    q_ptr,
-    k_ptr,
-    v_ptr,
-    out_ptr,
-    S,
-    stride_qb,
-    stride_qh,
-    stride_qs,
-    stride_qd,
-    stride_kb,
-    stride_kh,
-    stride_ks,
-    stride_kd,
-    stride_vb,
-    stride_vh,
-    stride_vs,
-    stride_vd,
-    H: tl.constexpr,
-    HEAD_DIM: tl.constexpr,
```

**Why it matters:** Runtime dimensions/strides become `tl.constexpr` and/or modulo wrapping is removed. Compile-time constants let Triton fold address arithmetic, remove integer divide/modulo from every element's index path, and prove alignment.

#### 6. Grid and program addressing — added in `to`
Source lines: from 124-143 → to 170-175

```diff
 
 
 def kernel_function(Q, K, V, causal):
-    """Compute scaled dot-product attention with a single fused Triton kernel.
-
-    The kernel fuses QK^T, scaling, optional causal masking, numerically stable
-    online softmax, and the weighted V reduction. It does not materialize the
-    attention-score or probability matrices.
-    """
-    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
-        raise ValueError("Q, K, and V must have shape [batch, heads, seq, dim]")
-    if Q.shape != K.shape or Q.shape != V.shape:
-        raise ValueError("Q, K, and V must have identical shapes")
-    if Q.dtype != K.dtype or Q.dtype != V.dtype:
-        raise ValueError("Q, K, and V must have the same dtype")
... 2 changed lines omitted ...
-    if not Q.is_cuda:
-        raise ValueError("Q, K, and V must be CUDA tensors")
-
-    batch, heads, seq, head_dim = Q.shape
-    output = torch.empty(
-        (batch, heads, seq, head_dim), device=Q.device, dtype=Q.dtype
+    batch, heads, seq, head_dim = Q.shape
+    output = torch.empty(Q.shape, dtype=Q.dtype, device=Q.device)
+
+    grid = lambda meta: (
+        triton.cdiv(seq, meta["BLOCK_M"]),
+        batch * heads,
     )
 
```

**Why it matters:** Program-ID mapping, launch geometry, or address construction changes. Flattening axes or replacing per-program metadata loads with direct offsets removes fixed integer/global-load work per CTA and changes how tiles map to L2.

#### What did *not* change

- **fp32 accumulation preserved.** Both accumulate in fp32.
- **Masked output store preserved.** Both mask the output store.
- **Kernel entry signature preserved.** Both expose the same `kernel_function` signature.

#### Caveats

- The PyTorch reference times differ between the two records (0.068 ms vs 0.354 ms); the kernel ratio is fixed-harness evidence, not a same-session controlled A/B.
- The fixed harness and NCU disagree on direction. NCU locks clocks at or below boost, so its duration should be greater than or equal to the fixed-harness time; this pattern is most plausibly a different autotune configuration or a clock/power difference. Use NCU for code-level attribution and do not claim a fixed-harness win for this pair.
- NCU duration is below the fixed-harness time for `rounds_10`; the benchmark and profiler likely selected different autotune configurations or ran at different clocks.

**Takeaway:** Detected mask_elision, autotune_space, pipelining, constexpr_addressing, grid_addressing; fixed harness 0.134 → 0.307 ms; NCU 185.454 → 68.597 µs.

#### LLM commentary

For noncausal FP16 attention with shape (2, 8, 1024, 64), rounds_10 replaces a fixed-tile, stride-aware kernel with an autotuned implementation using contiguous addressing, divisibility-specialized accesses, and base-2 online softmax. Benchmark latency increases from 0.1338 to 0.3072 ms, while NCU kernel duration decreases from 185.454 to 68.597 µs. These measurements disagree on direction, so this pair does not establish a fixed-harness win; the mechanisms below are not isolated measured causes.

**Larger tiles and configurable launch resources.** Larger query tiles can reuse K/V across more query rows and reduce program count, while increasing accumulator storage and register pressure. Warp and stage choices change resource and latency-hiding tradeoffs.
Evidence: The original uses BLOCK_M=16, BLOCK_N=64 and four warps. The replacement has eight configurations spanning BLOCK_M=32–128, BLOCK_N=32–128, two to eight warps, and two or three stages. NCU reports grid size 1024→128, registers/thread 64→244, and occupancy 16.162%→13.308%.
Fair note: The replacement's NCU grid is consistent with BLOCK_M=128, but does not identify BLOCK_N. NCU describes a shift from memory-hierarchy pressure to tensor-pipeline pressure; it does not isolate a tile-size effect.

**Compile-time divisibility removes boundary predicates.** Unconditional loads and stores and omitted score selects reduce explicit boundary-handling work when sequence tiles divide the sequence length.
Evidence: SEQ is constexpr, and divisibility branches select unmasked Q/K/V loads and output stores. Every replacement candidate's BLOCK_M and BLOCK_N divides 1024; causal masking is disabled for this case. The original explicitly masks accesses and scores.
Fair note: The full diff contradicts the supplied invariant claiming masked stores are preserved: rounds_10 uses an unconditional store for this case. Compiler-level instruction savings are not measured separately.

**Runtime strides replaced by contiguous offsets.** Compile-time sequence dimensions and direct offsets permit simpler address arithmetic, but narrow the supported input-layout assumptions.
Evidence: The original passes separate Q/K/V strides and decodes batch/head from program_id(1). The replacement uses base=batch_head*SEQ*HEAD_DIM and row*HEAD_DIM+dim offsets, without stride arguments.
Fair note: Both retain a query-block by batch-head grid; the mapping is simplified rather than newly introduced. The provided case specifies shape and dtype, not strides, so general noncontiguous-input correctness is not established.

**Base-2 softmax and direct dot accumulation.** Scaling scores by log2(e) permits exp2-based online softmax. Passing the rescaled accumulator directly into tl.dot changes the accumulation expression and potential instruction scheduling.
Evidence: tl.exp becomes tl.exp2 with scale=1.4426950408889634*HEAD_DIM**-0.5. The replacement uses acc=acc*alpha followed by tl.dot(p.to(q.dtype), v, acc), instead of adding a separately expressed dot result.
Fair note: The mathematical normalization structure remains online softmax, but identical numerical results and an isolated performance contribution are not demonstrated.

**Helper-based loop with explicit stage configurations.** Loop structure, alignment hints, and stage settings offer scheduling opportunities with resource tradeoffs.
Evidence: The replacement introduces _attention_blocks, tl.range, tl.multiple_of(start, BLOCK_N), and explicit num_stages settings.
Fair note: The source does not prove achieved load/compute overlap. The causal prefix/diagonal split is inactive for this noncausal case.

What did not change: Both expose kernel_function(Q, K, V, causal).; Both use FP32 online-softmax state and output accumulation.; Both fuse attention computation without materializing full score or probability matrices.

- Both sides have usable NCU captures, but neither capture isolates individual source changes.
- Reference timings differ: 0.0677 versus 0.354 ms; these are not same-session controlled A/B records.
- The replacement's NCU duration is below its benchmark latency. Different autotune selections or clock/power conditions are plausible, not established explanations.
- Input validation is removed, and no correctness results are supplied.

**LLM takeaway:** NCU observes a shorter kernel duration alongside higher register use and tensor-pipeline utilization, whereas the benchmark reports longer latency. Treat this as conflicting performance evidence, not proof that any individual rewrite caused an improvement.

### Pair: `generated` → `vllm_triton_prefill`

Sources: `generated` [final_kernel.py](../problems/flash_attention/generated/flash_attention/final_kernel.py) sha `d1c93f8eb483…` · `vllm_triton_prefill` [kernel.py](../problems/flash_attention/baselines/vllm_triton_prefill/flash_attention/kernel.py) sha `2c38b86851dd…` · full diff: [flash_attention__generated_to_vllm_triton_prefill.diff](source-diffs/flash_attention__generated_to_vllm_triton_prefill.diff)

#### Evidence

| Metric | from | to | Δ |
|---|---:|---:|---:|
| Fixed-harness kernel (ms) | 0.134 | 0.082 | -0.052 |
| Fixed-harness TFLOPS | 32.100 | 52.510 | +20.410 |
| PyTorch reference (ms) | 0.068 | 0.064 | -0.004 |
| NCU duration (µs) | 185.454 | 80.740 | -104.714 |
| NCU SM % | 41.756 | 49.470 | +7.7 |
| NCU memory % | 51.616 | 36.277 | -15.3 |
| NCU DRAM % | 7.240 | 16.077 | +8.8 |
| NCU occupancy % | 16.162 | 13.894 | -2.3 |
| Registers/thread | 64.000 | 255.000 | +191 |
| NCU TFLOPS | 23.159 | 53.196 | +30.037 |
| NCU grid / block | 1024 / 128 | 128 / 128 | — |

#### 1. Mask-free fast path added
Source lines: from — → to 145-148

```diff
+
+    q = tl.load(
+        Q + off_q,
+        mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :]),
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary.

#### 2. Causal loop specialization — added in `to`
Source lines: from — → to 93-123

```diff
+def _fwd_kernel(
+    Q,
+    K,
+    V,
+    Sinks,
+    sm_scale,
+    B_Start_Loc,
+    B_Seqlen,
+    Out,
+    stride_qbs,
+    stride_qh,
+    stride_kbs,
... 7 changed lines omitted ...
+    BLOCK_DMODEL: tl.constexpr,
+    BLOCK_N: tl.constexpr,
+    IS_CAUSAL: tl.constexpr,
+    SLIDING_WINDOW_Q: tl.constexpr,
+    SLIDING_WINDOW_K: tl.constexpr,
+    SINKS_BIAS_KEY0: tl.constexpr,
+    USE_SINKS: tl.constexpr,
+    Lk: tl.constexpr,
+):
+    cur_batch = tl.program_id(0)
+    cur_head = tl.program_id(1)
+    start_m = tl.program_id(2)
```

**Why it matters:** The causal loop is split into an unmasked prefix and a masked diagonal, or fully masked blocks are pruned. Only causal cases are affected. This case is non-causal, so the change cannot affect its hot loop; it matters for causal cases.

#### 3. Software pipelining — added in `to`
Source lines: from — → to 314-343

```diff
+
+    _fwd_kernel[grid](
+        q,
+        k,
+        v,
+        sinks if sinks is not None else q,
+        sm_scale,
+        b_start_loc,
+        b_seq_len,
+        o,
+        q.stride(0),
+        q.stride(1),
... 6 changed lines omitted ...
+        kv_group_num=kv_group_num,
+        BLOCK_M=BLOCK,
+        BLOCK_DMODEL=triton.next_power_of_2(Lk),
+        BLOCK_N=BLOCK_N,
+        IS_CAUSAL=is_causal,
+        SLIDING_WINDOW_Q=sliding_window_q,
+        SLIDING_WINDOW_K=sliding_window_k,
+        USE_SINKS=sinks is not None,
+        SINKS_BIAS_KEY0=sinks_bias_key0,
+        num_warps=num_warps,
+        num_stages=1,
+        Lk=Lk,
```

**Why it matters:** Pipeline depth or loop form changes: more stages overlap global loads with tensor-core work at higher shared-memory/register cost; fewer stages reduce pressure but expose load latency.

#### 4. Grid and program addressing — added in `to`
Source lines: from — → to 124-144

```diff
+
+    cur_kv_head = cur_head // kv_group_num
+
+    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
+    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
+
+    block_start_loc = BLOCK_M * start_m
+
+    # initialize offsets
+    offs_n = tl.arange(0, BLOCK_N)
+    offs_d = tl.arange(0, BLOCK_DMODEL)
+    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
+    off_q = (
+        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
+        + cur_head * stride_qh
+        + offs_d[None, :]
+    )
+    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
+    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]
+
+    mask_d = offs_d < Lk
```

**Why it matters:** Program-ID mapping, launch geometry, or address construction changes. Flattening axes or replacing per-program metadata loads with direct offsets removes fixed integer/global-load work per CTA and changes how tiles map to L2.

#### 5. Compile-time strides and index arithmetic — removed in `to`
Source lines: from 9-32 → to —

```diff
-def _flash_attention_kernel(
-    q_ptr,
-    k_ptr,
-    v_ptr,
-    out_ptr,
-    S,
-    stride_qb,
-    stride_qh,
-    stride_qs,
-    stride_qd,
-    stride_kb,
-    stride_kh,
-    stride_ks,
-    stride_kd,
-    stride_vb,
-    stride_vh,
-    stride_vs,
-    stride_vd,
-    H: tl.constexpr,
-    HEAD_DIM: tl.constexpr,
-    CAUSAL: tl.constexpr,
-    BLOCK_M: tl.constexpr,
-    BLOCK_N: tl.constexpr,
-):
```

**Why it matters:** Runtime dimensions/strides become `tl.constexpr` and/or modulo wrapping is removed. Compile-time constants let Triton fold address arithmetic, remove integer divide/modulo from every element's index path, and prove alignment.

#### 6. Host wrapper and launch contract — removed in `to`
Source lines: from 123-143 → to —

**Repeated:** This change also appears in `generated` → `rounds_5`; excerpt omitted.

**Why it matters:** Host-side wrapper, launch geometry, or contract checks change. This can alter grid shape and argument passing without changing device code; assertions and validation are not performance-relevant.

#### What did *not* change

- **Online-softmax recurrence preserved.** Both use the online-softmax recurrence (running max/denominator, `exp2`, and a P·V dot); no full score matrix is materialized.
- **fp32 accumulation preserved.** Both accumulate in fp32.
- **Masked output store preserved.** Both mask the output store.
- **Kernel entry signature preserved.** Both expose the same `kernel_function` signature.

#### Caveats

- `vllm_triton_prefill` is an archived external baseline (`flash_attention/baselines/vllm_triton_prefill.py`). This is a source comparison, not a controlled one-change A/B; it also supports variable sequence lengths, GQA, attention sinks.

**Takeaway:** Detected mask_elision, causal_specialization, pipelining, grid_addressing, constexpr_addressing, host_launch; fixed harness 0.134 → 0.082 ms; NCU 185.454 → 80.740 µs.

#### LLM commentary

For non-causal fp16 attention with shape (2, 8, 1024, 64), the external vllm_triton_prefill baseline is faster in both supplied measurements: bench reports 0.1338 → 0.0818 ms, and NCU reports 185.454 → 80.74 µs. The directions agree, but the measured ratios differ: 1.6356968215158925 in bench versus 2.296928412187268 in NCU. The source replaces the generated kernel with a substantially different implementation; individual changes are mechanisms to investigate, not measured causes of the timing difference.

**Larger query and key tiles with fewer programs.** Larger query tiles offer more K/V reuse within each program, while larger key tiles change dot-product and softmax work granularity. Larger accumulators also increase resource demand.
Evidence: The generated wrapper uses BLOCK_M=16 and BLOCK_N=64. The baseline's >= sm_80 rule selects 128 for both dimensions; NCU records grid size 1024 → 128, with block size 128 unchanged.
Fair note: NCU reports registers/thread 64 → 255 and occupancy 16.162% → 13.894%; the faster implementation does not have higher occupancy. Tiling is not isolated experimentally.

**Explicit base-2 scaling and accumulator-form dot.** The baseline folds the base-2 conversion into the score scale and expresses the P·V update using the dot accumulator argument, changing arithmetic expression and compiler optimization opportunities.
Evidence: Generated uses tl.exp and acc * alpha + tl.dot(...). Baseline scales sm_scale by RCP_LN2, uses tl.math.exp2, and updates with tl.dot(p, v, acc).
Fair note: Both implement online softmax. Generated tl.exp may itself lower through base-2 instructions, so the source difference does not establish an instruction-count reduction.

**Explicit single-stage configuration.** Stage count changes the balance between load/compute overlap and resource pressure.
Evidence: Baseline explicitly passes num_stages=1; generated omits num_stages. Both pass num_warps=4 for this head dimension.
Fair note: The generated kernel's effective stage count is not provided. Calling this an addition of software pipelining or increased overlap is not supported.

**Flattened batch/head layout with sequence metadata.** The baseline changes program ordering and address construction, trading direct batch/head offsets for per-program sequence metadata loads.
Evidence: Generated uses grid (query blocks, batch*heads) and batch/head decomposition. Baseline views tensors as (-1, 1, D), launches (batch, head, query blocks), and loads B_Seqlen and B_Start_Loc.
Fair note: This adds metadata accesses rather than removing them. Both signatures use runtime strides; the diff does not establish a general removal of constexpr addressing.

**Masks retained; causal pruning inactive.** Mask handling and loop bounds can affect work, but only active paths are relevant to this case.
Evidence: Both retain masked Q/K/V loads and output stores. Baseline uses runtime sequence-length masks and adds causal loop-bound pruning, while this case has causal=false.
Fair note: The supplied 'mask-free fast path added' label is not supported by the full diff. Divisible dimensions alone do not prove runtime sequence masks are eliminated; causal pruning cannot explain this non-causal comparison.

What did not change: Both use tiled online softmax without materializing the full score matrix.; Both maintain fp32 accumulation and return output with the input dtype.; Both preserve masked output stores and the kernel_function(Q, K, V, causal) entry signature.

- Both sides have usable NCU captures, but profiling observations do not isolate individual mechanisms.
- NCU classifies generated as memory-hierarchy limited and baseline as compute limited; these are profile classifications, not causal proofs.
- This is an archived external baseline comparison, not a controlled one-change A/B.
- Baseline metadata tensors are memoized during warmup; first-call setup cost is not represented by the steady-state comparison.
- Additional baseline capabilities include variable sequence lengths, GQA, sliding windows, and attention sinks; these are not active benefits for this case.

**LLM takeaway:** Both bench and NCU support a measured advantage for the external baseline on this case, with different magnitudes. Larger tiles, changed arithmetic expression, launch mapping, and stage configuration are plausible mechanisms, but none is established as the cause.

### Pair: `rounds_5` → `rounds_10`

Sources: `rounds_5` [optimized_kernel.py](../problems/flash_attention/case_0/rounds_5/flash_attention/optimized_kernel.py) sha `138f765d4df3…` · `rounds_10` [optimized_kernel.py](../problems/flash_attention/case_0/rounds_10/flash_attention/optimized_kernel.py) sha `9b1431e00661…` · full diff: [flash_attention__rounds_5_to_rounds_10.diff](source-diffs/flash_attention__rounds_5_to_rounds_10.diff)

#### Evidence

| Metric | from | to | Δ |
|---|---:|---:|---:|
| Fixed-harness kernel (ms) | 0.437 | 0.307 | -0.130 |
| Fixed-harness TFLOPS | 9.820 | 13.980 | +4.160 |
| PyTorch reference (ms) | 0.330 | 0.354 | +0.024 |
| NCU duration (µs) | 72.624 | 68.597 | -4.027 |
| NCU SM % | 54.565 | 58.061 | +3.5 |
| NCU memory % | 20.845 | 22.142 | +1.3 |
| NCU DRAM % | 16.678 | 17.146 | +0.5 |
| NCU occupancy % | 12.984 | 13.308 | +0.3 |
| Registers/thread | 248.000 | 244.000 | -4 |
| NCU TFLOPS | 59.140 | 62.610 | +3.471 |
| NCU grid / block | 128 / 128 | 128 / 128 | — |

#### 1. Masking or predication changed
Source lines: from 33-33 → to 34-34

```diff
             k = tl.load(
                 k_ptrs + start * HEAD_DIM,
                 mask=keys[None, :] < SEQ,
-                other=0.0,
+                other=0,
             )
             v = tl.load(
                 v_ptrs + start * HEAD_DIM,
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 2. Autotune candidate space expanded or changed
Source lines: from 65-70 → to 68-107

```diff
 
 @triton.autotune(
     configs=[
-        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
-        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
-        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
-        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
-        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
-        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
+        triton.Config(
+            {"BLOCK_M": 32, "BLOCK_N": 64},
+            num_warps=2,
+            num_stages=3,
+        ),
+        triton.Config(
... 22 changed lines omitted ...
+            num_stages=3,
+        ),
+        triton.Config(
+            {"BLOCK_M": 128, "BLOCK_N": 64},
+            num_warps=4,
+            num_stages=3,
+        ),
+        triton.Config(
+            {"BLOCK_M": 128, "BLOCK_N": 64},
+            num_warps=8,
+            num_stages=2,
+        ),
     ],
     key=["SEQ", "HEAD_DIM", "BATCH_HEADS", "CAUSAL"],
 )
```

**Why it matters:** The candidate tile/warp/stage space changes. A different candidate set can select a better register/shared-memory/occupancy balance. The source alone does not prove which configuration ran; NCU grid/block/occupancy is the evidence.

#### 3. Software pipelining — changed
Source lines: from 23-23 → to 23-24

```diff
     k_ptrs = K + base + dims[:, None] + cols[None, :] * HEAD_DIM
     v_ptrs = V + base + cols[:, None] * HEAD_DIM + dims[None, :]
 
-    for start in tl.range(lo, hi, BLOCK_N):
+    for start in tl.range(begin, end, BLOCK_N):
+        start = tl.multiple_of(start, BLOCK_N)
         keys = start + cols
 
         if SEQ % BLOCK_N == 0:
```

**Why it matters:** Pipeline depth or loop form changes: more stages overlap global loads with tensor-core work at higher shared-memory/register cost; fewer stages reduce pressure but expose load latency.

#### 4. Causal loop specialization — added in `to`
Source lines: from 101-105 → to 139-143

```diff
     acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
 
     if CAUSAL:
-        diagonal_start = tile * BLOCK_M
-        acc, normalizer, maximum = _attention_inner(
-            q, K, V, base, rows, dims,
-            acc, normalizer, maximum,
-            0, diagonal_start,
+        prefix_end = (query_block * BLOCK_M // BLOCK_N) * BLOCK_N
+
+        m, l, acc = _attention_blocks(
+            q, K, V, base, rows, m, l, acc,
+            0, prefix_end,
             SEQ, HEAD_DIM, BLOCK_M, BLOCK_N, False,
         )
```

**Why it matters:** The causal loop is split into an unmasked prefix and a masked diagonal, or fully masked blocks are pruned. Only causal cases are affected. This case is non-causal, so the change cannot affect its hot loop; it matters for causal cases.

#### 5. Grid and program addressing — changed
Source lines: from 84-90 → to 121-128

```diff
     BLOCK_M: tl.constexpr,
     BLOCK_N: tl.constexpr,
 ):
-    tile = tl.program_id(0)
-    head = tl.program_id(1)
-    base = head * SEQ * HEAD_DIM
-    rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
-    dims = tl.arange(0, HEAD_DIM)
-
-    q_ptrs = Q + base + rows[:, None] * HEAD_DIM + dims[None, :]
+    query_block = tl.program_id(0)
+    batch_head = tl.program_id(1)
+    base = batch_head * SEQ * HEAD_DIM
+
+    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
+    dims = tl.arange(0, HEAD_DIM)
+    q_ptrs = Q + base + rows[:, None] * HEAD_DIM + dims[None, :]
+
     if SEQ % BLOCK_M == 0:
         q = tl.load(q_ptrs)
     else:
```

**Why it matters:** Program-ID mapping, launch geometry, or address construction changes. Flattening axes or replacing per-program metadata loads with direct offsets removes fixed integer/global-load work per CTA and changes how tiles map to L2.

#### 6. Online-softmax recurrence — added in `to`
Source lines: from 50-60 → to 53-63

```diff
             )
 
-        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1) * scale)
-        probabilities = tl.exp2(scores * scale - new_maximum[:, None])
-        correction = tl.exp2(maximum - new_maximum)
-        block_sum = tl.sum(probabilities, axis=1)
-
-        acc = acc * correction[:, None]
-        acc = tl.dot(probabilities.to(q.dtype), v, acc)
-        normalizer = normalizer * correction + block_sum
-        maximum = new_maximum
-
-    return acc, normalizer, maximum
+        next_m = tl.maximum(m, tl.max(scores, axis=1))
+        p = tl.exp2(scores - next_m[:, None])
+        alpha = tl.exp2(m - next_m)
+        block_l = tl.sum(p, axis=1)
+
+        acc = acc * alpha[:, None]
+        acc = tl.dot(p.to(q.dtype), v, acc)
+        l = l * alpha + block_l
+        m = next_m
+
+    return m, l, acc
 
 
 @triton.autotune(
```

**Why it matters:** The online-softmax recurrence is touched. This is correctness-critical: changes affect numerical stability and rescale/dot work, not just scheduling; verify with the fixed test.

#### What did *not* change

- **fp32 accumulation preserved.** Both accumulate in fp32.
- **Masked output store preserved.** Both mask the output store.
- **Autotune key preserved.** Both autotune on the same key list.
- **Kernel entry signature preserved.** Both expose the same `kernel_function` signature.

#### Caveats

- NCU duration is below the fixed-harness time for `rounds_5`, `rounds_10`; the benchmark and profiler likely selected different autotune configurations or ran at different clocks.

**Takeaway:** Detected mask_elision, autotune_space, pipelining, causal_specialization, grid_addressing, online_softmax; fixed harness 0.437 → 0.307 ms; NCU 72.624 → 68.597 µs.

#### LLM commentary

For non-causal fp16 attention with B=2, H=8, S=1024, and D=64, rounds_10 measures faster than rounds_5 in both datasets: benchmark time is 0.4374 → 0.3072 ms, and NCU kernel duration is 72.624 → 68.597 µs. The measurements agree on direction but differ substantially in magnitude. The diff changes autotune candidates, adds a loop-alignment hint, and rewrites softmax scaling and output normalization; no individual edit is established as the measured cause.

**Tile, warp, and stage candidates revised.** Different candidates offer different reuse, parallelism, and resource-pressure tradeoffs.
Evidence: The candidate list grows from six to eight configurations, introduces BLOCK_N=128 and two-warp candidates, and mostly uses three stages. NCU reports registers/thread 248 → 244, occupancy 12.984% → 13.308%, and SM utilization 54.565% → 58.061%.
Fair note: Both captures report grid=128 and block=128. The selected configuration is not explicitly supplied, and these resource changes do not isolate an autotuning effect.

**Explicit BLOCK_N alignment hint added.** The compiler receives additional information about loop-index alignment, potentially affecting address generation or optimization.
Evidence: The new inner loop adds `start = tl.multiple_of(start, BLOCK_N)`. Both versions retain a `tl.range` traversal.
Fair note: The loop form itself is not newly pipelined. Stage-count changes belong to the candidate configurations, and no generated-code evidence isolates the hint's effect.

**Score scaling moved before the row maximum.** Moving scaling changes the expression graph used by the stable online-softmax recurrence and may affect scheduling or floating-point behavior.
Evidence: Previously, raw dot products fed `tl.max(scores) * scale`, while probabilities used `scores * scale`. The new version computes `scores = tl.dot(q, k) * scale` before the maximum and exponentials.
Fair note: Online softmax already existed; it was rewritten, not added. The recurrence remains mathematically equivalent for these finite inputs, but no numerical-error results are provided.

**Reciprocal multiplication replaced by division.** Expression form and explicit conversion can affect lowering and numerical rounding.
Evidence: `reciprocal = 1.0 / normalizer; result = acc * reciprocal[:, None]` becomes `(acc / l[:, None]).to(O.dtype.element_ty)`.
Fair note: Source syntax alone does not establish different machine instructions or a timing contribution.

**Existing causal split receives aligned prefix bounds.** Aligning the prefix boundary changes which causal blocks use masking, particularly when tile dimensions differ.
Evidence: The prefix endpoint changes from `tile * BLOCK_M` to `(query_block * BLOCK_M // BLOCK_N) * BLOCK_N`. Both versions already have unmasked-prefix and masked-diagonal calls.
Fair note: CAUSAL=false here, so this branch is inactive and cannot explain this case's hot-loop measurements.

**Flagged mask and grid edits are largely nonfunctional here.** Separating renaming from substantive changes avoids attributing performance to unsupported mechanisms.
Evidence: Program IDs and address formulas are unchanged apart from names. Masked-load fallback values change from `0.0` to `0`; both versions already bypass load masks when tile dimensions divide SEQ.
Fair note: All listed tile dimensions divide 1024. The diff does not demonstrate new mask elimination or a changed grid mapping.

What did not change: FP32 accumulation is preserved.; Autotune keys remain SEQ, HEAD_DIM, BATCH_HEADS, and CAUSAL.; The kernel_function signature and two-dimensional launch-grid formula are preserved.; Both retain conditional unmasked stores for divisible sequence lengths and masked fallback stores otherwise.

- Both sides have usable NCU captures, but profiling does not isolate individual edits.
- NCU durations are substantially below benchmark times; different autotune selections or clocks are possible explanations, not established facts.
- Both profiles remain compute-bound with low occupancy and high register pressure.
- The empty-input early return is removed, but this case is nonempty.

**LLM takeaway:** rounds_10 has lower measured times in both datasets, with a much smaller relative improvement under NCU. The source offers plausible optimization mechanisms, but the evidence does not identify a particular change as the cause.

### Pair: `rounds_5` → `vllm_triton_prefill`

Sources: `rounds_5` [optimized_kernel.py](../problems/flash_attention/case_0/rounds_5/flash_attention/optimized_kernel.py) sha `138f765d4df3…` · `vllm_triton_prefill` [kernel.py](../problems/flash_attention/baselines/vllm_triton_prefill/flash_attention/kernel.py) sha `2c38b86851dd…` · full diff: [flash_attention__rounds_5_to_vllm_triton_prefill.diff](source-diffs/flash_attention__rounds_5_to_vllm_triton_prefill.diff)

#### Evidence

| Metric | from | to | Δ |
|---|---:|---:|---:|
| Fixed-harness kernel (ms) | 0.437 | 0.082 | -0.356 |
| Fixed-harness TFLOPS | 9.820 | 52.510 | +42.690 |
| PyTorch reference (ms) | 0.330 | 0.064 | -0.266 |
| NCU duration (µs) | 72.624 | 80.740 | +8.116 |
| NCU SM % | 54.565 | 49.470 | -5.1 |
| NCU memory % | 20.845 | 36.277 | +15.4 |
| NCU DRAM % | 16.678 | 16.077 | -0.6 |
| NCU occupancy % | 12.984 | 13.894 | +0.9 |
| Registers/thread | 248.000 | 255.000 | +7 |
| NCU TFLOPS | 59.140 | 53.196 | -5.944 |
| NCU grid / block | 128 / 128 | 128 / 128 | — |

#### 1. Mask-free fast path added
Source lines: from — → to 145-166

```diff
+
+    q = tl.load(
+        Q + off_q,
+        mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :]),
+        other=0.0,
+    )
+
+    k_ptrs = K + off_k
+    v_ptrs = V + off_v
+
+    # initialize pointer to m and l
+    if USE_SINKS:
+        sink = tl.load(Sinks + cur_head) * 1.4426950408889634
+        if SINKS_BIAS_KEY0:
+            # Sinks bias the logit of key 0, so the softmax starts empty and
+            # normalizes over the biased scores as usual.
+            m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
+            l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
+        else:
+            # Sinks are a null logit that only inflates the denominator.
+            m_i = tl.full([BLOCK_M], sink, dtype=tl.float32)
+            l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 2. Autotune candidate space reduced or changed
Source lines: from 63-88 → to —

```diff
-@triton.autotune(
-    configs=[
-        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
-        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
-        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
-        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
-        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
-        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
-    ],
-    key=["SEQ", "HEAD_DIM", "BATCH_HEADS", "CAUSAL"],
-)
-@triton.jit
... 2 changed lines omitted ...
-    SEQ: tl.constexpr,
-    HEAD_DIM: tl.constexpr,
-    BATCH_HEADS: tl.constexpr,
-    CAUSAL: tl.constexpr,
-    BLOCK_M: tl.constexpr,
-    BLOCK_N: tl.constexpr,
-):
-    tile = tl.program_id(0)
-    head = tl.program_id(1)
-    base = head * SEQ * HEAD_DIM
-    rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
-    dims = tl.arange(0, HEAD_DIM)
```

**Why it matters:** The candidate tile/warp/stage space changes. A different candidate set can select a better register/shared-memory/occupancy balance. The source alone does not prove which configuration ran; NCU grid/block/occupancy is the evidence.

#### 3. Causal loop specialization — added in `to`
Source lines: from — → to 93-123

```diff
+def _fwd_kernel(
+    Q,
+    K,
+    V,
+    Sinks,
+    sm_scale,
+    B_Start_Loc,
+    B_Seqlen,
+    Out,
+    stride_qbs,
+    stride_qh,
+    stride_kbs,
... 7 changed lines omitted ...
+    BLOCK_DMODEL: tl.constexpr,
+    BLOCK_N: tl.constexpr,
+    IS_CAUSAL: tl.constexpr,
+    SLIDING_WINDOW_Q: tl.constexpr,
+    SLIDING_WINDOW_K: tl.constexpr,
+    SINKS_BIAS_KEY0: tl.constexpr,
+    USE_SINKS: tl.constexpr,
+    Lk: tl.constexpr,
+):
+    cur_batch = tl.program_id(0)
+    cur_head = tl.program_id(1)
+    start_m = tl.program_id(2)
```

**Why it matters:** The causal loop is split into an unmasked prefix and a masked diagonal, or fully masked blocks are pruned. Only causal cases are affected. This case is non-causal, so the change cannot affect its hot loop; it matters for causal cases.

#### 4. Software pipelining — added in `to`
Source lines: from — → to 314-344

```diff
+
+    _fwd_kernel[grid](
+        q,
+        k,
+        v,
+        sinks if sinks is not None else q,
+        sm_scale,
+        b_start_loc,
+        b_seq_len,
+        o,
+        q.stride(0),
+        q.stride(1),
... 7 changed lines omitted ...
+        BLOCK_M=BLOCK,
+        BLOCK_DMODEL=triton.next_power_of_2(Lk),
+        BLOCK_N=BLOCK_N,
+        IS_CAUSAL=is_causal,
+        SLIDING_WINDOW_Q=sliding_window_q,
+        SLIDING_WINDOW_K=sliding_window_k,
+        USE_SINKS=sinks is not None,
+        SINKS_BIAS_KEY0=sinks_bias_key0,
+        num_warps=num_warps,
+        num_stages=1,
+        Lk=Lk,
+    )
```

**Why it matters:** Pipeline depth or loop form changes: more stages overlap global loads with tensor-core work at higher shared-memory/register cost; fewer stages reduce pressure but expose load latency.

#### 5. Grid and program addressing — added in `to`
Source lines: from — → to 124-144

```diff
+
+    cur_kv_head = cur_head // kv_group_num
+
+    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
+    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
+
+    block_start_loc = BLOCK_M * start_m
+
+    # initialize offsets
+    offs_n = tl.arange(0, BLOCK_N)
+    offs_d = tl.arange(0, BLOCK_DMODEL)
+    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
+    off_q = (
+        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
+        + cur_head * stride_qh
+        + offs_d[None, :]
+    )
+    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
+    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]
+
+    mask_d = offs_d < Lk
```

**Why it matters:** Program-ID mapping, launch geometry, or address construction changes. Flattening axes or replacing per-program metadata loads with direct offsets removes fixed integer/global-load work per CTA and changes how tiles map to L2.

#### 6. Online-softmax recurrence — removed in `to`
Source lines: from 49-60 → to —

```diff
-
-        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1) * scale)
-        probabilities = tl.exp2(scores * scale - new_maximum[:, None])
-        correction = tl.exp2(maximum - new_maximum)
-        block_sum = tl.sum(probabilities, axis=1)
-
-        acc = acc * correction[:, None]
-        acc = tl.dot(probabilities.to(q.dtype), v, acc)
-        normalizer = normalizer * correction + block_sum
-        maximum = new_maximum
-
-    return acc, normalizer, maximum
```

**Why it matters:** The online-softmax recurrence is touched. This is correctness-critical: changes affect numerical stability and rescale/dot work, not just scheduling; verify with the fixed test.

#### What did *not* change

- **fp32 accumulation preserved.** Both accumulate in fp32.
- **Masked output store preserved.** Both mask the output store.
- **Kernel entry signature preserved.** Both expose the same `kernel_function` signature.

#### Caveats

- The PyTorch reference times differ between the two records (0.330 ms vs 0.064 ms); the kernel ratio is fixed-harness evidence, not a same-session controlled A/B.
- The fixed harness and NCU disagree on direction. NCU locks clocks at or below boost, so its duration should be greater than or equal to the fixed-harness time; this pattern is most plausibly a different autotune configuration or a clock/power difference. Use NCU for code-level attribution and do not claim a fixed-harness win for this pair.
- NCU duration is below the fixed-harness time for `rounds_5`; the benchmark and profiler likely selected different autotune configurations or ran at different clocks.
- `vllm_triton_prefill` is an archived external baseline (`flash_attention/baselines/vllm_triton_prefill.py`). This is a source comparison, not a controlled one-change A/B; it also supports variable sequence lengths, GQA, attention sinks.

**Takeaway:** Detected mask_elision, autotune_space, causal_specialization, pipelining, grid_addressing, online_softmax; fixed harness 0.437 → 0.082 ms; NCU 72.624 → 80.740 µs.

#### LLM commentary

This compares rounds_5 with an archived vLLM prefill baseline for non-causal FP16 attention at (B,H,S,D)=(2,8,1024,64). Bench reports 0.4374 → 0.0818 ms, but NCU reports 72.624 → 80.74 µs, reversing the ordering; no fixed-harness win is claimed. The full diff shows changes to tiling, staging, masking, addressing, and softmax expression, not a controlled single-change experiment.

**Autotuned configurations replaced by heuristic tiling and one stage.** Tile sizes and pipeline depth change reuse, loop work, and register/shared-memory demands. These are plausible mechanisms, not established explanations for the timing difference.
Evidence: rounds_5 has six configurations with BLOCK_M=32/64/128, BLOCK_N=32/64, four warps, and two or three stages. The baseline uses heuristic BLOCK_M=BLOCK_N, four warps for D=64, and num_stages=1; its heuristic selects 128 on CUDA capability >=8. Both NCU captures report grid=128 and block=128.
Fair note: The exact rounds_5 autotune selection is not supplied. NCU registers increase from 248 to 255 per thread, while occupancy increases from 12.984% to 13.894%; neither establishes a staging benefit.

**Compile-time mask-free paths replaced by runtime sequence masking.** Runtime bounds predicates and score selects can add work compared with compile-time divisibility specialization.
Evidence: rounds_5 uses SEQ % BLOCK_M/BLOCK_N guards, which resolve to unconditional accesses for this case. The baseline loads sequence lengths from B_Seqlen and predicates Q/K/V loads, score selection, and output stores against them.
Fair note: The supplied 'mask-free fast path added' label reverses the direction visible in the full diff. D=64 permits dimension predicates to simplify, but sequence length is runtime metadata in the baseline.

**Direct offsets replaced by metadata-driven sequence addressing.** Program mapping and per-program metadata loads change address-generation work and tile traversal.
Evidence: rounds_5 maps program IDs to query tile and flattened batch/head with base=head*SEQ*HEAD_DIM. The baseline uses batch/head/query-tile axes and loads B_Seqlen and B_Start_Loc; its wrapper flattens B*H into sequences through zero-copy views.
Fair note: Both profiled launches contain 128 CTAs with 128 threads each. Metadata tensors are cached after warmup, so their construction is not represented as recurring work in the captured region.

**Online recurrence retained, with changed scaling and operation ordering.** Scaling placement, accumulator update ordering, and normalization expression affect arithmetic and scheduling while remaining correctness-sensitive.
Evidence: Both maintain running maxima, normalization sums, and FP32 output accumulators using exp2. The baseline scales qk before its maximum reduction, loads V later in the loop, and ends with acc/l_i; rounds_5 scales the reduced maximum and ends with reciprocal multiplication.
Fair note: The recurrence was not removed. Constants also differ in their source representation, and no numerical-error or correctness-test results are supplied.

**Split causal prefix/diagonal replaced by a masked loop.** For causal inputs, unmasked-prefix specialization differs from applying causal predicates throughout the bounded loop.
Evidence: rounds_5 invokes separate unmasked-prefix and masked-diagonal helper passes. The baseline bounds end_n and applies pos_q >= pos_k inside its loop.
Fair note: This case is non-causal, so causal specialization cannot explain its hot-loop timing. Sliding-window and sink features are also disabled by this wrapper call.

What did not change: Same kernel_function(Q, K, V, causal) contract and output shape/dtype.; FP32 accumulation and tiled online softmax are preserved.; Both sources support bounds-masked output stores; rounds_5 selects an unconditional store for this divisible case.

- Both sides have usable NCU captures, but individual-change attribution remains unisolated.
- Reference timings differ: 0.3296 versus 0.0638 ms, so this is not a same-session controlled A/B.
- rounds_5 NCU duration is below its bench duration. Different autotune selections or clock/power conditions are possible explanations, not established facts.
- NCU classifies both kernels as compute-bound with low occupancy and high register pressure.

**LLM takeaway:** The strongest supported result is disagreement between bench and NCU, not a demonstrated performance win. The source identifies concrete scheduling and masking mechanisms, but does not establish which change explains either timing ordering.

### Pair: `rounds_10` → `vllm_triton_prefill`

Sources: `rounds_10` [optimized_kernel.py](../problems/flash_attention/case_0/rounds_10/flash_attention/optimized_kernel.py) sha `9b1431e00661…` · `vllm_triton_prefill` [kernel.py](../problems/flash_attention/baselines/vllm_triton_prefill/flash_attention/kernel.py) sha `2c38b86851dd…` · full diff: [flash_attention__rounds_10_to_vllm_triton_prefill.diff](source-diffs/flash_attention__rounds_10_to_vllm_triton_prefill.diff)

#### Evidence

| Metric | from | to | Δ |
|---|---:|---:|---:|
| Fixed-harness kernel (ms) | 0.307 | 0.082 | -0.225 |
| Fixed-harness TFLOPS | 13.980 | 52.510 | +38.530 |
| PyTorch reference (ms) | 0.354 | 0.064 | -0.290 |
| NCU duration (µs) | 68.597 | 80.740 | +12.143 |
| NCU SM % | 58.061 | 49.470 | -8.6 |
| NCU memory % | 22.142 | 36.277 | +14.1 |
| NCU DRAM % | 17.146 | 16.077 | -1.1 |
| NCU occupancy % | 13.308 | 13.894 | +0.6 |
| Registers/thread | 244.000 | 255.000 | +11 |
| NCU TFLOPS | 62.610 | 53.196 | -9.415 |
| NCU grid / block | 128 / 128 | 128 / 128 | — |

#### 1. Mask-free fast path added
Source lines: from — → to 145-166

```diff
+
+    q = tl.load(
+        Q + off_q,
+        mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :]),
+        other=0.0,
+    )
+
+    k_ptrs = K + off_k
+    v_ptrs = V + off_v
+
+    # initialize pointer to m and l
+    if USE_SINKS:
+        sink = tl.load(Sinks + cur_head) * 1.4426950408889634
+        if SINKS_BIAS_KEY0:
+            # Sinks bias the logit of key 0, so the softmax starts empty and
+            # normalizes over the biased scores as usual.
+            m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
+            l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
+        else:
+            # Sinks are a null logit that only inflates the denominator.
+            m_i = tl.full([BLOCK_M], sink, dtype=tl.float32)
+            l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 2. Autotune candidate space reduced or changed
Source lines: from 66-110 → to —

```diff
-@triton.autotune(
-    configs=[
-        triton.Config(
-            {"BLOCK_M": 32, "BLOCK_N": 64},
-            num_warps=2,
-            num_stages=3,
-        ),
-        triton.Config(
-            {"BLOCK_M": 32, "BLOCK_N": 128},
-            num_warps=4,
-            num_stages=3,
-        ),
... 21 changed lines omitted ...
-            {"BLOCK_M": 128, "BLOCK_N": 64},
-            num_warps=4,
-            num_stages=3,
-        ),
-        triton.Config(
-            {"BLOCK_M": 128, "BLOCK_N": 64},
-            num_warps=8,
-            num_stages=2,
-        ),
-    ],
-    key=["SEQ", "HEAD_DIM", "BATCH_HEADS", "CAUSAL"],
-)
```

**Why it matters:** The candidate tile/warp/stage space changes. A different candidate set can select a better register/shared-memory/occupancy balance. The source alone does not prove which configuration ran; NCU grid/block/occupancy is the evidence.

#### 3. Causal loop specialization — added in `to`
Source lines: from — → to 93-123

```diff
+def _fwd_kernel(
+    Q,
+    K,
+    V,
+    Sinks,
+    sm_scale,
+    B_Start_Loc,
+    B_Seqlen,
+    Out,
+    stride_qbs,
+    stride_qh,
+    stride_kbs,
... 7 changed lines omitted ...
+    BLOCK_DMODEL: tl.constexpr,
+    BLOCK_N: tl.constexpr,
+    IS_CAUSAL: tl.constexpr,
+    SLIDING_WINDOW_Q: tl.constexpr,
+    SLIDING_WINDOW_K: tl.constexpr,
+    SINKS_BIAS_KEY0: tl.constexpr,
+    USE_SINKS: tl.constexpr,
+    Lk: tl.constexpr,
+):
+    cur_batch = tl.program_id(0)
+    cur_head = tl.program_id(1)
+    start_m = tl.program_id(2)
```

**Why it matters:** The causal loop is split into an unmasked prefix and a masked diagonal, or fully masked blocks are pruned. Only causal cases are affected. This case is non-causal, so the change cannot affect its hot loop; it matters for causal cases.

#### 4. Software pipelining — removed in `to`
Source lines: from 7-26 → to —

```diff
-def _attention_blocks(
-    q, K, V, base, rows, m, l, acc,
-    begin, end,
-    SEQ: tl.constexpr,
-    HEAD_DIM: tl.constexpr,
-    BLOCK_M: tl.constexpr,
-    BLOCK_N: tl.constexpr,
-    MASK_CAUSAL: tl.constexpr,
-):
-    cols = tl.arange(0, BLOCK_N)
-    dims = tl.arange(0, HEAD_DIM)
-    scale: tl.constexpr = 1.4426950408889634 * HEAD_DIM ** -0.5
-
-    k_ptrs = K + base + dims[:, None] + cols[None, :] * HEAD_DIM
-    v_ptrs = V + base + cols[:, None] * HEAD_DIM + dims[None, :]
-
-    for start in tl.range(begin, end, BLOCK_N):
-        start = tl.multiple_of(start, BLOCK_N)
-        keys = start + cols
-
```

**Why it matters:** Pipeline depth or loop form changes: more stages overlap global loads with tensor-core work at higher shared-memory/register cost; fewer stages reduce pressure but expose load latency.

#### 5. Grid and program addressing — added in `to`
Source lines: from — → to 124-144

```diff
+
+    cur_kv_head = cur_head // kv_group_num
+
+    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
+    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
+
+    block_start_loc = BLOCK_M * start_m
+
+    # initialize offsets
+    offs_n = tl.arange(0, BLOCK_N)
+    offs_d = tl.arange(0, BLOCK_DMODEL)
+    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
+    off_q = (
+        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
+        + cur_head * stride_qh
+        + offs_d[None, :]
+    )
+    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
+    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]
+
+    mask_d = offs_d < Lk
```

**Why it matters:** Program-ID mapping, launch geometry, or address construction changes. Flattening axes or replacing per-program metadata loads with direct offsets removes fixed integer/global-load work per CTA and changes how tiles map to L2.

#### 6. Online-softmax recurrence — removed in `to`
Source lines: from 52-63 → to —

```diff
-
-        next_m = tl.maximum(m, tl.max(scores, axis=1))
-        p = tl.exp2(scores - next_m[:, None])
-        alpha = tl.exp2(m - next_m)
-        block_l = tl.sum(p, axis=1)
-
-        acc = acc * alpha[:, None]
-        acc = tl.dot(p.to(q.dtype), v, acc)
-        l = l * alpha + block_l
-        m = next_m
-
-    return m, l, acc
```

**Why it matters:** The online-softmax recurrence is touched. This is correctness-critical: changes affect numerical stability and rescale/dot work, not just scheduling; verify with the fixed test.

#### What did *not* change

- **fp32 accumulation preserved.** Both accumulate in fp32.
- **Masked output store preserved.** Both mask the output store.
- **Kernel entry signature preserved.** Both expose the same `kernel_function` signature.

#### Caveats

- The PyTorch reference times differ between the two records (0.354 ms vs 0.064 ms); the kernel ratio is fixed-harness evidence, not a same-session controlled A/B.
- The fixed harness and NCU disagree on direction. NCU locks clocks at or below boost, so its duration should be greater than or equal to the fixed-harness time; this pattern is most plausibly a different autotune configuration or a clock/power difference. Use NCU for code-level attribution and do not claim a fixed-harness win for this pair.
- NCU duration is below the fixed-harness time for `rounds_10`; the benchmark and profiler likely selected different autotune configurations or ran at different clocks.
- `vllm_triton_prefill` is an archived external baseline (`flash_attention/baselines/vllm_triton_prefill.py`). This is a source comparison, not a controlled one-change A/B; it also supports variable sequence lengths, GQA, attention sinks.

**Takeaway:** Detected mask_elision, autotune_space, causal_specialization, pipelining, grid_addressing, online_softmax; fixed harness 0.307 → 0.082 ms; NCU 68.597 → 80.740 µs.

#### LLM commentary

This compares rounds_10 with the archived external vllm_triton_prefill baseline for non-causal FP16 attention at (B,H,S,D)=(2,8,1024,64). Bench reports 0.3072 → 0.0818 ms, favoring the external baseline, but usable NCU captures report 68.597 → 80.740 µs, favoring rounds_10. The measurements disagree on direction, so this pair does not establish a fixed-harness win.

**Autotuning replaced by heuristic tiles and one stage.** Tile dimensions and pipeline depth alter loop count, operand reuse, overlap, and resource pressure; these are mechanisms, not isolated measured causes.
Evidence: rounds_10 has eight configurations with two or three stages. The external baseline uses heuristic block sizes, four warps for D=64, and num_stages=1. Both NCU captures report grid=128 and block=128; registers increase from 244 to 255, while occupancy changes from 13.308% to 13.894%.
Fair note: The original selected configuration is not explicitly reported. Equal launch counts and thread counts do not establish identical KV tiles or compiled scheduling, and one stage does not prove all compiler overlap is absent.

**Compile-time mask elision replaced by runtime sequence checks.** Runtime bounds introduce predicates and score selection where the original can use unconditional accesses for divisible dimensions.
Evidence: rounds_10 guards Q/K/V accesses and output stores with constexpr SEQ divisibility checks; all listed candidate tiles divide 1024. The external kernel loads sequence length from B_Seqlen and predicates Q/K/V, scores, and stores against it.
Fair note: The full diff contradicts the supplied 'mask-free fast path added' label: that fast path exists on the original side. D=64 matches the external padded dimension, so dimension predicates may simplify; sequence predicates depend on loaded metadata.

**Direct flattened offsets replaced by metadata-driven addressing.** Per-program metadata loads and stride-based addressing support variable-length sequences but add address-construction work and change program ordering.
Evidence: The original maps query blocks and batch-heads onto two axes with base=batch_head*SEQ*HEAD_DIM. The external wrapper views tensors without copying, folds B*H into sequences, and launches batch/head/query-block axes; each program loads B_Seqlen and B_Start_Loc.
Fair note: Metadata tensors are cached after warmup. NCU reports one kernel on each side and the same total grid size, not evidence of extra metadata-construction launches in the captured region.

**Online softmax retained with arithmetic and load-order differences.** Both kernels rescale running maxima, denominators, and FP32 accumulators; arithmetic details and operand lifetimes can affect numerical behavior and scheduling.
Evidence: Both use exp2, cast probabilities to FP16, and accumulate probability-value products in FP32. The external kernel loads V later, uses RCP_LN2=1.4426950216 rather than 1.4426950408889634, and uses -1.0e8 for masked scores rather than negative infinity.
Fair note: The recurrence is not removed despite the supplied label. No correctness-test result or isolated performance attribution is provided.

**Split causal prefix replaced by per-iteration masking.** For causal inputs, the original separates an unmasked prefix from a masked diagonal; the external kernel prunes the loop endpoint and applies causal masking within iterations.
Evidence: rounds_10 calls _attention_blocks with separate MASK_CAUSAL settings. The external loop checks IS_CAUSAL when forming its mask.
Fair note: This case is non-causal, so these causal branches do not explain its timing differences. Sliding windows and sinks are also disabled by this wrapper invocation.

What did not change: The kernel_function(Q, K, V, causal) interface and output shape/dtype contract are preserved.; Both implement tiled attention with online softmax and FP32 accumulation.; Both provide output bounds protection, although the original elides its store mask for this divisible case.

- Bench reference times differ: 0.354 versus 0.0638 ms; this is not a same-session controlled A/B.
- rounds_10 NCU duration is below its bench time. Different autotune selections or clock/power conditions are possible explanations, not established causes.
- NCU labels both compute-bound with low occupancy and high register pressure; these observations do not isolate individual source changes.
- This is a multi-change source comparison against a more general external baseline.

**LLM takeaway:** The diff supports concrete masking, addressing, scheduling, and arithmetic mechanisms, but no individual change is established as a measured cause. Conflicting bench and NCU directions prevent a fixed-harness win claim.
