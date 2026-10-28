# Code diff analysis: `flashattention-ada5000_v1`

Generated 2026-10-28T17:19:41.411927 by the offline report.

## How to read this

- `-`/`+` in each heading name the `from`/`to` variants; they do not
  mean worse/better.
- Excerpts omit unchanged or repeated lines; code text is verbatim.
- A **Why it matters** paragraph explains a detected mechanism. It is not proof that the mechanism caused a measured difference.
- Benchmark and NCU numbers come from different methodologies and are reported separately. "inferred, not measured on both sides" marks a claim where one side lacks a usable capture.

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
