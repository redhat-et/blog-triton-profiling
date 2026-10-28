# Code diff analysis: `matmulbias-ada5000_v1`

Generated 2026-10-28T22:02:46.502225 by the offline report.

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
- Deterministic fingerprint: `sha256:b7190918fcd2d4d16905ac0a3d5e11aed8710acc079c98b3cfe9263e02368527`

The deterministic sections remain authoritative; the LLM
commentary is an optional reading aid.

## Coverage

| Problem | Pairs analyzed | Skipped variants |
|---|---:|---:|
| `matmul_bias` | 6 | 0 |

Pairs expected: 6; analyzed: 6.

## Summary

- Pairs analyzed: 6
- Change groups detected: 36
- Categories: autotune_space (6), constexpr_addressing (3), dtype_math (3), grid_addressing (5), grouped_swizzle (6), host_launch (1), mask_elision (12)
- Methodology conflicts: 1

## `matmul_bias` — benchmark case `0`

Case: `(1024, 1024, 1024, 'float16')`

Case context: `M=1024`, `N=1024`, `K=1024`, `dtype=float16`

Divisibility: `K % BLOCK_K == 0` (1024 / [32, 64]: all divide); `M % BLOCK_M == 0` (1024 / [32, 64, 128]: all divide); `N % BLOCK_N == 0` (1024 / [64, 128]: all divide)

Sibling note: `rounds_5` and `rounds_10` optimize the same source (`5ba2fb340021…`); they are independent runs, not a chain.

### Pair: `generated` → `rounds_5`

Sources: `generated` [final_kernel.py](../problems/matmul_bias/generated/matmul_bias/final_kernel.py) sha `5ba2fb340021…` · `rounds_5` [optimized_kernel.py](../problems/matmul_bias/case_0/rounds_5/matmul_bias/optimized_kernel.py) sha `e71fe54ed298…` · full diff: [matmul_bias__generated_to_rounds_5.diff](source-diffs/matmul_bias__generated_to_rounds_5.diff)

#### Evidence

| Metric | from | to | Δ |
|---|---:|---:|---:|
| Fixed-harness kernel (ms) | 0.040 | 0.025 | -0.015 |
| Fixed-harness TFLOPS | 53.100 | 85.050 | +31.950 |
| PyTorch reference (ms) | 0.026 | 0.024 | -0.002 |
| NCU duration (µs) | 53.156 | 35.613 | -17.543 |
| NCU SM % | 38.158 | 57.470 | +19.3 |
| NCU memory % | 69.740 | 53.679 | -16.1 |
| NCU DRAM % | 14.900 | 22.131 | +7.2 |
| NCU occupancy % | 53.983 | 14.208 | -39.8 |
| Registers/thread | 53.000 | 150.000 | +97 |
| NCU TFLOPS | 40.400 | 60.301 | +19.901 |
| NCU grid / block | 512 / 128 | 128 / 128 | — |

#### 1. Mask-free fast path added
Source lines: from — → to 73-92

```diff
+    for block in range(tl.cdiv(K, BLOCK_K)):
+        if M % BLOCK_M == 0 and K % BLOCK_K == 0:
+            a = tl.load(a_ptrs)
+        else:
+            a = tl.load(
+                a_ptrs,
+                mask=(rows[:, None] < M)
+                & (block * BLOCK_K + ks[None, :] < K),
+                other=0,
+            )
+
+        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
+            b = tl.load(b_ptrs)
+        else:
+            b = tl.load(
+                b_ptrs,
+                mask=(block * BLOCK_K + ks[:, None] < K)
+                & (cols[None, :] < N),
+                other=0,
+            )
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 2. Autotune candidate space expanded or changed
Source lines: from — → to 6-34

```diff
 import triton.language as tl
 
 
+@triton.autotune(
+    configs=[
+        triton.Config(
+            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
+            num_warps=4, num_stages=3,
+        ),
+        triton.Config(
+            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
+            num_warps=4, num_stages=4,
+        ),
+        triton.Config(
+            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
... 5 changed lines omitted ...
+        ),
+        triton.Config(
+            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
+            num_warps=4, num_stages=4,
+        ),
+        triton.Config(
+            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
+            num_warps=8, num_stages=3,
+        ),
+    ],
+    key=["M", "N", "K"],
+)
 @triton.jit
 def _matmul_bias_kernel(
```

**Why it matters:** The candidate tile/warp/stage space changes. A different candidate set can select a better register/shared-memory/occupancy balance. The source alone does not prove which configuration ran; NCU grid/block/occupancy is the evidence.

#### 3. Grouped CTA order — added in `to`
Source lines: from 26-28 → to 53-62

```diff
     BLOCK_N: tl.constexpr,
     BLOCK_K: tl.constexpr,
 ):
-    """Compute one output tile of A @ B + bias, accumulating in fp32."""
-    pid_m = tl.program_id(0)
-    pid_n = tl.program_id(1)
+    pid = tl.program_id(0)
+    num_m = tl.cdiv(M, BLOCK_M)
+    num_n = tl.cdiv(N, BLOCK_N)
+    group_size = 8 * num_n
+    group_id = pid // group_size
+    first_m = group_id * 8
+    group_m = tl.minimum(num_m - first_m, 8)
+    local_pid = pid % group_size
+    pid_m = first_m + local_pid % group_m
+    pid_n = local_pid // group_m
 
     rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
```

**Why it matters:** Grouped CTA ordering changes which output tiles run concurrently. This is an L2-reuse heuristic for the A/B panels, not an arithmetic change.

#### 4. Compile-time strides and index arithmetic — removed in `to`
Source lines: from 15-21 → to 44-48

```diff
     M: tl.constexpr,
     N: tl.constexpr,
     K: tl.constexpr,
-    stride_am,
-    stride_ak,
-    stride_bk,
-    stride_bn,
-    stride_bias,
-    stride_om,
-    stride_on,
+    STRIDE_AM: tl.constexpr,
+    STRIDE_AK: tl.constexpr,
+    STRIDE_BK: tl.constexpr,
+    STRIDE_BN: tl.constexpr,
+    STRIDE_BIAS: tl.constexpr,
     BLOCK_M: tl.constexpr,
     BLOCK_N: tl.constexpr,
     BLOCK_K: tl.constexpr,
```

**Why it matters:** Runtime dimensions/strides become `tl.constexpr` and/or modulo wrapping is removed. Compile-time constants let Triton fold address arithmetic, remove integer divide/modulo from every element's index path, and prove alignment.

#### 5. Grid and program addressing — changed
Source lines: from 77-80 → to 124-130

```diff
 
     output = torch.empty((M, N), device=A.device, dtype=A.dtype)
-
-    block_m, block_n, block_k = 32, 64, 32
-    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
-
+    if M == 0 or N == 0:
+        return output
+
+    grid = lambda meta: (
+        triton.cdiv(M, meta["BLOCK_M"])
+        * triton.cdiv(N, meta["BLOCK_N"]),
+    )
     _matmul_bias_kernel[grid](
         A,
         B,
```

**Why it matters:** Program-ID mapping, launch geometry, or address construction changes. Flattening axes or replacing per-program metadata loads with direct offsets removes fixed integer/global-load work per CTA and changes how tiles map to L2.

#### 6. Dtype and math path — added in `to`
Source lines: from 67-68 → to 114-119

```diff
     M, K = A.shape
     K_b, N = B.shape
-    if K != K_b or bias.numel() != N:
-        raise ValueError("Incompatible matrix or bias dimensions")
+    if K != K_b or bias.ndim != 1 or bias.numel() != N:
+        raise ValueError("Incompatible matrix or bias dimensions")
+    if A.dtype not in (torch.float16, torch.bfloat16):
+        raise TypeError("Only float16 and bfloat16 inputs are supported")
+    if A.dtype != B.dtype or A.dtype != bias.dtype:
+        raise ValueError("All inputs must have the same dtype")
     if A.device != B.device or A.device != bias.device:
         raise ValueError("All inputs must be on the same device")
```

**Why it matters:** Dtype conversion points or math functions change. fp32 accumulation and fewer conversions trade precision against conversion/instruction cost; `exp2` avoids a base conversion.

#### What did *not* change

- **fp32 accumulation preserved.** Both accumulate in fp32.
- **Masked output store preserved.** Both mask the output store.

**Takeaway:** Detected mask_elision, autotune_space, grouped_swizzle, constexpr_addressing, grid_addressing, dtype_math; fixed harness 0.040 → 0.025 ms; NCU 53.156 → 35.613 µs.

#### LLM commentary

For the 1024×1024×1024 float16 case, rounds_5 replaces fixed tiling with autotuning, grouped tile scheduling, compile-time strides, and divisible-shape load fast paths. The fixed harness reports 0.0404 → 0.0253 ms, while NCU reports 53.156 → 35.613 µs; both favor rounds_5, although their absolute timings and improvement magnitudes differ. These measurements establish a pair-level improvement, not which individual edit explains it.

**Fixed tile replaced by six autotune candidates.** Larger output tiles can increase operand reuse while changing register demand, parallelism, and pipeline staging.
Evidence: The original uses 32×64×32 with four warps; rounds_5 searches M/N tiles of 64 or 128, K tiles of 32 or 64, and multiple warp/stage settings. NCU grid size changes from 512 to 128, with 128 threads per block on both sides; registers rise from 53 to 150 per thread and occupancy falls from 53.983% to 14.208%.
Fair note: The capture supports a different tiling/resource balance but does not uniquely identify the selected M/N/K tile and stage count. Lower occupancy coexists with shorter measured duration.

**Flattened grid with grouped tile ordering.** Grouping up to eight M tiles changes traversal order and offers an operand-cache reuse heuristic without changing the matrix arithmetic.
Evidence: Two-dimensional program IDs are replaced by a single ID mapped through group_size = 8 * num_n and group_m. NCU memory utilization changes from 69.74% to 53.679%, while DRAM utilization rises from 14.9% to 22.131%.
Fair note: These utilization percentages do not establish cache-hit improvements or reduced transferred bytes. Scheduling and tile-size effects are not isolated.

**Divisible-shape unmasked operand loads.** Compile-time divisibility branches permit unconditional A/B loads on this case while retaining masked fallback paths.
Evidence: The new loop uses unmasked loads when M/K or N/K divide their tile dimensions. All supplied candidate tiles divide the corresponding 1024 dimensions.
Fair note: Bias loads and output stores remain masked in source. No instruction-level comparison establishes how much predicate work differed after compilation.

**Specialized strides and incremented pointers.** Compile-time input strides permit address simplification; advancing pointers expresses K traversal directly. Contiguous output indexing removes runtime output-stride arguments.
Evidence: Runtime input strides become STRIDE_* tl.constexpr parameters; A/B pointers advance by BLOCK_K times their K stride. Stores use C + rows * N + cols instead of output strides.
Fair note: The full diff shows specialization added, contrary to the supplied detector title saying it was removed. Generated instruction savings are not measured separately.

**Explicit conversions and wrapper adjustments.** The epilogue explicitly converts bias to fp32 and the result to the output dtype. The wrapper adds an empty-output return and changes validation placement.
Evidence: rounds_5 uses bias.to(tl.float32) and result.to(C.dtype.element_ty), returns early for M == 0 or N == 0, and removes the initial explicit A/B rank check. Both versions already restrict inputs to matching float16 or bfloat16 dtypes.
Fair note: This is not evidence of a new arithmetic precision policy or fewer conversions. The empty-shape path is irrelevant to this nonempty case.

What did not change: Both compute matrix multiplication plus a fused per-column bias.; Both accumulate the dot product in fp32.; Both retain masked bias loads and masked output stores.; Both allocate an M×N output with A's dtype and device.

- Both sides have usable NCU captures; individual mechanism attribution remains inferred because multiple edits changed together.
- NCU labels shift from memory-hierarchy limited to compute/tensor-pipeline limited, with low occupancy and register pressure on rounds_5; these are profile characterizations, not isolated causal findings.
- The supplied evidence does not report correctness-test results or autotuning overhead accounting.

**LLM takeaway:** Both measurement paths support a fixed-harness win for rounds_5 on this case. The profiles are consistent with a different reuse/resource balance, but they do not isolate any edit as the measured cause.

### Pair: `generated` → `rounds_10`

Sources: `generated` [final_kernel.py](../problems/matmul_bias/generated/matmul_bias/final_kernel.py) sha `5ba2fb340021…` · `rounds_10` [optimized_kernel.py](../problems/matmul_bias/case_0/rounds_10/matmul_bias/optimized_kernel.py) sha `30091c7800b1…` · full diff: [matmul_bias__generated_to_rounds_10.diff](source-diffs/matmul_bias__generated_to_rounds_10.diff)

#### Evidence

| Metric | from | to | Δ |
|---|---:|---:|---:|
| Fixed-harness kernel (ms) | 0.040 | 0.025 | -0.015 |
| Fixed-harness TFLOPS | 53.100 | 86.090 | +32.990 |
| PyTorch reference (ms) | 0.026 | 0.024 | -0.002 |
| NCU duration (µs) | 53.156 | 35.590 | -17.566 |
| NCU SM % | 38.158 | 57.547 | +19.4 |
| NCU memory % | 69.740 | 53.720 | -16.0 |
| NCU DRAM % | 14.900 | 21.779 | +6.9 |
| NCU occupancy % | 53.983 | 14.226 | -39.8 |
| Registers/thread | 53.000 | 150.000 | +97 |
| NCU TFLOPS | 40.400 | 60.339 | +19.939 |
| NCU grid / block | 512 / 128 | 128 / 128 | — |

#### 1. Mask-free fast path added
Source lines: from — → to 76-95

```diff
+    ks = tl.arange(0, BLOCK_K)
+
+    a_ptrs = A + rows[:, None] * STRIDE_AM + ks[None, :] * STRIDE_AK
+    b_ptrs = B + ks[:, None] * STRIDE_BK + cols[None, :] * STRIDE_BN
+
+    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
+
+    for block in range(tl.cdiv(K, BLOCK_K)):
+        if M % BLOCK_M == 0 and K % BLOCK_K == 0:
+            a = tl.load(a_ptrs)
+        else:
+            a = tl.load(
+                a_ptrs,
+                mask=(rows[:, None] < M)
+                & (block * BLOCK_K + ks[None, :] < K),
+                other=0.0,
+            )
+
+        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
+            b = tl.load(b_ptrs)
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 2. Autotune candidate space expanded or changed
Source lines: from — → to 6-42

```diff
 import triton.language as tl
 
 
+@triton.autotune(
+    configs=[
+        triton.Config(
+            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
+            num_warps=4, num_stages=3,
+        ),
+        triton.Config(
+            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64},
+            num_warps=4, num_stages=3,
+        ),
+        triton.Config(
+            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64},
... 13 changed lines omitted ...
+        ),
+        triton.Config(
+            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
+            num_warps=4, num_stages=3,
+        ),
+        triton.Config(
+            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
+            num_warps=8, num_stages=3,
+        ),
+    ],
+    key=["M", "N", "K"],
+)
 @triton.jit
 def _matmul_bias_kernel(
```

**Why it matters:** The candidate tile/warp/stage space changes. A different candidate set can select a better register/shared-memory/occupancy balance. The source alone does not prove which configuration ran; NCU grid/block/occupancy is the evidence.

#### 3. Grouped CTA order — added in `to`
Source lines: from 26-28 → to 61-72

```diff
     BLOCK_N: tl.constexpr,
     BLOCK_K: tl.constexpr,
 ):
-    """Compute one output tile of A @ B + bias, accumulating in fp32."""
-    pid_m = tl.program_id(0)
-    pid_n = tl.program_id(1)
+    pid = tl.program_id(0)
+    num_m = tl.cdiv(M, BLOCK_M)
+    num_n = tl.cdiv(N, BLOCK_N)
+
+    group_size: tl.constexpr = 8
+    group_width = group_size * num_n
+    group_id = pid // group_width
+    first_m = group_id * group_size
+    active_m = tl.minimum(num_m - first_m, group_size)
+    local_pid = pid % group_width
+    pid_m = first_m + local_pid % active_m
+    pid_n = local_pid // active_m
 
     rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
```

**Why it matters:** Grouped CTA ordering changes which output tiles run concurrently. This is an L2-reuse heuristic for the A/B panels, not an arithmetic change.

#### 4. Compile-time strides and index arithmetic — removed in `to`
Source lines: from 15-21 → to 52-56

**Repeated:** This change also appears in `generated` → `rounds_5`; excerpt omitted.

**Why it matters:** Runtime dimensions/strides become `tl.constexpr` and/or modulo wrapping is removed. Compile-time constants let Triton fold address arithmetic, remove integer divide/modulo from every element's index path, and prove alignment.

#### 5. Grid and program addressing — changed
Source lines: from 77-80 → to 135-141

**Repeated:** This change also appears in `generated` → `rounds_5`; excerpt omitted.

**Why it matters:** Program-ID mapping, launch geometry, or address construction changes. Flattening axes or replacing per-program metadata loads with direct offsets removes fixed integer/global-load work per CTA and changes how tiles map to L2.

#### 6. Dtype and math path — changed
Source lines: from 67-74 → to 125-132

```diff
     M, K = A.shape
     K_b, N = B.shape
-    if K != K_b or bias.numel() != N:
-        raise ValueError("Incompatible matrix or bias dimensions")
-    if A.device != B.device or A.device != bias.device:
-        raise ValueError("All inputs must be on the same device")
-    if A.dtype != B.dtype or A.dtype != bias.dtype:
-        raise ValueError("All inputs must have the same dtype")
-    if A.dtype not in (torch.float16, torch.bfloat16):
-        raise TypeError("Only float16 and bfloat16 inputs are supported")
+    if K != K_b or bias.ndim != 1 or bias.numel() != N:
+        raise ValueError("Incompatible input shapes")
+    if A.dtype not in (torch.float16, torch.bfloat16):
+        raise TypeError("Expected float16 or bfloat16 inputs")
+    if B.dtype != A.dtype or bias.dtype != A.dtype:
+        raise TypeError("Input dtypes must match")
+    if A.device != B.device or A.device != bias.device:
+        raise ValueError("Input devices must match")
 
     output = torch.empty((M, N), device=A.device, dtype=A.dtype)
```

**Why it matters:** Dtype conversion points or math functions change. fp32 accumulation and fewer conversions trade precision against conversion/instruction cost; `exp2` avoids a base conversion.

#### What did *not* change

- **fp32 accumulation preserved.** Both accumulate in fp32.
- **Masked output store preserved.** Both mask the output store.

**Takeaway:** Detected mask_elision, autotune_space, grouped_swizzle, constexpr_addressing, grid_addressing, dtype_math; fixed harness 0.040 → 0.025 ms; NCU 53.156 → 35.590 µs.

#### LLM commentary

For the 1024×1024×1024 float16 case, rounds_10 is faster in both measurements: benchmark latency is 0.0249 ms versus 0.0404 ms, and NCU kernel duration is 35.590 µs versus 53.156 µs. The revision introduces autotuning, grouped tile scheduling, compile-time strides, and divisibility-specialized input loads. These are plausible performance mechanisms, but the combined comparison does not isolate any individual change as the measured cause.

**Fixed tiling replaced by eight autotune candidates.** Tile dimensions, reduction depth, warp count, and pipeline stages trade operand reuse against register demand and occupancy.
Evidence: The baseline fixes BLOCK_M/N/K at 32/64/32 with four warps. rounds_10 searches eight configurations. NCU reports grid size 512→128 with 128 threads per block on both sides; the destination grid is consistent with 64×128 output tiles among the listed candidates. Registers per thread rise from 53 to 150, while achieved occupancy falls from 53.983% to 14.226%.
Fair note: The capture does not identify BLOCK_K or pipeline stages. Lower occupancy accompanies the faster measurement; it is not evidence of a benefit by itself.

**Divisibility-specialized input loads.** Compile-time guards permit unconditional A/B loads when tile dimensions divide the matrix dimensions, avoiding explicit source-level predicates on that path.
Evidence: rounds_10 guards unmasked loads with M % BLOCK_M, N % BLOCK_N, and K % BLOCK_K checks. All candidate tile dimensions divide 1024. The baseline explicitly masks both input loads.
Fair note: The baseline compiler may also simplify provably redundant predicates; no instruction-level comparison establishes the actual savings. Bias loads and output stores remain masked.

**Flattened grid with grouped tile ordering.** Grouping output tiles along M can improve reuse of operand panels in cache without changing the arithmetic.
Evidence: The two-dimensional program-ID mapping becomes a one-dimensional mapping with group_size = 8. NCU memory throughput utilization falls from 69.740% to 53.720%, while SM utilization rises from 38.158% to 57.547%.
Fair note: These aggregate counters do not establish improved cache reuse. Tile size, scheduling, and other changes were measured together.

**Compile-time strides and incremental pointers.** Specialized strides permit address folding, and pointer increments express progression through K without rebuilding full source-level addresses each iteration.
Evidence: Input and bias strides become tl.constexpr. A/B pointers are initialized before the loop and incremented by BLOCK_K times their reduction stride. Output addressing becomes rows * N + cols instead of using output strides.
Fair note: The full diff shows compile-time stride specialization added, contrary to the supplied detector title saying removed. Compiler output is unavailable, so instruction savings remain inferred.

**Explicit epilogue casts and wrapper adjustments.** Explicit casts clarify conversion points, while wrapper changes affect input handling rather than the matrix multiplication algorithm.
Evidence: rounds_10 explicitly converts bias to fp32 and the result to the output element type. It adds an empty-output return and removes the explicit A/B rank check before shape unpacking.
Fair note: Both versions retain fp32 accumulation and same-dtype output allocation. The diff does not establish a different effective numerical path or a timing contribution from these changes for this case.

What did not change: Both compute matrix multiplication with fused column-bias addition.; Both accumulate with tl.dot into fp32 accumulators.; Both retain masked bias loads and masked output stores.; Both allocate output with A's dtype and device.

- Both sides have usable NCU captures, and benchmark and NCU agree on the faster side, but their absolute timings and relative improvements differ.
- NCU labels the baseline memory-hierarchy limited and the destination compute limited with register pressure and low occupancy; these are profiling characterizations, not isolated causal findings.
- No per-change ablation, generated instruction comparison, or numerical-error results are provided.

**LLM takeaway:** The evidence supports a fixed-harness win for rounds_10 on this case, corroborated by NCU. Larger output tiles and the accompanying addressing, masking, and scheduling changes provide plausible mechanisms, but their individual contributions are not measured.

### Pair: `generated` → `rocm_blog_fusion`

Sources: `generated` [final_kernel.py](../problems/matmul_bias/generated/matmul_bias/final_kernel.py) sha `5ba2fb340021…` · `rocm_blog_fusion` [kernel.py](../problems/matmul_bias/baselines/rocm_blog_fusion/matmul_bias/kernel.py) sha `ef1f4165e29f…` · full diff: [matmul_bias__generated_to_rocm_blog_fusion.diff](source-diffs/matmul_bias__generated_to_rocm_blog_fusion.diff)

#### Evidence

| Metric | from | to | Δ |
|---|---:|---:|---:|
| Fixed-harness kernel (ms) | 0.040 | 0.043 | +0.003 |
| Fixed-harness TFLOPS | 53.100 | 50.100 | -3.000 |
| PyTorch reference (ms) | 0.026 | 0.182 | +0.156 |
| NCU duration (µs) | 53.156 | 34.956 | -18.200 |
| NCU SM % | 38.158 | 59.414 | +21.3 |
| NCU memory % | 69.740 | 38.108 | -31.6 |
| NCU DRAM % | 14.900 | 24.227 | +9.3 |
| NCU occupancy % | 53.983 | 8.317 | -45.7 |
| Registers/thread | 53.000 | 255.000 | +202 |
| NCU TFLOPS | 40.400 | 61.433 | +21.032 |
| NCU grid / block | 512 / 128 | 64 / 128 | — |

#### 1. Mask-free fast path removed
Source lines: from 29-49 → to —

```diff
-
-    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    offs_k = tl.arange(0, BLOCK_K)
-
-    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
-
-    for k_start in range(0, K, BLOCK_K):
-        k = k_start + offs_k
-
-        a = tl.load(
-            a_ptr + rows[:, None] * stride_am + k[None, :] * stride_ak,
-            mask=(rows[:, None] < M) & (k[None, :] < K),
-            other=0.0,
-        )
-        b = tl.load(
-            b_ptr + k[:, None] * stride_bk + cols[None, :] * stride_bn,
-            mask=(k[:, None] < K) & (cols[None, :] < N),
-            other=0.0,
-        )
-        acc = tl.dot(a, b, acc)
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary.

#### 2. Mask-free fast path removed
Source lines: from 50-57 → to —

```diff
-
-    # Fuse the bias addition into the matmul epilogue.
-    bias = tl.load(bias_ptr + cols * stride_bias, mask=cols < N, other=0.0)
-    result = acc + bias[None, :]
-
-    out_ptrs = out_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on
-    out_mask = (rows[:, None] < M) & (cols[None, :] < N)
-    tl.store(out_ptrs, result, mask=out_mask)
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary.

#### 3. Autotune candidate space expanded or changed
Source lines: from — → to 47-83

```diff
+def matmul_autotune_config(pre_hook=None):
+    return [
+        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8}, num_stages=s,
+            num_warps=w, pre_hook=pre_hook)
+        for BM in [128]
+        for BN in [128, 256]
+        for BK in [64, 128]
+        for s in ([3, 4, 5])
+        for w in [4, 8]
+    ]
+
+
... 13 changed lines omitted ...
+        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
+        GROUP_SIZE_M: tl.constexpr
+):
+    pid = tl.program_id(axis=0)
+    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
+    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
+    num_pid_in_group = GROUP_SIZE_M * num_pid_n
+    group_id = pid // num_pid_in_group
+    first_pid_m = group_id * GROUP_SIZE_M
+    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
+    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
+    pid_n = (pid % num_pid_in_group) // group_size_m
```

**Why it matters:** The candidate tile/warp/stage space changes. A different candidate set can select a better register/shared-memory/occupancy balance. The source alone does not prove which configuration ran; NCU grid/block/occupancy is the evidence.

#### 4. Grouped CTA order — removed in `to`
Source lines: from 6-28 → to —

```diff
-@triton.jit
-def _matmul_bias_kernel(
-    a_ptr,
-    b_ptr,
-    bias_ptr,
-    out_ptr,
-    M: tl.constexpr,
-    N: tl.constexpr,
-    K: tl.constexpr,
-    stride_am,
-    stride_ak,
-    stride_bk,
-    stride_bn,
-    stride_bias,
-    stride_om,
-    stride_on,
-    BLOCK_M: tl.constexpr,
-    BLOCK_N: tl.constexpr,
-    BLOCK_K: tl.constexpr,
-):
-    """Compute one output tile of A @ B + bias, accumulating in fp32."""
-    pid_m = tl.program_id(0)
-    pid_n = tl.program_id(1)
```

**Why it matters:** Grouped CTA ordering changes which output tiles run concurrently. This is an L2-reuse heuristic for the A/B panels, not an arithmetic change.

#### 5. Grid and program addressing — removed in `to`
Source lines: from 60-79 → to —

```diff
-def kernel_function(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
-    """Return A @ B + bias, fusing matrix multiplication and bias addition."""
-    if A.ndim != 2 or B.ndim != 2 or bias.ndim != 1:
-        raise ValueError("Expected A and B to be matrices and bias to be a vector")
-
-    M, K = A.shape
-    K_b, N = B.shape
-    if K != K_b or bias.numel() != N:
-        raise ValueError("Incompatible matrix or bias dimensions")
-    if A.device != B.device or A.device != bias.device:
-        raise ValueError("All inputs must be on the same device")
-    if A.dtype != B.dtype or A.dtype != bias.dtype:
-        raise ValueError("All inputs must have the same dtype")
-    if A.dtype not in (torch.float16, torch.bfloat16):
-        raise TypeError("Only float16 and bfloat16 inputs are supported")
-
-    output = torch.empty((M, N), device=A.device, dtype=A.dtype)
-
-    block_m, block_n, block_k = 32, 64, 32
-    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
```

**Why it matters:** Program-ID mapping, launch geometry, or address construction changes. Flattening axes or replacing per-program metadata loads with direct offsets removes fixed integer/global-load work per CTA and changes how tiles map to L2.

#### 6. Host wrapper and launch contract — removed in `to`
Source lines: from 80-100 → to —

```diff
-
-    _matmul_bias_kernel[grid](
-        A,
-        B,
-        bias,
-        output,
-        M,
-        N,
-        K,
-        A.stride(0),
-        A.stride(1),
-        B.stride(0),
-        B.stride(1),
-        bias.stride(0),
-        output.stride(0),
-        output.stride(1),
-        BLOCK_M=block_m,
-        BLOCK_N=block_n,
-        BLOCK_K=block_k,
-        num_warps=4,
-    )
```

**Why it matters:** Host-side wrapper, launch geometry, or contract checks change. This can alter grid shape and argument passing without changing device code; assertions and validation are not performance-relevant.

#### What did *not* change

- **fp32 accumulation preserved.** Both accumulate in fp32.
- **Masked output store preserved.** Both mask the output store.

#### Caveats

- The PyTorch reference times differ between the two records (0.026 ms vs 0.182 ms); the kernel ratio is fixed-harness evidence, not a same-session controlled A/B.
- The fixed harness and NCU disagree on direction. NCU locks clocks at or below boost, so its duration should be greater than or equal to the fixed-harness time; this pattern is most plausibly a different autotune configuration or a clock/power difference. Use NCU for code-level attribution and do not claim a fixed-harness win for this pair.
- NCU duration is below the fixed-harness time for `rocm_blog_fusion`; the benchmark and profiler likely selected different autotune configurations or ran at different clocks.
- `rocm_blog_fusion` is an archived external baseline (`matmul_bias/baselines/rocm_blog_fusion.py`). This is a source comparison, not a controlled one-change A/B.

**Takeaway:** Detected mask_elision, autotune_space, grouped_swizzle, grid_addressing, host_launch; fixed harness 0.040 → 0.043 ms; NCU 53.156 → 34.956 µs.

#### LLM commentary

For the 1024×1024×1024 float16 case, the diff replaces a fixed-tile fused matmul+bias kernel with an autotuned baseline using larger tiles and grouped program ordering. Bench reports generated at 0.0404 ms versus rocm_blog_fusion at 0.0429 ms, while NCU reports the opposite ordering: 53.156 µs versus 34.956 µs. These measurements disagree, so this pair does not establish a fixed-harness win or isolate a measured cause.

**Fixed small tiles replaced by larger autotuned candidates.** Larger tiles change panel reuse, accumulator size, parallelism, and register demand; stage and warp choices also change resource balance.
Evidence: Generated fixes BLOCK_M/N/K=32/64/32 with four warps. The baseline searches BM=128, BN=128 or 256, BK=64 or 128, stages=3/4/5, and warps=4/8. NCU records grids of 512 versus 64, registers/thread of 53 versus 255, and occupancy of 53.983% versus 8.317%. Both captures report block size 128.
Fair note: The baseline's profiled grid is consistent with 128×128 output tiles, but the exact BK and stage selection are not reported. NCU labels generated memory-bound and the baseline compute-bound; these are profile observations, not proof that a particular tuning choice explains timing.

**Grouped ordering added through a flattened launch.** Grouping output tiles changes their execution order and offers a panel-cache reuse heuristic without changing the intended matrix operation.
Evidence: Generated uses independent tl.program_id(0) and tl.program_id(1). The baseline uses one flattened program ID and computes grouped pid_m/pid_n with GROUP_SIZE_M=8.
Fair note: The full diff shows grouping added, contrary to the supplied detection title saying it was removed. No isolated measurement attributes a timing difference to grouping.

**Boundary addressing and K-loop pointer construction changed.** The baseline wraps row and column load offsets modulo M/N, masks only the K dimension for matrix loads, and advances pointers each iteration. Generated explicitly masks matrix row/column boundaries and constructs addresses from k_start.
Evidence: Generated loads use row/K and K/column masks. The baseline uses offs_am % M, offs_bn % N, K-tail masks, and pointer increments. Both retain masked output stores.
Fair note: All candidate tile dimensions divide this case's dimensions. The full diff does not show a dedicated mask-free fast path being removed, and source masks do not establish the final emitted predicates.

**Accumulator cast moved before bias addition.** Rounding the matrix product before adding bias can change numerical results relative to adding bias to the fp32 accumulator.
Evidence: Generated computes result = acc + bias before storing. The baseline first executes accumulator.to(c_ptr.dtype.element_ty), then adds bias.
Fair note: Both accumulate dot products in fp32, but their epilogue precision order is not identical. No numerical-error measurements are provided.

**Wrapper contract and bias addressing narrowed.** The baseline requires contiguous A and addresses bias as unit-stride, whereas generated passes the bias stride explicitly.
Evidence: Generated validates dimensions, devices, and dtypes and uses cols * stride_bias. The baseline asserts A contiguity and loads bias_ptr + offs_bn without a bias-stride argument.
Fair note: These are interface and supported-layout differences, not established timing explanations for this case.

What did not change: Both fuse matrix multiplication and column-bias addition into one device kernel.; Both use fp32 dot-product accumulation and masked output stores.; Both allocate an M×N output with the input dtype.

- Both sides have usable NCU captures, but bench and NCU disagree on timing direction.
- The baseline's NCU duration is below its bench time; different autotune selections or clock/power conditions are possible explanations, not established facts.
- Reference timings differ: 0.0262 ms versus 0.1819 ms, limiting same-session comparability.
- This archived external-baseline comparison changes multiple mechanisms and is not a controlled one-change A/B.

**LLM takeaway:** The evidence documents a substantial tiling, resource-usage, and scheduling shift, plus an epilogue precision difference. It does not identify a measured cause of improved performance or establish a fixed-harness winner.

### Pair: `rounds_5` → `rounds_10`

Sources: `rounds_5` [optimized_kernel.py](../problems/matmul_bias/case_0/rounds_5/matmul_bias/optimized_kernel.py) sha `e71fe54ed298…` · `rounds_10` [optimized_kernel.py](../problems/matmul_bias/case_0/rounds_10/matmul_bias/optimized_kernel.py) sha `30091c7800b1…` · full diff: [matmul_bias__rounds_5_to_rounds_10.diff](source-diffs/matmul_bias__rounds_5_to_rounds_10.diff)

#### Evidence

| Metric | from | to | Δ |
|---|---:|---:|---:|
| Fixed-harness kernel (ms) | 0.025 | 0.025 | -0.000 |
| Fixed-harness TFLOPS | 85.050 | 86.090 | +1.040 |
| PyTorch reference (ms) | 0.024 | 0.024 | +0.000 |
| NCU duration (µs) | 35.613 | 35.590 | -0.023 |
| NCU SM % | 57.470 | 57.547 | +0.1 |
| NCU memory % | 53.679 | 53.720 | +0.0 |
| NCU DRAM % | 22.131 | 21.779 | -0.4 |
| NCU occupancy % | 14.208 | 14.226 | +0.0 |
| Registers/thread | 150.000 | 150.000 | +0 |
| NCU TFLOPS | 60.301 | 60.339 | +0.038 |
| NCU grid / block | 128 / 128 | 128 / 128 | — |

#### 1. Masking or predication changed
Source lines: from 81-81 → to 91-91

```diff
                 a_ptrs,
                 mask=(rows[:, None] < M)
                 & (block * BLOCK_K + ks[None, :] < K),
-                other=0,
+                other=0.0,
             )
 
         if N % BLOCK_N == 0 and K % BLOCK_K == 0:
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 2. Masking or predication changed
Source lines: from 91-91 → to 101-101

```diff
                 b_ptrs,
                 mask=(block * BLOCK_K + ks[:, None] < K)
                 & (cols[None, :] < N),
-                other=0,
+                other=0.0,
             )
 
         acc = tl.dot(a, b, acc)
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 3. Masking or predication changed
Source lines: from 101-101 → to 111-111

```diff
     bias = tl.load(
         Bias + cols * STRIDE_BIAS,
         mask=cols < N,
-        other=0,
+        other=0.0,
     ).to(tl.float32)
     result = (acc + bias[None, :]).to(C.dtype.element_ty)
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 4. Autotune candidate space expanded or changed
Source lines: from 13-13 → to 13-17

```diff
             num_warps=4, num_stages=3,
         ),
         triton.Config(
-            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
+            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64},
+            num_warps=4, num_stages=3,
+        ),
+        triton.Config(
+            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64},
             num_warps=4, num_stages=4,
         ),
         triton.Config(
```

**Why it matters:** The candidate tile/warp/stage space changes. A different candidate set can select a better register/shared-memory/occupancy balance. The source alone does not prove which configuration ran; NCU grid/block/occupancy is the evidence.

#### 5. Grouped CTA order — changed
Source lines: from 56-62 → to 64-72

```diff
     pid = tl.program_id(0)
     num_m = tl.cdiv(M, BLOCK_M)
     num_n = tl.cdiv(N, BLOCK_N)
-    group_size = 8 * num_n
-    group_id = pid // group_size
-    first_m = group_id * 8
-    group_m = tl.minimum(num_m - first_m, 8)
-    local_pid = pid % group_size
-    pid_m = first_m + local_pid % group_m
-    pid_n = local_pid // group_m
+
+    group_size: tl.constexpr = 8
+    group_width = group_size * num_n
+    group_id = pid // group_width
+    first_m = group_id * group_size
+    active_m = tl.minimum(num_m - first_m, group_size)
+    local_pid = pid % group_width
+    pid_m = first_m + local_pid % active_m
+    pid_n = local_pid // active_m
 
     rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
```

**Why it matters:** Grouped CTA ordering changes which output tiles run concurrently. This is an L2-reuse heuristic for the A/B panels, not an arithmetic change.

#### 6. Dtype and math path — changed
Source lines: from 71-71 → to 81-81

```diff
     a_ptrs = A + rows[:, None] * STRIDE_AM + ks[None, :] * STRIDE_AK
     b_ptrs = B + ks[:, None] * STRIDE_BK + cols[None, :] * STRIDE_BN
 
-    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
+    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
 
     for block in range(tl.cdiv(K, BLOCK_K)):
         if M % BLOCK_M == 0 and K % BLOCK_K == 0:
```

**Why it matters:** Dtype conversion points or math functions change. fp32 accumulation and fewer conversions trade precision against conversion/instruction cost; `exp2` avoids a base conversion.

#### What did *not* change

- **Grouped CTA order preserved.** Both keep a grouped CTA order (`group_id`/`pid_m`) for L2 reuse.
- **fp32 accumulation preserved.** Both accumulate in fp32.
- **Masked output store preserved.** Both mask the output store.
- **Autotune key preserved.** Both autotune on the same key list.
- **Kernel entry signature preserved.** Both expose the same `kernel_function` signature.

**Takeaway:** Detected mask_elision, autotune_space, grouped_swizzle, dtype_math; fixed harness 0.025 → 0.025 ms; NCU 35.613 → 35.590 µs.

#### LLM commentary

For the 1024×1024×1024 float16 matmul_bias case, rounds_10 revises the autotune candidates while preserving the kernel’s arithmetic and grouped tile mapping. Bench latency decreases from 0.0253 to 0.0249 ms, while NCU duration decreases from 35.613 to 35.590 µs; both favor rounds_10, but both comparisons are marked non-material. The evidence supports near-equivalent measured performance, not attribution of the small difference to a particular edit.

**Candidate tiles and pipeline settings revised.** Different tile sizes, warp counts, and stage counts offer different register, shared-memory, and parallelism tradeoffs. This is the main substantive tuning mechanism in the diff.
Evidence: The list grows from six to seven candidates, introduces BLOCK_M=32 configurations, removes BLOCK_M=128 configurations, changes the 64×128×32 candidate from four to three stages, and adds an eight-warp 64×128×64 candidate. Both NCU captures report grid=128, block=128, and 150 registers/thread.
Fair note: The selected configurations are not explicitly reported. Matching launch dimensions and register counts do not establish identical selected configurations or generated code, and do not isolate a measured cause.

**Grouped indexing refactored without changing the mapping.** Grouped scheduling is an L2-reuse heuristic, but this edit preserves the same eight-row grouping rather than introducing a new scheduling mechanism.
Evidence: The old group_size=8*num_n becomes group_width=group_size*num_n with group_size declared tl.constexpr=8; group_m is renamed active_m. The pid_m and pid_n formulas remain algebraically equivalent.
Fair note: The automatic change label overstates the semantic difference: the diff does not demonstrate a changed CTA order or improved cache reuse.

**Zero literals and accumulator syntax normalized.** Changing masked-load fallback literals from 0 to 0.0 does not remove masks. Passing tl.float32 positionally instead of by keyword preserves the accumulator dtype.
Evidence: A, B, and bias loads change only other=0 to other=0.0; acc remains tl.zeros with tl.float32. Every listed tile divides the relevant 1024 dimensions, so the existing A/B divisibility guards permit unmasked paths on both sides.
Fair note: There is no demonstrated new mask-elision mechanism or arithmetic-precision change. The bias mask and output-store mask remain present in source.

**Validation messages and dtype-mismatch exception changed.** This changes invalid-input behavior, not the valid case’s matrix computation.
Evidence: Mismatched input dtypes now raise TypeError instead of ValueError; shape, supported-dtype, and device error messages are rewritten.
Fair note: The provided float16 case does not exercise these error paths.

What did not change: Both compute matrix multiplication with bias using fp32 accumulation and cast the result to the output element type.; Both preserve eight-row grouped CTA ordering, the autotune key list, and the kernel_function signature.; Both retain divisibility-guarded A/B loads and a masked output store.; Both usable NCU captures report one kernel, 128-thread blocks, 150 registers/thread, and a compute-bound classification.

- Bench and NCU agree on direction but differ substantially in the size of the observed difference; neither comparison is marked material.
- No timing variability or statistical confidence estimates are supplied.
- NCU occupancy is nearly unchanged at 14.208% versus 14.226%, and SM throughput is 57.470% versus 57.547%; the reported limiter remains tensor pipelines, low occupancy, and register pressure.
- Both sides have usable NCU captures, but no controlled isolation of individual edits or generated-code comparison is provided.

**LLM takeaway:** The meaningful tuning change is the revised autotune search space; the swizzle, masking, and accumulation edits are largely equivalent refactoring for this case. Measurements show only a non-material advantage for rounds_10, with no established causal attribution.

### Pair: `rounds_5` → `rocm_blog_fusion`

Sources: `rounds_5` [optimized_kernel.py](../problems/matmul_bias/case_0/rounds_5/matmul_bias/optimized_kernel.py) sha `e71fe54ed298…` · `rocm_blog_fusion` [kernel.py](../problems/matmul_bias/baselines/rocm_blog_fusion/matmul_bias/kernel.py) sha `ef1f4165e29f…` · full diff: [matmul_bias__rounds_5_to_rocm_blog_fusion.diff](source-diffs/matmul_bias__rounds_5_to_rocm_blog_fusion.diff)

#### Evidence

| Metric | from | to | Δ |
|---|---:|---:|---:|
| Fixed-harness kernel (ms) | 0.025 | 0.043 | +0.018 |
| Fixed-harness TFLOPS | 85.050 | 50.100 | -34.950 |
| PyTorch reference (ms) | 0.024 | 0.182 | +0.158 |
| NCU duration (µs) | 35.613 | 34.956 | -0.657 |
| NCU SM % | 57.470 | 59.414 | +1.9 |
| NCU memory % | 53.679 | 38.108 | -15.6 |
| NCU DRAM % | 22.131 | 24.227 | +2.1 |
| NCU occupancy % | 14.208 | 8.317 | -5.9 |
| Registers/thread | 150.000 | 255.000 | +105 |
| NCU TFLOPS | 60.301 | 61.433 | +1.132 |
| NCU grid / block | 128 / 128 | 64 / 128 | — |

#### 1. Mask-free fast path added
Source lines: from — → to 84-107

```diff
+
+    tl.assume(pid_m >= 0)
+    tl.assume(pid_n >= 0)
+    tl.assume(stride_am > 0)
+    tl.assume(stride_ak > 0)
+    tl.assume(stride_bn > 0)
+    tl.assume(stride_bk > 0)
+    tl.assume(stride_cm > 0)
+    tl.assume(stride_cn > 0)
+
+    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
+    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
+    offs_k = tl.arange(0, BLOCK_SIZE_K)
+    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
+    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
+
+    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
+    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
+        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
+        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
+        accumulator = tl.dot(a, b, accumulator)
+        a_ptrs += BLOCK_SIZE_K * stride_ak
+        b_ptrs += BLOCK_SIZE_K * stride_bk
+    c = accumulator.to(c_ptr.dtype.element_ty)
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 2. Mask-free fast path added
Source lines: from — → to 108-135

```diff
+
+    bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
+    output = c + bias[None, :]
+
+    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
+    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
+    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
+    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
+    tl.store(c_ptrs, output, mask=c_mask)
+
+
+# MatMulBias kernel wrapper function (the blog's fusion_matmulbias,
... 4 changed lines omitted ...
+    M, K = a.shape
+    K, N = b.shape
+    assert bias.shape[0] == N, "BIAS has incompatible dimensions"
+    o = torch.empty((M, N), device=a.device, dtype=a.dtype)
+    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
+    matmulbias_kernel[grid](
+        a, b, o,
+        bias,
+        M, N, K,
+        a.stride(0), a.stride(1),
+        b.stride(0), b.stride(1),
+        o.stride(0), o.stride(1),
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 3. Mask-free fast path removed
Source lines: from 63-82 → to —

```diff
-
-    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    ks = tl.arange(0, BLOCK_K)
-
-    a_ptrs = A + rows[:, None] * STRIDE_AM + ks[None, :] * STRIDE_AK
-    b_ptrs = B + ks[:, None] * STRIDE_BK + cols[None, :] * STRIDE_BN
-
-    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
-
-    for block in range(tl.cdiv(K, BLOCK_K)):
-        if M % BLOCK_M == 0 and K % BLOCK_K == 0:
-            a = tl.load(a_ptrs)
-        else:
-            a = tl.load(
-                a_ptrs,
-                mask=(rows[:, None] < M)
-                & (block * BLOCK_K + ks[None, :] < K),
-                other=0,
-            )
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 4. Autotune candidate space expanded or changed
Source lines: from — → to 32-51

```diff
+def _matmul_launch_metadata(grid, kernel, args):
+    ret = {}
+    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
+    BM, BN, BK = args["BLOCK_SIZE_M"], args["BLOCK_SIZE_N"], args["BLOCK_SIZE_K"]
+    ws_str = "_ws" if WS else ""
+    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}] [BM={BM}, BN={BN} BK={BK}]"
+    if "output_ptr" in args:
+        bytes_per_elem = args["output_ptr"].element_size()
+    else:
+        bytes_per_elem = 2
+    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
+    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
+    return ret
+
+
+def matmul_autotune_config(pre_hook=None):
+    return [
+        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8}, num_stages=s,
+            num_warps=w, pre_hook=pre_hook)
+        for BM in [128]
```

**Why it matters:** The candidate tile/warp/stage space changes. A different candidate set can select a better register/shared-memory/occupancy balance. The source alone does not prove which configuration ran; NCU grid/block/occupancy is the evidence.

#### 5. Grouped CTA order — removed in `to`
Source lines: from 35-62 → to —

```diff
-@triton.jit
-def _matmul_bias_kernel(
-    A,
-    B,
-    Bias,
-    C,
-    M: tl.constexpr,
-    N: tl.constexpr,
-    K: tl.constexpr,
-    STRIDE_AM: tl.constexpr,
-    STRIDE_AK: tl.constexpr,
-    STRIDE_BK: tl.constexpr,
... 4 changed lines omitted ...
-    BLOCK_K: tl.constexpr,
-):
-    pid = tl.program_id(0)
-    num_m = tl.cdiv(M, BLOCK_M)
-    num_n = tl.cdiv(N, BLOCK_N)
-    group_size = 8 * num_n
-    group_id = pid // group_size
-    first_m = group_id * 8
-    group_m = tl.minimum(num_m - first_m, 8)
-    local_pid = pid % group_size
-    pid_m = first_m + local_pid % group_m
-    pid_n = local_pid // group_m
```

**Why it matters:** Grouped CTA ordering changes which output tiles run concurrently. This is an L2-reuse heuristic for the A/B panels, not an arithmetic change.

#### 6. Grid and program addressing — removed in `to`
Source lines: from 110-145 → to 138-138

```diff
     )
 
-
-def kernel_function(A, B, bias):
-    M, K = A.shape
-    K_b, N = B.shape
-    if K != K_b or bias.ndim != 1 or bias.numel() != N:
-        raise ValueError("Incompatible matrix or bias dimensions")
-    if A.dtype not in (torch.float16, torch.bfloat16):
-        raise TypeError("Only float16 and bfloat16 inputs are supported")
-    if A.dtype != B.dtype or A.dtype != bias.dtype:
-        raise ValueError("All inputs must have the same dtype")
-    if A.device != B.device or A.device != bias.device:
-        raise ValueError("All inputs must be on the same device")
... 13 changed lines omitted ...
-        output,
-        M,
-        N,
-        K,
-        A.stride(0),
-        A.stride(1),
-        B.stride(0),
-        B.stride(1),
-        bias.stride(0),
-    )
-    return output
+    return o
```

**Why it matters:** Program-ID mapping, launch geometry, or address construction changes. Flattening axes or replacing per-program metadata loads with direct offsets removes fixed integer/global-load work per CTA and changes how tiles map to L2.

#### What did *not* change

- **Grouped CTA order preserved.** Both keep a grouped CTA order (`group_id`/`pid_m`) for L2 reuse.
- **fp32 accumulation preserved.** Both accumulate in fp32.
- **Masked output store preserved.** Both mask the output store.

#### Caveats

- The PyTorch reference times differ between the two records (0.024 ms vs 0.182 ms); the kernel ratio is fixed-harness evidence, not a same-session controlled A/B.
- NCU duration is below the fixed-harness time for `rocm_blog_fusion`; the benchmark and profiler likely selected different autotune configurations or ran at different clocks.
- `rocm_blog_fusion` is an archived external baseline (`matmul_bias/baselines/rocm_blog_fusion.py`). This is a source comparison, not a controlled one-change A/B.

**Takeaway:** Detected mask_elision, autotune_space, grouped_swizzle, grid_addressing; fixed harness 0.025 → 0.043 ms; NCU 35.613 → 34.956 µs.

#### LLM commentary

For the 1024×1024×1024 float16 case, rounds_5 → rocm_blog_fusion changes the autotune space, dimension specialization, boundary addressing, and bias epilogue while retaining fused matmul+bias. Bench reports 0.0253 → 0.0429 ms, favoring rounds_5, whereas NCU reports 35.613 → 34.956 µs, slightly favoring rocm_blog_fusion with a non-material difference. These measurements disagree, so this pair does not establish a fixed-harness win or identify a change responsible for the timing difference.

**Larger tile candidates and different resource balance.** Tile sizes, pipeline stages, and warp counts alter reuse, accumulator storage, and scheduling opportunities. These are mechanisms, not isolated explanations of performance.
Evidence: The source moves from six configurations to combinations with BM=128, BN in [128,256], BK in [64,128], stages in [3,4,5], and warps in [4,8]. NCU records grid 128 → 64, block size 128 on both sides, registers/thread 150 → 255, and occupancy 14.208% → 8.317%.
Fair note: The captures show different resource footprints, but do not establish the exact configuration used by bench or attribute timing to a particular tuning parameter.

**Explicit compile-time fast paths replaced by runtime boundary expressions.** Compile-time dimensions permit explicit elimination of boundary predicates. Runtime dimensions, modulo addressing, and K-tail predicates expose different optimization opportunities and integer work.
Evidence: rounds_5 declares M/N/K and strides tl.constexpr and conditionally uses unmasked A/B loads when dimensions divide tiles. rocm_blog_fusion removes those constexpr annotations, wraps row/column offsets with % M and % N, adds positive-stride assumptions, and expresses K masks on every loop iteration. All candidate tile dimensions in the full diff divide this case.
Fair note: The full diff supports removal of explicit mask-free branches, not the generated 'fast path added' labels. No instruction-level evidence establishes which predicates or modulo operations survive compilation.

**Accumulator cast moves before bias addition.** Rounding the dot-product result before adding bias changes numerical evaluation order and the precision available to the epilogue.
Evidence: rounds_5 converts bias to fp32, adds it to the fp32 accumulator, then casts the result. rocm_blog_fusion first casts the accumulator to the output element type and then adds bias.
Fair note: Both use fp32 dot accumulation, but their epilogues are not numerically identical. No numerical-error results or isolated epilogue timing are supplied.

**Bias stride support and wrapper checks change.** These changes affect supported inputs and host-side validation rather than establishing a device-kernel performance explanation.
Evidence: rounds_5 passes bias.stride(0), checks dtype/device compatibility, and handles empty outputs. rocm_blog_fusion addresses bias with unit offsets, requires contiguous A, and passes output strides instead of constructing contiguous output addresses with N.
Fair note: No evidence indicates that these interface differences were exercised by this nonempty case or explains their timing contribution.

What did not change: Both fuse matmul and column-bias addition into one kernel; each NCU capture contains one kernel.; Both use fp32 tl.dot accumulation and masked output stores.; Both retain grouped CTA ordering with group size 8 and a flattened one-dimensional launch; the generated grouped-swizzle removal label contradicts the full diff.

- Both sides have usable NCU captures, but neither capture isolates individual source changes.
- Reference timings differ: 0.0241 versus 0.1819 ms. This is not a controlled same-session, one-change A/B.
- Different autotune selections or clocks are possible explanations for bench/NCU disagreement, not established facts.
- NCU classifies both kernels as compute-bound and flags low occupancy and register pressure; these observations do not prove a measured cause.

**LLM takeaway:** The strongest evidence is a changed tuning/resource profile and different specialization and epilogue semantics, with grouped scheduling preserved. Opposing bench and NCU rankings prevent a reliable performance-win claim or causal attribution.

### Pair: `rounds_10` → `rocm_blog_fusion`

Sources: `rounds_10` [optimized_kernel.py](../problems/matmul_bias/case_0/rounds_10/matmul_bias/optimized_kernel.py) sha `30091c7800b1…` · `rocm_blog_fusion` [kernel.py](../problems/matmul_bias/baselines/rocm_blog_fusion/matmul_bias/kernel.py) sha `ef1f4165e29f…` · full diff: [matmul_bias__rounds_10_to_rocm_blog_fusion.diff](source-diffs/matmul_bias__rounds_10_to_rocm_blog_fusion.diff)

#### Evidence

| Metric | from | to | Δ |
|---|---:|---:|---:|
| Fixed-harness kernel (ms) | 0.025 | 0.043 | +0.018 |
| Fixed-harness TFLOPS | 86.090 | 50.100 | -35.990 |
| PyTorch reference (ms) | 0.024 | 0.182 | +0.158 |
| NCU duration (µs) | 35.590 | 34.956 | -0.634 |
| NCU SM % | 57.547 | 59.414 | +1.9 |
| NCU memory % | 53.720 | 38.108 | -15.6 |
| NCU DRAM % | 21.779 | 24.227 | +2.4 |
| NCU occupancy % | 14.226 | 8.317 | -5.9 |
| Registers/thread | 150.000 | 255.000 | +105 |
| NCU TFLOPS | 60.339 | 61.433 | +1.094 |
| NCU grid / block | 128 / 128 | 64 / 128 | — |

#### 1. Mask-free fast path added
Source lines: from — → to 84-107

```diff
+
+    tl.assume(pid_m >= 0)
+    tl.assume(pid_n >= 0)
+    tl.assume(stride_am > 0)
+    tl.assume(stride_ak > 0)
+    tl.assume(stride_bn > 0)
+    tl.assume(stride_bk > 0)
+    tl.assume(stride_cm > 0)
+    tl.assume(stride_cn > 0)
+
+    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
+    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
+    offs_k = tl.arange(0, BLOCK_SIZE_K)
+    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
+    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
+
+    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
+    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
+        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
+        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
+        accumulator = tl.dot(a, b, accumulator)
+        a_ptrs += BLOCK_SIZE_K * stride_ak
+        b_ptrs += BLOCK_SIZE_K * stride_bk
+    c = accumulator.to(c_ptr.dtype.element_ty)
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 2. Mask-free fast path added
Source lines: from — → to 108-135

```diff
+
+    bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
+    output = c + bias[None, :]
+
+    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
+    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
+    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
+    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
+    tl.store(c_ptrs, output, mask=c_mask)
+
+
+# MatMulBias kernel wrapper function (the blog's fusion_matmulbias,
... 4 changed lines omitted ...
+    M, K = a.shape
+    K, N = b.shape
+    assert bias.shape[0] == N, "BIAS has incompatible dimensions"
+    o = torch.empty((M, N), device=a.device, dtype=a.dtype)
+    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
+    matmulbias_kernel[grid](
+        a, b, o,
+        bias,
+        M, N, K,
+        a.stride(0), a.stride(1),
+        b.stride(0), b.stride(1),
+        o.stride(0), o.stride(1),
```

**Why it matters:** Predicated loads/stores or tensor selects are added or removed. When the case dimensions divide the tile, a `% == 0` guard can resolve at compile time and the hot loop issues unconditional accesses; otherwise the masked path remains necessary. For this case, every detected candidate tile divides the guarded dimension, so the guard can resolve at compile time.

#### 3. Autotune candidate space expanded or changed
Source lines: from — → to 32-51

```diff
+def _matmul_launch_metadata(grid, kernel, args):
+    ret = {}
+    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
+    BM, BN, BK = args["BLOCK_SIZE_M"], args["BLOCK_SIZE_N"], args["BLOCK_SIZE_K"]
+    ws_str = "_ws" if WS else ""
+    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}] [BM={BM}, BN={BN} BK={BK}]"
+    if "output_ptr" in args:
+        bytes_per_elem = args["output_ptr"].element_size()
+    else:
+        bytes_per_elem = 2
+    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
+    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
+    return ret
+
+
+def matmul_autotune_config(pre_hook=None):
+    return [
+        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8}, num_stages=s,
+            num_warps=w, pre_hook=pre_hook)
+        for BM in [128]
```

**Why it matters:** The candidate tile/warp/stage space changes. A different candidate set can select a better register/shared-memory/occupancy balance. The source alone does not prove which configuration ran; NCU grid/block/occupancy is the evidence.

#### 4. Grouped CTA order — removed in `to`
Source lines: from 64-83 → to —

```diff
-
-    group_size: tl.constexpr = 8
-    group_width = group_size * num_n
-    group_id = pid // group_width
-    first_m = group_id * group_size
-    active_m = tl.minimum(num_m - first_m, group_size)
-    local_pid = pid % group_width
-    pid_m = first_m + local_pid % active_m
-    pid_n = local_pid // active_m
-
-    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    ks = tl.arange(0, BLOCK_K)
-
-    a_ptrs = A + rows[:, None] * STRIDE_AM + ks[None, :] * STRIDE_AK
-    b_ptrs = B + ks[:, None] * STRIDE_BK + cols[None, :] * STRIDE_BN
-
-    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
-
-    for block in range(tl.cdiv(K, BLOCK_K)):
```

**Why it matters:** Grouped CTA ordering changes which output tiles run concurrently. This is an L2-reuse heuristic for the A/B panels, not an arithmetic change.

#### 5. Compile-time strides and index arithmetic — removed in `to`
Source lines: from 43-63 → to —

```diff
-@triton.jit
-def _matmul_bias_kernel(
-    A,
-    B,
-    Bias,
-    C,
-    M: tl.constexpr,
-    N: tl.constexpr,
-    K: tl.constexpr,
-    STRIDE_AM: tl.constexpr,
-    STRIDE_AK: tl.constexpr,
-    STRIDE_BK: tl.constexpr,
-    STRIDE_BN: tl.constexpr,
-    STRIDE_BIAS: tl.constexpr,
-    BLOCK_M: tl.constexpr,
-    BLOCK_N: tl.constexpr,
-    BLOCK_K: tl.constexpr,
-):
-    pid = tl.program_id(0)
-    num_m = tl.cdiv(M, BLOCK_M)
-    num_n = tl.cdiv(N, BLOCK_N)
```

**Why it matters:** Runtime dimensions/strides become `tl.constexpr` and/or modulo wrapping is removed. Compile-time constants let Triton fold address arithmetic, remove integer divide/modulo from every element's index path, and prove alignment.

#### 6. Grid and program addressing — removed in `to`
Source lines: from 121-156 → to 138-138

```diff
     )
 
-
-def kernel_function(A, B, bias):
-    M, K = A.shape
-    K_b, N = B.shape
-    if K != K_b or bias.ndim != 1 or bias.numel() != N:
-        raise ValueError("Incompatible input shapes")
-    if A.dtype not in (torch.float16, torch.bfloat16):
-        raise TypeError("Expected float16 or bfloat16 inputs")
-    if B.dtype != A.dtype or bias.dtype != A.dtype:
-        raise TypeError("Input dtypes must match")
-    if A.device != B.device or A.device != bias.device:
-        raise ValueError("Input devices must match")
... 13 changed lines omitted ...
-        output,
-        M,
-        N,
-        K,
-        A.stride(0),
-        A.stride(1),
-        B.stride(0),
-        B.stride(1),
-        bias.stride(0),
-    )
-    return output
+    return o
```

**Why it matters:** Program-ID mapping, launch geometry, or address construction changes. Flattening axes or replacing per-program metadata loads with direct offsets removes fixed integer/global-load work per CTA and changes how tiles map to L2.

#### What did *not* change

- **Grouped CTA order preserved.** Both keep a grouped CTA order (`group_id`/`pid_m`) for L2 reuse.
- **fp32 accumulation preserved.** Both accumulate in fp32.
- **Masked output store preserved.** Both mask the output store.

#### Caveats

- The PyTorch reference times differ between the two records (0.024 ms vs 0.182 ms); the kernel ratio is fixed-harness evidence, not a same-session controlled A/B.
- NCU duration is below the fixed-harness time for `rocm_blog_fusion`; the benchmark and profiler likely selected different autotune configurations or ran at different clocks.
- `rocm_blog_fusion` is an archived external baseline (`matmul_bias/baselines/rocm_blog_fusion.py`). This is a source comparison, not a controlled one-change A/B.

**Takeaway:** Detected mask_elision, autotune_space, grouped_swizzle, constexpr_addressing, grid_addressing; fixed harness 0.025 → 0.043 ms; NCU 35.590 → 34.956 µs.

#### LLM commentary

For the 1024×1024×1024 float16 case, moving from rounds_10 to rocm_blog_fusion changes autotuning, specialization, boundary addressing, and bias-add precision while retaining fused matmul+bias. Bench reports 0.0249 ms versus 0.0429 ms, favoring rounds_10, but NCU reports 35.590 µs versus 34.956 µs, slightly favoring rocm_blog_fusion and marking the difference non-material. These measurements disagree, so this pair does not establish a fixed-harness win or identify a measured cause.

**Larger candidate tiles and broader launch tuning.** Tile size, warp count, and pipeline stages trade reuse against register demand and available parallelism.
Evidence: rounds_10 offers M tiles of 32/64 and N tiles of 64/128; rocm_blog_fusion offers M=128, N=128/256, K=64/128, stages 3/4/5, and warps 4/8. NCU shows grid 128→64, registers/thread 150→255, and occupancy 14.226%→8.317%, with block size 128 on both sides.
Fair note: The profiler documents different resource footprints, not their causal contribution to timing. Exact selected configurations are not supplied.

**Explicit specialization replaced by runtime arguments and assumptions.** Compile-time dimensions and strides permit address simplification; runtime assumptions offer different optimization information.
Evidence: rounds_10 declares M/N/K and input/bias strides as tl.constexpr. rocm_blog_fusion removes those annotations, adds positive-stride assumptions, wraps input row/column indices modulo M/N, and accepts output strides.
Fair note: Source signatures do not establish the generated instruction count or whether compiler specialization eliminates particular operations.

**Explicit mask-free load branches removed.** Boundary handling changes from compile-time divisibility branches to wrapped M/N addressing and K-masked loads.
Evidence: rounds_10 explicitly uses unconditional A/B loads when dimensions divide their tiles. rocm_blog_fusion always expresses K masks. Every tile dimension shown in the full candidate sets divides 1024; both sources retain masked output stores.
Fair note: The supplied 'mask-free fast path added' label reverses the full diff's direction. Source-level masks may still be optimized away; no instruction-level evidence is provided.

**Accumulator narrowed before bias addition.** Moving the cast changes rounding order and can change numerical results despite preserving fp32 dot accumulation.
Evidence: rounds_10 converts bias to fp32, adds it to the fp32 accumulator, then casts the result. rocm_blog_fusion casts the accumulator to the output dtype before adding bias.
Fair note: No numerical-error comparison or timing attribution isolating this change is supplied.

**Input validation and layout support changed.** Host checks affect accepted inputs, and bias addressing determines support for strided bias tensors.
Evidence: rocm_blog_fusion asserts contiguous A and uses bias_ptr + offs_bn. rounds_10 passes bias.stride(0), checks matching dtypes/devices, and handles empty M/N explicitly.
Fair note: The provided case does not establish that these contract differences affect its execution or explain benchmark timing.

What did not change: Both fuse matmul and column-broadcast bias into one kernel; NCU captures one kernel per side.; Both use fp32 dot-product accumulation and masked output stores.; Both preserve grouped CTA ordering with group size 8 and a flattened one-dimensional launch grid; grouped swizzling was not removed.

- Both sides have usable NCU captures, but bench and NCU favor opposite sides.
- Reference times differ: 0.0243 ms versus 0.1819 ms. These records are not a same-session controlled A/B.
- Different autotune selections or clocks are possible explanations for cross-tool timing differences, not established facts.
- This archived external baseline comparison changes multiple mechanisms simultaneously; neither timing record isolates an individual change.

**LLM takeaway:** The strongest observed distinction is rocm_blog_fusion's higher register usage and lower occupancy, alongside changed addressing and rounding semantics. Conflicting timing evidence prevents a fixed-harness winner claim or causal speedup attribution.
