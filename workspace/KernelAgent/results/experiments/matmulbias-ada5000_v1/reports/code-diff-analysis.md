# Code diff analysis: `matmulbias-ada5000_v1`

Generated 2026-10-28T17:19:40.569925 by the offline report.

## How to read this

- `-`/`+` in each heading name the `from`/`to` variants; they do not
  mean worse/better.
- Excerpts omit unchanged or repeated lines; code text is verbatim.
- A **Why it matters** paragraph explains a detected mechanism. It is not proof that the mechanism caused a measured difference.
- Benchmark and NCU numbers come from different methodologies and are reported separately. "inferred, not measured on both sides" marks a claim where one side lacks a usable capture.

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
