"""Fused matmul + bias Triton kernel from the ROCm blog, as a baseline.

Adapted from the ROCm blog's matmulbias kernel (the fusion path of
workspace/ROCm/triton/matmulbias.py) to the matmul_bias problem contract:

- the wrapper is renamed to ``kernel_function`` (the top-level entry point
  bench.py / test_fixed.py / profile-kernel.py import), with the same
  ``(a, b, bias)`` argument order;
- the output dtype follows the inputs (float16 or bfloat16) instead of the
  blog's hardcoded float16: the wrapper allocates with ``dtype=a.dtype`` and
  the kernel casts the accumulator to the output element type, so the
  bfloat16 benchmark cases pass the contract check.

Everything else — the kernel body, the autotune config space, the grid
lambda, and the launch metadata (which feeds proton's flops/bytes metrics)
— is unchanged from the blog source, so performance comparisons against
KernelAgent-generated kernels stay honest. Only the fusion path is kept:
the blog's nonfusion variant (separate matmul and bias kernels) is not
part of this baseline.

KernelAgent contract satisfied here:
    def kernel_function(A, B, bias)  ->  (A @ B) + bias, shape (M, N),
    same dtype as A, computed inside @triton.jit kernels with fp32
    accumulation (tl.dot) and masked boundary loads/stores.
"""
import torch

import triton
import triton.language as tl


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    BM, BN, BK = args["BLOCK_SIZE_M"], args["BLOCK_SIZE_N"], args["BLOCK_SIZE_K"]
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}] [BM={BM}, BN={BN} BK={BK}]"
    if "output_ptr" in args:
        bytes_per_elem = args["output_ptr"].element_size()
    else:
        bytes_per_elem = 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


def matmul_autotune_config(pre_hook=None):
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8}, num_stages=s,
            num_warps=w, pre_hook=pre_hook)
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in ([3, 4, 5])
        for w in [4, 8]
    ]


# MatMulBias Fusion kernel
@triton.autotune(
    configs=matmul_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmulbias_kernel(
        a_ptr, b_ptr, c_ptr,
        bias_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(c_ptr.dtype.element_ty)

    bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
    output = c + bias[None, :]

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, output, mask=c_mask)


# MatMulBias kernel wrapper function (the blog's fusion_matmulbias,
# renamed to the KernelAgent contract's entry point)
def kernel_function(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    assert bias.shape[0] == N, "BIAS has incompatible dimensions"
    o = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmulbias_kernel[grid](
        a, b, o,
        bias,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        o.stride(0), o.stride(1),
    )

    return o
