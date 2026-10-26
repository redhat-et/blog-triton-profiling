"""Fused Triton matrix multiplication with bias addition.

Computes C = A @ B + bias in one kernel: the matrix product accumulates in
FP32, bias is added in FP32, and the result is cast to the input dtype on
store.
"""

import torch
import triton
import triton.language as tl


def matmul_autotune_config(pre_hook=None):
    return [
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK}, num_stages=s,
                      num_warps=w, pre_hook=pre_hook)
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in ([3, 4, 5])
        for w in [4, 8]
    ]


@triton.autotune(
    configs=matmul_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_bias_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tiled matrix multiplication. The K mask handles non-tile-aligned
    # dimensions; the M and N masks also protect boundary tiles.
    for ki in range(tl.cdiv(K, BLOCK_K)):
        k = ki * BLOCK_K + offs_k

        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    # Fused bias epilogue: add bias before writing the output.
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=out_mask)


def kernel_function(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Return the fused matrix multiplication and bias result.

    A has shape [M, K], B has shape [K, N], and bias has shape [N].
    """
    if A.ndim != 2 or B.ndim != 2 or bias.ndim != 1:
        raise ValueError("Expected A and B to be 2D tensors and bias to be 1D")
    if A.shape[1] != B.shape[0] or bias.shape[0] != B.shape[1]:
        raise ValueError("Incompatible matrix or bias dimensions")
    if A.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("A must have dtype float16 or bfloat16")
    if B.dtype != A.dtype or bias.dtype != A.dtype:
        raise TypeError("A, B, and bias must have the same dtype")
    if A.device != B.device or A.device != bias.device or A.device.type != "cuda":
        raise ValueError("A, B, and bias must be on the same CUDA device")

    M, K = A.shape
    N = B.shape[1]
    output = torch.empty((M, N), device=A.device, dtype=A.dtype)

    block_m = 64
    block_n = 64
    block_k = 32
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    _matmul_bias_kernel[grid](
        A,
        B,
        bias,
        output,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output