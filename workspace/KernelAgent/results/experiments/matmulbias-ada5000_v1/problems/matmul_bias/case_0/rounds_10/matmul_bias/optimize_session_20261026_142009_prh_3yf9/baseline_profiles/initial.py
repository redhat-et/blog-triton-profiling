import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_bias_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one output tile of A @ B + bias, accumulating in fp32."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k

        a = tl.load(
            a_ptr + rows[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(rows[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k[:, None] * stride_bk + cols[None, :] * stride_bn,
            mask=(k[:, None] < K) & (cols[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    # Fuse the bias addition into the matmul epilogue.
    bias = tl.load(bias_ptr + cols * stride_bias, mask=cols < N, other=0.0)
    result = acc + bias[None, :]

    out_ptrs = out_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on
    out_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(out_ptrs, result, mask=out_mask)


def kernel_function(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Return A @ B + bias, fusing matrix multiplication and bias addition."""
    if A.ndim != 2 or B.ndim != 2 or bias.ndim != 1:
        raise ValueError("Expected A and B to be matrices and bias to be a vector")

    M, K = A.shape
    K_b, N = B.shape
    if K != K_b or bias.numel() != N:
        raise ValueError("Incompatible matrix or bias dimensions")
    if A.device != B.device or A.device != bias.device:
        raise ValueError("All inputs must be on the same device")
    if A.dtype != B.dtype or A.dtype != bias.dtype:
        raise ValueError("All inputs must have the same dtype")
    if A.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Only float16 and bfloat16 inputs are supported")

    output = torch.empty((M, N), device=A.device, dtype=A.dtype)

    block_m, block_n, block_k = 32, 64, 32
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
        bias.stride(0),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output