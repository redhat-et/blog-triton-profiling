import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4, num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4, num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4, num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8, num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_bias_kernel(
    A,
    B,
    Bias,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group_size: tl.constexpr = 8
    programs_per_group = group_size * num_n
    group_id = pid // programs_per_group
    first_m = group_id * group_size
    actual_group_size = tl.minimum(num_m - first_m, group_size)
    local_pid = pid % programs_per_group
    pid_m = first_m + local_pid % actual_group_size
    pid_n = local_pid // actual_group_size

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)

    a_ptrs = A + rows[:, None] * stride_am + ks[None, :] * stride_ak
    b_ptrs = B + ks[:, None] * stride_bk + cols[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for block in range(tl.cdiv(K, BLOCK_K)):
        if M % BLOCK_M == 0 and K % BLOCK_K == 0:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=(rows[:, None] < M)
                & (block * BLOCK_K + ks[None, :] < K),
                other=0.0,
            )

        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(
                b_ptrs,
                mask=(cols[None, :] < N)
                & (block * BLOCK_K + ks[:, None] < K),
                other=0.0,
            )

        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(
        Bias + cols * stride_bias, mask=cols < N, other=0.0
    ).to(tl.float32)
    result = (acc + bias[None, :]).to(Out.dtype.element_ty)
    out_ptrs = Out + rows[:, None] * N + cols[None, :]
    tl.store(
        out_ptrs,
        result,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def kernel_function(A, B, bias):
    M, K = A.shape
    N = B.shape[1]
    output = torch.empty((M, N), device=A.device, dtype=A.dtype)
    if M == 0 or N == 0:
        return output

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
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
    )
    return output