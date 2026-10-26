import torch
import triton
import triton.language as tl


@triton.jit
def _attention_inner(
    q,
    K,
    V,
    base,
    rows,
    m,
    l,
    acc,
    start,
    end,
    SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, HEAD_DIM)
    scale: tl.constexpr = 1.4426950408889634 * HEAD_DIM**-0.5

    for offset in tl.range(start, end, BLOCK_N, num_stages=PIPE_STAGES):
        keys = offset + cols
        k_ptrs = K + base + keys[None, :] * HEAD_DIM + dims[:, None]
        v_ptrs = V + base + keys[:, None] * HEAD_DIM + dims[None, :]

        if SEQ % BLOCK_N == 0:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=keys[None, :] < SEQ, other=0)
            v = tl.load(v_ptrs, mask=keys[:, None] < SEQ, other=0)

        scores = tl.dot(q, k) * scale

        if SEQ % BLOCK_N != 0:
            scores = tl.where(keys[None, :] < SEQ, scores, -float("inf"))
        if MASK_CAUSAL:
            scores = tl.where(
                rows[:, None] >= keys[None, :], scores, -float("inf")
            )

        next_m = tl.maximum(m, tl.max(scores, axis=1))
        correction = tl.exp2(m - next_m)
        p = tl.exp2(scores - next_m[:, None])

        acc = acc * correction[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc)
        l = l * correction + tl.sum(p, axis=1)
        m = next_m

    return m, l, acc


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "PIPE_STAGES": 2},
            num_warps=4, num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "PIPE_STAGES": 2},
            num_warps=4, num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "PIPE_STAGES": 2},
            num_warps=4, num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "PIPE_STAGES": 2},
            num_warps=4, num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "PIPE_STAGES": 3},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "PIPE_STAGES": 2},
            num_warps=4, num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "PIPE_STAGES": 2},
            num_warps=4, num_stages=2,
        ),
    ],
    key=["SEQ", "HEAD_DIM", "BATCH_HEADS", "CAUSAL"],
)
@triton.jit
def _flash_attention_kernel(
    Q,
    K,
    V,
    O,
    SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BATCH_HEADS: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    base = pid_bh * SEQ * HEAD_DIM

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + base + rows[:, None] * HEAD_DIM + dims[None, :]

    if SEQ % BLOCK_M == 0:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=rows[:, None] < SEQ, other=0)

    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    if CAUSAL:
        prefix_end = (pid_m * BLOCK_M // BLOCK_N) * BLOCK_N
        m, l, acc = _attention_inner(
            q, K, V, base, rows, m, l, acc,
            0, prefix_end,
            SEQ, HEAD_DIM, BLOCK_M, BLOCK_N, False, PIPE_STAGES,
        )
        diagonal_end = tl.minimum((pid_m + 1) * BLOCK_M, SEQ)
        m, l, acc = _attention_inner(
            q, K, V, base, rows, m, l, acc,
            prefix_end, diagonal_end,
            SEQ, HEAD_DIM, BLOCK_M, BLOCK_N, True, PIPE_STAGES,
        )
    else:
        m, l, acc = _attention_inner(
            q, K, V, base, rows, m, l, acc,
            0, SEQ,
            SEQ, HEAD_DIM, BLOCK_M, BLOCK_N, False, PIPE_STAGES,
        )

    result = acc / l[:, None]
    out_ptrs = O + base + rows[:, None] * HEAD_DIM + dims[None, :]

    if SEQ % BLOCK_M == 0:
        tl.store(out_ptrs, result)
    else:
        tl.store(out_ptrs, result, mask=rows[:, None] < SEQ)


def kernel_function(Q, K, V, causal):
    batch, heads, seq, head_dim = Q.shape
    output = torch.empty(Q.shape, device=Q.device, dtype=Q.dtype)
    if seq == 0 or batch * heads == 0:
        return output

    grid = lambda meta: (
        triton.cdiv(seq, meta["BLOCK_M"]),
        batch * heads,
    )
    _flash_attention_kernel[grid](
        Q,
        K,
        V,
        output,
        SEQ=seq,
        HEAD_DIM=head_dim,
        BATCH_HEADS=batch * heads,
        CAUSAL=bool(causal),
    )
    return output