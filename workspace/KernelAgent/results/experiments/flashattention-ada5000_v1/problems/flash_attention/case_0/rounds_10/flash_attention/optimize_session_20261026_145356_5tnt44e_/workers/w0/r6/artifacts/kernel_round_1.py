import torch
import triton
import triton.language as tl


@triton.jit
def _attention_loop(
    q, K, V, base, rows, m, l, acc,
    start, stop,
    SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, HEAD_DIM)
    scale: tl.constexpr = 1.4426950408889634 * HEAD_DIM ** -0.5

    k_ptrs = (
        K + base
        + dims[:, None]
        + (start + cols[None, :]) * HEAD_DIM
    )
    v_ptrs = (
        V + base
        + (start + cols[:, None]) * HEAD_DIM
        + dims[None, :]
    )

    for block in range(start, stop, BLOCK_N):
        keys = block + cols

        if SEQ % BLOCK_N == 0:
            k = tl.load(k_ptrs)
        else:
            k = tl.load(k_ptrs, keys[None, :] < SEQ, other=0)

        scores = tl.dot(q, k)
        scores = scores * scale

        if SEQ % BLOCK_N != 0:
            scores = tl.where(keys[None, :] < SEQ, scores, -float("inf"))
        if MASK_CAUSAL:
            scores = tl.where(
                rows[:, None] >= keys[None, :],
                scores,
                -float("inf"),
            )

        m_next = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp2(scores - m_next[:, None])
        alpha = tl.exp2(m - m_next)
        l_next = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        if SEQ % BLOCK_N == 0:
            v = tl.load(v_ptrs)
        else:
            v = tl.load(v_ptrs, keys[:, None] < SEQ, other=0)

        acc = tl.dot(p.to(q.dtype), v, acc)
        m = m_next
        l = l_next

        k_ptrs += BLOCK_N * HEAD_DIM
        v_ptrs += BLOCK_N * HEAD_DIM

    return m, l, acc


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32},
            num_warps=4, num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64},
            num_warps=4, num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128},
            num_warps=4, num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32},
            num_warps=4, num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128},
            num_warps=4, num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64},
            num_warps=4, num_stages=2,
        ),
    ],
    key=["SEQ", "HEAD_DIM", "BATCH_HEADS", "CAUSAL"],
)
@triton.jit
def _flash_attention_kernel(
    Q, K, V, O,
    SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BATCH_HEADS: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    base = pid_bh * SEQ * HEAD_DIM
    row_start = pid_m * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + base + rows[:, None] * HEAD_DIM + dims[None, :]
    if SEQ % BLOCK_M == 0:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, rows[:, None] < SEQ, other=0)

    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.full((BLOCK_M, HEAD_DIM), 0.0, tl.float32)

    if CAUSAL:
        prefix_end = (row_start // BLOCK_N) * BLOCK_N
        m, l, acc = _attention_loop(
            q, K, V, base, rows, m, l, acc,
            0, prefix_end,
            SEQ, HEAD_DIM, BLOCK_M, BLOCK_N, False,
        )
        diagonal_end = tl.minimum(row_start + BLOCK_M, SEQ)
        m, l, acc = _attention_loop(
            q, K, V, base, rows, m, l, acc,
            prefix_end, diagonal_end,
            SEQ, HEAD_DIM, BLOCK_M, BLOCK_N, True,
        )
    else:
        m, l, acc = _attention_loop(
            q, K, V, base, rows, m, l, acc,
            0, SEQ,
            SEQ, HEAD_DIM, BLOCK_M, BLOCK_N, False,
        )

    result = (acc * tl.div_rn(1.0, l)[:, None]).to(O.dtype.element_ty)
    out_ptrs = O + base + rows[:, None] * HEAD_DIM + dims[None, :]

    if SEQ % BLOCK_M == 0:
        tl.store(out_ptrs, result)
    else:
        tl.store(out_ptrs, result, rows[:, None] < SEQ)


def kernel_function(Q, K, V, causal):
    batch, heads, seq, head_dim = Q.shape
    output = torch.empty(Q.shape, device=Q.device, dtype=Q.dtype)
    grid = lambda meta: (
        triton.cdiv(seq, meta["BLOCK_M"]),
        batch * heads,
    )
    _flash_attention_kernel[grid](
        Q, K, V, output,
        SEQ=seq,
        HEAD_DIM=head_dim,
        BATCH_HEADS=batch * heads,
        CAUSAL=bool(causal),
    )
    return output