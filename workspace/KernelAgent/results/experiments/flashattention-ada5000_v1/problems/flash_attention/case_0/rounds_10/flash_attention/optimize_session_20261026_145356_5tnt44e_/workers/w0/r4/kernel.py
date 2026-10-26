import torch
import triton
import triton.language as tl


@triton.jit
def _attention_loop(
    q, K, V, base, rows, m, l, acc,
    start, end,
    SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
):
    scale: tl.constexpr = 1.4426950408889634 * HEAD_DIM ** -0.5
    cols = tl.arange(0, BLOCK_N)

    k_ptr = tl.make_block_ptr(
        base=K + base,
        shape=(HEAD_DIM, SEQ),
        strides=(1, HEAD_DIM),
        offsets=(0, start),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    v_ptr = tl.make_block_ptr(
        base=V + base,
        shape=(SEQ, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(start, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )

    for block in tl.range(start, end, BLOCK_N):
        if SEQ % BLOCK_N == 0:
            k = tl.load(k_ptr)
        else:
            k = tl.load(k_ptr, boundary_check=(1,), padding_option="zero")

        scores = tl.dot(q, k)
        scores = scores * scale

        if SEQ % BLOCK_N != 0:
            scores = tl.where(
                block + cols[None, :] < SEQ, scores, -float("inf")
            )
        if MASK_CAUSAL:
            scores = tl.where(
                rows[:, None] >= block + cols[None, :],
                scores,
                -float("inf"),
            )

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp2(m - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        if SEQ % BLOCK_N == 0:
            v = tl.load(v_ptr)
        else:
            v = tl.load(v_ptr, boundary_check=(0,), padding_option="zero")

        acc = tl.dot(p.to(q.dtype), v, acc)
        m = m_new
        k_ptr = tl.advance(k_ptr, (0, BLOCK_N))
        v_ptr = tl.advance(v_ptr, (BLOCK_N, 0))

    return m, l, acc


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=2, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2
        ),
    ],
    key=["SEQ", "HEAD_DIM", "BATCH_HEADS", "CAUSAL"],
)
@triton.jit
def _flash_attention(
    Q, K, V, O,
    SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BATCH_HEADS: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    base = bh * SEQ * HEAD_DIM
    rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)

    q_ptr = tl.make_block_ptr(
        base=Q + base,
        shape=(SEQ, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(tile * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    if SEQ % BLOCK_M == 0:
        q = tl.load(q_ptr)
    else:
        q = tl.load(q_ptr, boundary_check=(0,), padding_option="zero")

    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    if CAUSAL:
        prefix_end = (tile * BLOCK_M // BLOCK_N) * BLOCK_N
        m, l, acc = _attention_loop(
            q, K, V, base, rows, m, l, acc,
            0, prefix_end,
            SEQ, HEAD_DIM, BLOCK_M, BLOCK_N, False,
        )
        diagonal_end = tl.minimum((tile + 1) * BLOCK_M, SEQ)
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

    result = (acc / l[:, None]).to(O.dtype.element_ty)
    out_ptrs = O + base + rows[:, None] * HEAD_DIM + dims[None, :]
    if SEQ % BLOCK_M == 0:
        tl.store(out_ptrs, result)
    else:
        tl.store(out_ptrs, result, mask=rows[:, None] < SEQ)


def kernel_function(Q, K, V, causal):
    batch, heads, seq, head_dim = Q.shape
    output = torch.empty(Q.shape, device=Q.device, dtype=Q.dtype)
    grid = lambda meta: (triton.cdiv(seq, meta["BLOCK_M"]), batch * heads)
    _flash_attention[grid](
        Q, K, V, output,
        SEQ=seq,
        HEAD_DIM=head_dim,
        BATCH_HEADS=batch * heads,
        CAUSAL=bool(causal),
    )
    return output