import torch
import triton
import triton.language as tl


@triton.jit
def _attention_blocks(
    q, K, V, base, rows, m, l, acc,
    begin, end,
    SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, HEAD_DIM)
    scale: tl.constexpr = 1.4426950408889634 * HEAD_DIM ** -0.5

    k_ptrs = K + base + dims[:, None] + cols[None, :] * HEAD_DIM
    v_ptrs = V + base + cols[:, None] * HEAD_DIM + dims[None, :]

    for start in tl.range(begin, end, BLOCK_N):
        start = tl.multiple_of(start, BLOCK_N)
        keys = start + cols

        if SEQ % BLOCK_N == 0:
            k = tl.load(k_ptrs + start * HEAD_DIM)
            v = tl.load(v_ptrs + start * HEAD_DIM)
        else:
            k = tl.load(
                k_ptrs + start * HEAD_DIM,
                mask=keys[None, :] < SEQ,
                other=0,
            )
            v = tl.load(
                v_ptrs + start * HEAD_DIM,
                mask=keys[:, None] < SEQ,
                other=0,
            )

        scores = tl.dot(q, k) * scale

        if SEQ % BLOCK_N != 0:
            scores = tl.where(keys[None, :] < SEQ, scores, -float("inf"))
        if MASK_CAUSAL:
            scores = tl.where(
                keys[None, :] <= rows[:, None],
                scores,
                -float("inf"),
            )

        next_m = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp2(scores - next_m[:, None])
        alpha = tl.exp2(m - next_m)
        block_l = tl.sum(p, axis=1)

        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc)
        l = l * alpha + block_l
        m = next_m

    return m, l, acc


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64},
            num_warps=2,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=2,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64},
            num_warps=8,
            num_stages=2,
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
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    base = batch_head * SEQ * HEAD_DIM

    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
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
        prefix_end = (query_block * BLOCK_M // BLOCK_N) * BLOCK_N

        m, l, acc = _attention_blocks(
            q, K, V, base, rows, m, l, acc,
            0, prefix_end,
            SEQ, HEAD_DIM, BLOCK_M, BLOCK_N, False,
        )

        diagonal_end = tl.minimum((query_block + 1) * BLOCK_M, SEQ)
        m, l, acc = _attention_blocks(
            q, K, V, base, rows, m, l, acc,
            prefix_end, diagonal_end,
            SEQ, HEAD_DIM, BLOCK_M, BLOCK_N, True,
        )
    else:
        m, l, acc = _attention_blocks(
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
    output = torch.empty(Q.shape, dtype=Q.dtype, device=Q.device)

    grid = lambda meta: (
        triton.cdiv(seq, meta["BLOCK_M"]),
        batch * heads,
    )

    _flash_attention[grid](
        Q, K, V, output,
        SEQ=seq,
        HEAD_DIM=head_dim,
        BATCH_HEADS=batch * heads,
        CAUSAL=bool(causal),
    )
    return output