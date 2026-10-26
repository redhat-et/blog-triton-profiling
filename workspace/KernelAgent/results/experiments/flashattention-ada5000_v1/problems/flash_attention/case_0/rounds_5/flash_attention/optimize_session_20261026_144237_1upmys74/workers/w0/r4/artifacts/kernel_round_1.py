import torch
import triton
import triton.language as tl


@triton.jit
def _attention_inner(
    q, K, V, base, rows, dims,
    acc, normalizer, maximum,
    lo, hi,
    SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_N)
    scale: tl.constexpr = 1.4426950408889634 * HEAD_DIM ** -0.5

    k_ptrs = K + base + dims[:, None] + cols[None, :] * HEAD_DIM
    v_ptrs = V + base + cols[:, None] * HEAD_DIM + dims[None, :]

    for start in tl.range(lo, hi, BLOCK_N):
        keys = start + cols

        if SEQ % BLOCK_N == 0:
            k = tl.load(k_ptrs + start * HEAD_DIM)
            v = tl.load(v_ptrs + start * HEAD_DIM)
        else:
            k = tl.load(
                k_ptrs + start * HEAD_DIM,
                mask=keys[None, :] < SEQ,
                other=0.0,
            )
            v = tl.load(
                v_ptrs + start * HEAD_DIM,
                mask=keys[:, None] < SEQ,
                other=0.0,
            )

        scores = tl.dot(q, k)

        if SEQ % BLOCK_N != 0:
            scores = tl.where(keys[None, :] < SEQ, scores, -float("inf"))
        if MASK_CAUSAL:
            scores = tl.where(
                rows[:, None] >= keys[None, :], scores, -float("inf")
            )

        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1) * scale)
        probabilities = tl.exp2(scores * scale - new_maximum[:, None])
        correction = tl.exp2(maximum - new_maximum)
        block_sum = tl.sum(probabilities, axis=1)

        acc = acc * correction[:, None]
        acc = tl.dot(probabilities.to(q.dtype), v, acc)
        normalizer = normalizer * correction + block_sum
        maximum = new_maximum

    return acc, normalizer, maximum


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
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
    tile = tl.program_id(0)
    head = tl.program_id(1)
    base = head * SEQ * HEAD_DIM
    rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + base + rows[:, None] * HEAD_DIM + dims[None, :]
    if SEQ % BLOCK_M == 0:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=rows[:, None] < SEQ, other=0.0)

    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    if CAUSAL:
        diagonal_start = tile * BLOCK_M
        acc, normalizer, maximum = _attention_inner(
            q, K, V, base, rows, dims,
            acc, normalizer, maximum,
            0, diagonal_start,
            SEQ, HEAD_DIM, BLOCK_M, BLOCK_N, False,
        )
        acc, normalizer, maximum = _attention_inner(
            q, K, V, base, rows, dims,
            acc, normalizer, maximum,
            diagonal_start, tl.minimum(diagonal_start + BLOCK_M, SEQ),
            SEQ, HEAD_DIM, BLOCK_M, BLOCK_N, True,
        )
    else:
        acc, normalizer, maximum = _attention_inner(
            q, K, V, base, rows, dims,
            acc, normalizer, maximum,
            0, SEQ,
            SEQ, HEAD_DIM, BLOCK_M, BLOCK_N, False,
        )

    reciprocal = 1.0 / normalizer
    result = acc * reciprocal[:, None]
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
        Q, K, V, output,
        SEQ=seq,
        HEAD_DIM=head_dim,
        BATCH_HEADS=batch * heads,
        CAUSAL=bool(causal),
    )
    return output