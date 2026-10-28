import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=2, num_stages=2),
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
):
    block_m = tl.program_id(0)
    batch_head = tl.program_id(1)

    rows = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, HEAD_DIM)
    base = batch_head * SEQ * HEAD_DIM

    q_ptrs = Q + base + rows[:, None] * HEAD_DIM + dims[None, :]
    if SEQ % BLOCK_M == 0:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, rows[:, None] < SEQ, other=0.0)

    k_ptrs = K + base + dims[:, None] + cols[None, :] * HEAD_DIM
    v_ptrs = V + base + cols[:, None] * HEAD_DIM + dims[None, :]

    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    normalizer = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.full((BLOCK_M, HEAD_DIM), 0.0, tl.float32)
    scale: tl.constexpr = 1.4426950408889634 * HEAD_DIM**-0.5

    if CAUSAL:
        end = tl.minimum((block_m + 1) * BLOCK_M, SEQ)
    else:
        end = SEQ

    for start in tl.range(0, end, BLOCK_N):
        keys = start + cols

        if SEQ % BLOCK_N == 0:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, keys[None, :] < SEQ, other=0.0)
            v = tl.load(v_ptrs, keys[:, None] < SEQ, other=0.0)

        scores = tl.dot(q, k)
        scores = scores * scale

        if SEQ % BLOCK_N != 0:
            scores = tl.where(keys[None, :] < SEQ, scores, -float("inf"))

        if CAUSAL:
            if start + BLOCK_N > block_m * BLOCK_M:
                scores = tl.where(
                    rows[:, None] >= keys[None, :],
                    scores,
                    -float("inf"),
                )

        new_m = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp2(m - new_m)
        probabilities = tl.exp2(scores - new_m[:, None])
        block_sum = tl.sum(probabilities, axis=1)

        acc = acc * alpha[:, None]
        acc = tl.dot(probabilities.to(q.dtype), v, acc)
        normalizer = normalizer * alpha + block_sum
        m = new_m

        k_ptrs += BLOCK_N * HEAD_DIM
        v_ptrs += BLOCK_N * HEAD_DIM

    result = acc / normalizer[:, None]
    out_ptrs = O + base + rows[:, None] * HEAD_DIM + dims[None, :]

    if SEQ % BLOCK_M == 0:
        tl.store(out_ptrs, result)
    else:
        tl.store(out_ptrs, result, rows[:, None] < SEQ)


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