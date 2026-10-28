import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
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
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, HEAD_DIM)
    base = pid_bh * SEQ * HEAD_DIM

    q = tl.load(
        Q + base + rows[:, None] * HEAD_DIM + dims[None, :],
        mask=rows[:, None] < SEQ,
        other=0.0,
    )

    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    scale: tl.constexpr = 1.4426950408889634 * HEAD_DIM**-0.5

    end = SEQ
    if CAUSAL:
        end = tl.minimum(SEQ, (pid_m + 1) * BLOCK_M)

    for start in range(0, end, BLOCK_N):
        keys = start + cols

        k = tl.load(
            K + base + dims[:, None] + keys[None, :] * HEAD_DIM,
            mask=keys[None, :] < SEQ,
            other=0.0,
        )
        scores = tl.dot(q, k) * scale
        scores = tl.where(keys[None, :] < SEQ, scores, -float("inf"))

        if CAUSAL:
            if start + BLOCK_N > pid_m * BLOCK_M:
                scores = tl.where(
                    rows[:, None] >= keys[None, :],
                    scores,
                    -float("inf"),
                )

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp2(m - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(
            V + base + keys[:, None] * HEAD_DIM + dims[None, :],
            mask=keys[:, None] < SEQ,
            other=0.0,
        )
        acc = tl.dot(p.to(q.dtype), v, acc)
        m = m_new

    result = acc / l[:, None]
    tl.store(
        O + base + rows[:, None] * HEAD_DIM + dims[None, :],
        result,
        mask=rows[:, None] < SEQ,
    )


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