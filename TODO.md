# TODO — deferred cleanup

The items below are deliberate contracts rather than defects. Everything
else from the `/simplify` pass has been fixed: the optimizer timeout now
ships as compatibility patch `0007` (no runtime monkey-patch), the optimizer
problem module embeds the whole `spec.py`, the Nsight UIs launch through one
sanitizing wrapper on `PATH`, the Makefile forwards `KA_TEST_TIMEOUT` and
`KA_PROFILE_WARMUP`, and the image no longer bakes an LLM backend.

## `KERNELAGENT_REV` duplicated in `Makefile:203` and `Dockerfile.cuda:54`

AGENTS.md explicitly requires the two pins to be identical, so the duplication
is the contract.

**Next step:** leave as-is; the build's `rev-parse` assertion catches drift.

## Makefile/Dockerfile copies of Python constants

Mount paths, the profiler list, and `KA_VERIFY_RECORD` are restated in the
Makefile/Dockerfile beside `ka_common.py` (`PROBLEMS_HOME_DEFAULT`,
`SUPPORTED_PROFILERS`, `IMPL_ARTIFACTS`). Make cannot import `ka_common`, and
deriving them would add more machinery than it removes.

**Next step:** leave as-is unless a constant changes often.

## `check_install.py` hardcoded GPU figures

The gate pins the RTX 5000 Ada laptop entry added by patch 0006 and the
desktop RTX 5090 entry added by patch 0008. AGENTS.md says not to weaken
`check_install.py`.

**Next step:** leave as-is; it is an intentional build gate.

## `KA_PROBLEM ?= matmul_bias`

The Makefile hardcodes one default problem for single-problem targets while
`KA_ALL_PROBLEMS` is derived from `problems/*/spec.py`. Deriving the default
would change published workflows.

**Next step:** leave as-is; it is the documented default.

## `rocm_blog_fusion` autotune space on consumer NVIDIA GPUs

The blog-faithful space in `problems/matmul_bias/baselines/rocm_blog_fusion.py`
(24 configs, `BLOCK_SIZE_M=128` only) was sized for datacenter parts. Consumer
NVIDIA GPUs cap shared memory at 99 KB per block (128 KB per SM), so 18 of the
24 configs need 128–384 KiB and raise `OutOfResources`, which
`triton.autotune` silently scores as `inf`. Only 128×128×64 at stages 3/4 and
128×256×64 at stage 3 launch. That is still enough on the RTX 5000 Ada laptop
for the default case (1024³): 90.06 TFLOPS vs cuBLAS 90.26, with the autotuner
choosing 128×128×64, stages=4, warps=8. On a 170-SM RTX 5090 the same 128-row
tiles give only 64 CTAs at 1024³ (38% of the SMs), and the space has no
64-row/64-column tiles or `BK=32` to raise that count.

**Next step:** leave as-is; AGENTS.md requires the baseline to stay faithful
to the blog source. A tuned comparison would be a separately named variant
with a pruned, GPU-appropriate space.
