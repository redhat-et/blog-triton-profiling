# AGENTS.md — guidance for coding agents

## Project scope

This repository is demo material for blog posts: profiling Triton kernels on
NVIDIA CUDA and AMD ROCm, and generating, optimizing, and profiling Triton
kernels with KernelAgent. It is **not production grade**.

- Prefer small, readable changes that keep the demos runnable.
- Do not add production hardening, broad refactors, or abstractions for
  hypothetical use.
- Keep target names, output paths, and documented workflows stable — published
  posts link to them.

## Validation policy: run the demo, do not build a test suite

There is intentionally no unit, integration, or regression test suite and no
CI. Do not add tests, pytest configuration, test dependencies, or workflows.

The only automated check in the project is
`containerfiles/kernelagent/check_install.py`, which the final stage of
`containerfiles/Dockerfile.cuda` copies and RUNs. Keep that file, its `COPY`,
and its `RUN`: it is a build gate for the installed wheel, not a test suite. Do
not weaken it.

Validate changes by running the demo:

- Documentation or comment changes: `make help` and `make -n <target>`.
- Runtime or build changes: `make cuda-image` (or `make kernelagent-image`)
  must succeed — the build runs `check_install.py` — then run the relevant
  `make kernelagent-*` target.
- LLM-free targets: `kernelagent-verify`, `kernelagent-profile*`,
  `kernelagent-experiment-profile`, `kernelagent-experiment-baseline`,
  `kernelagent-report`.
- LLM/GPU targets: `kernelagent-test OPENAI_MODEL=...`, then generation,
  verification, optimization, and profiling.

A host-side developer test suite was deliberately removed. Do not restore it:
`containerfiles/test_entrypoint_cuda.py`,
`workspace/KernelAgent/scripts/ka_test_support.py`, and the
`workspace/KernelAgent/scripts/test_*.py` files (`test_bench_contract.py`,
`test_generate_kernel.py`, `test_baseline_sync.py`,
`test_experiment_kernels.py`, `test_ka_common.py`,
`test_profile_kernel.py`).

## Build and architecture map

- `Makefile` — every demo target. `cuda-image` builds the shared CUDA image
  used by `cuda-*`, `nsight-*`, and `kernelagent-*`; `kernelagent-image` is an
  alias. `rocm-image` builds the AMD image.
- `containerfiles/Dockerfile.cuda` — multi-stage build: `runtime-base`,
  `python-base`, `kernelagent-builder`, `runtime`.
- `containerfiles/Dockerfile.rocm` — AMD profiling image.
- `containerfiles/entrypoint.sh.cuda` — selects the `cuda` or `kernelagent`
  workflow from `CONTAINER_WORKFLOW` and routes the LLM backend.
- `workspace/CUDA`, `workspace/ROCm` — profiling demo materials.
- `workspace/KernelAgent/problems` — problem specs, mounted into the container.
  `workspace/KernelAgent/scripts` — CLIs baked into the image.
  `workspace/KernelAgent/results` — runtime artifacts (kernels, records,
  experiments, per-problem profiles), mounted and gitignored.

Runtime model prompts are `workspace/KernelAgent/problems/<problem>/problem.txt`,
consumed by `generate-kernel.py` and embedded in optimizer problem modules.
Never move prompt text into documentation or AGENTS.md.

## KernelAgent pin, patches, and dependency lock

- `KERNELAGENT_REV` must be identical in `Makefile` and
  `containerfiles/Dockerfile.cuda`, and a 40-character commit SHA. The builder
  fails unless `git rev-parse HEAD` matches the pin.
- Bump the revision deliberately. Every patch in
  `containerfiles/kernelagent/patches/` (`0001`–`0008`) must still apply; adapt
  the patch series when upstream changes instead of dropping compatibility.
- After a revision, patch, or dependency bump, rebuild the image and revalidate
  end to end: provider smoke test, generation, verification, optimization,
  profiling.
- Do not stage local KernelAgent source into the image. Provenance is the pin,
  the patch set, the lock, and the image labels.
- Dependency updates go through `containerfiles/kernelagent/requirements.in`;
  regenerate `requirements.lock` with the exact `uv pip compile` command in
  that file's header. Never hand-edit the lock. The committed lock targets
  CUDA 12.9, Python 3.12, linux x86_64; other combinations need a matching lock
  or the build fails on the torch pin.
- `containerfiles/kernelagent/build-constraints.txt` pins the wheel-build
  toolchain; bump it deliberately with the same revalidation.

## Runtime contracts and kernel acceptance

- Generated kernels define a top-level `kernel_function(...)` and perform the
  computation inside `@triton.jit` kernels. The problem specs forbid
  `torch.matmul`, `torch.mm`, `torch.bmm`, `torch.einsum`,
  `F.scaled_dot_product_attention`, cuBLAS, and other torch compute ops.
- Flash attention must use the online (running) softmax over tiled K/V blocks
  with `m_i` / `l_i` accumulators, without materializing the full `(S, S)`
  score matrix.
- Acceptance is the full `test_fixed.py` plus `bench.py`. An optimized kernel is
  accepted only when it passes the full fixed test, completes the benchmark,
  and is strictly faster on the selected case under the fixed `bench.py`.
- A numerical test cannot detect torch/cuBLAS delegation or a non-flash
  algorithm. Review `results/<problem>/final_kernel.py` and
  `optimized_kernel.py` before trusting them.
- External baselines and adapted kernels must satisfy the same contract
  (signature, output dtype following the input, tested shapes). Do not change
  the kernel body or autotune space when adapting a baseline for comparison.
- Never weaken `test_fixed.py`, `bench.py`, or `problem.txt` to make a kernel
  pass.

## Problem-directory contract

- Every `workspace/KernelAgent/problems/<name>/` holds `problem.txt`,
  `test_fixed.py`, `bench.py`, and `spec.py`; `problems/_harness.py` is
  shared. The Makefile derives the problem list from `problems/*/spec.py`, and
  `check_install.py` validates the full set at build time.
- `test_fixed.py` and `bench.py` are hand-written acceptance artifacts. Edit
  them only intentionally, keeping them consistent with `spec.py` and
  `problem.txt`, then re-run `make kernelagent-verify KA_PROBLEM=<problem>`.
- Preserve the `@GPU_TARGET@` token in `problem.txt`; it is resolved at
  generation and optimization time.

## Experiment, optimization, and baseline invariants

- Experiments record and report failures. A failed generation, crashed
  optimization, non-improving candidate, or failed verification is never
  silently replaced by another variant's kernel, and the run continues with
  the remaining budgets and problems.
- Captures profile verified sources only; archived kernels are hash-checked
  against their records.
- An experiment is immutable apart from added capture batches and external
  baselines: its problem set, case, budgets, and LLM configuration are fixed
  at creation, and retrying an optimization means a new experiment id.
- External baselines are archived, hashed, verified against the experiment's
  problem snapshot, and recorded. `KA_LABEL` must not collide with a variant name
  or the reserved `kernel`, `optimized`, `ref` stems. Re-adding the same source
  under the same label is a no-op; a different source is verified first so a
  contract failure is never hidden by the label collision, and a recorded
  baseline is never replaced by a later attempt.
- The single-kernel optimizer keeps `optimized_kernel.py` separate from the
  generated kernel; a failed attempt leaves a previously accepted kernel
  untouched. Results use atomic writes and a per-problem lock, so shared
  summaries always describe the kernel on disk.
- `report-experiment.py` must remain host-side standard-library-only.
  Experiment and report CLIs must not import torch or triton at module level.

## Configuration and environment

- `OPENAI_MODEL` selects both the model and its backend: `claude-*` ids use the
  Anthropic API unless the user sets `OPENAI_BASE_URL` (the public URL the
  entrypoint defaults to does not reroute `claude-*`); anything else uses the OpenAI
  API or an OpenAI-compatible server. `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`
  are passed by reference and must never be committed.
- `KA_REASONING_EFFORT` and the per-stage `KA_GENERATION_MODEL` /
  `KA_OPTIMIZATION_MODEL` (`OPENAI_MODEL[:EFFORT]`) experiment overrides
  behave as documented in `workspace/KernelAgent/README.md`.
- `CUDA_VISIBLE_DEVICES` selects devices, but optimization runs on the first
  selected device. Prefer UUIDs on hosts where CUDA and `nvidia-smi` ordering
  may differ. `KA_GPU_NAME`, `KA_GPU_TARGET`, and `KA_GPU_SPECS_FILE` keep
  their documented meanings.
- Host file/path inputs (`KA_KERNEL_SOURCE`, `KA_GPU_SPECS_FILE`, an existing
  experiment directory) are validated at make parse time, before any image
  build or container run. A host kernel source outside the container mounts is
  copied into the mounted results tree — the experiment's `sources/` directory
  for `kernelagent-experiment-baseline`, `results/.kernel-sources/` for
  `kernelagent-profile` — and the container is given that path; the original
  path is recorded as the baseline's `source_path`.
- When adding or removing targets or user-facing variables, update `make help`
  and the container entrypoint banner together.

## Do not edit / generated artifacts

- Do not commit `workspace/KernelAgent/results/*` (only `.gitkeep` is
  tracked) or the legacy `workspace/KernelAgent/profiles/` directory, generated
  kernels, profile captures, session directories, `.env` files, or credentials.
- Do not hand-edit `containerfiles/kernelagent/requirements.lock`.
- Do not casually edit `problem.txt`, `test_fixed.py`, `bench.py`, or `spec.py`;
  they define the runtime acceptance contract.
- Do not move runtime prompts into docs or this file.

## Documentation rules

- Human documentation (`README.md`, `workspace/KernelAgent/README.md`)
  explains what the demo is and how to run it.
- Agent and maintainer rules live only in this file. `CLAUDE.md` is a
  one-line import of it; do not duplicate content.
- Keep inline comments that explain mechanics. Put policy and procedure here
  and point to it from comments.
