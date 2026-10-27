# KernelAgent — Agentic Triton Kernel Generation

Materials for the third blog in this series: using an LLM agent
([KernelAgent](https://github.com/meta-pytorch/KernelAgent)) to *generate*
Triton GPU kernels from a written problem specification, then verifying,
benchmarking, optimizing, and profiling the results with the same tools
covered in the CUDA and ROCm posts.

This workflow is **CUDA-only** (KernelAgent generates Triton kernels for
NVIDIA GPUs). It runs in the **shared CUDA image** — the same image the
`cuda-*` / `nsight-*` targets use, with the Nsight CLI tools and UIs and
a Jupyter notebook server on board. The `kernelagent-*` targets mount
`workspace/KernelAgent`'s `problems/`, `results/`, and `profiles/`
directories as the container's data (the CLI scripts themselves are baked
into the image, not mounted — see `ka_vol` in the Makefile) and configure
the LLM backend; no separate KernelAgent image exists.

```text
workspace/KernelAgent/
  problems/
    matmul_bias/      Fused matmul + bias problem spec
      baselines/      Hand-written baseline kernels (the ROCm blog's fusion kernel)
    flash_attention/  Flash attention problem spec
  scripts/
    generate-kernel.py  Generate / optimize / verify kernels from the fixed specs
    verify-llm.py       Smoke-test the configured LLM provider
    profile-kernel.py   Profile a saved, optimized, ref, or external kernel
    experiment-kernels.py  Comparison experiments: run, add capture batches,
                           add external baselines
    report-experiment.py   Render an experiment's Markdown/JSON report
    ka_common.py           Helpers the CLIs share (atomic writes, spec loading,
                           the CUDA probe) — imported, not run directly
  results/            KernelAgent sessions, kernels, verification records
  profiles/           Proton / Nsight captures
```

Each problem directory holds the complete spec the agent works from:

- `problem.txt` — the problem description given to the agent
- `test_fixed.py` — a fixed correctness test used to verify kernels
- `bench.py` — benchmark cases against the PyTorch baseline
- `spec.py` — case data, input construction, and reference ops

`problems/_harness.py` is shared by every problem: it holds the CUDA guard,
kernel import, and verification/benchmark scaffolding.

Because the test is fixed and written by hand, results are reproducible and
independently verifiable: the same `final_kernel.py` can be re-verified and
re-profiled at any time, without an LLM.

The problem-directory contract and kernel review rules are in
[AGENTS.md](../../AGENTS.md).

## Requirements

- NVIDIA GPU
- Podman or Docker with GPU support
- `make`
- An LLM for generation and optimization only (see below); verification
  and profiling need none

## LLM backends

One variable selects both the model and its backend: `OPENAI_MODEL`. The backend
is inferred from the model id — a `claude-*` id talks to the Anthropic
API, anything else (or the same id alongside an `OPENAI_BASE_URL`)
talks to the OpenAI API or any OpenAI-compatible server:

| Model id | Variables |
|---|---|
| `claude-*` | `OPENAI_MODEL` (default `claude-sonnet-4-6`, also the `make` default), `ANTHROPIC_API_KEY` |
| anything else | `OPENAI_MODEL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL` (optional) |

Secrets are passed into the container by reference — export them in your
shell (or on the make command line); they are never stored in the repo.

### OpenAI API

```bash
export OPENAI_API_KEY=sk-...
make kernelagent-test OPENAI_MODEL=gpt-5.2
```

### Any OpenAI-compatible local server

vLLM, llama.cpp, Ollama, LM Studio, and most local inference servers expose
an OpenAI-compatible `/v1` API. Point `OPENAI_BASE_URL` at the server; it
runs on the host, use `host.containers.internal` as the hostname. Set
`OPENAI_API_KEY` to any non-empty value if your server does not check keys
(the OpenAI client refuses to start without one):

```bash
export OPENAI_API_KEY=local
make kernelagent-test \
  OPENAI_MODEL=<model id from your server> \
  OPENAI_BASE_URL=http://host.containers.internal:8000/v1
```

`OPENAI_BASE_URL` also routes `claude-*` ids through the server instead of
the Anthropic API.

List the model id your server exposes with
`curl -s http://127.0.0.1:8000/v1/models`.

### Anthropic API

```bash
export ANTHROPIC_API_KEY=sk-ant-...
make kernelagent-test OPENAI_MODEL=claude-sonnet-4-6
```

### Reasoning effort

`KA_REASONING_EFFORT` requests a reasoning level from the selected model
on either backend: `low`, `medium`, or `high`. It applies to every model
id — including ones the providers would otherwise leave at their default
(such as `gpt-6.1-sol` or a model served by a local OpenAI-compatible
server). Leave it empty for the previous behavior, where only `gpt-5*`
ids (and `o1`/`o3`) requested high reasoning:

```bash
make kernelagent-test OPENAI_MODEL=gpt-5.2 KA_REASONING_EFFORT=low
```

On the OpenAI-compatible backend this sends `reasoning_effort`; on the
Anthropic backend it enables adaptive thinking with the matching effort
level. An invalid value fails `kernelagent-test` immediately.

Comparison experiments can additionally put the generation and
optimization stages on different models or reasoning levels — see
`KA_GENERATION_MODEL` / `KA_OPTIMIZATION_MODEL` under
[Comparison experiments](#comparison-experiments).

## The workflow

### 0. Build the image

```bash
make kernelagent-image
```

Alias for `make cuda-image`: KernelAgent is part of the shared CUDA
image, so this is the same build the `cuda-*` / `nsight-*` targets use.

The image builds KernelAgent from source, pinned and verified: the
container build clones `KERNELAGENT_UPSTREAM` at exactly `KERNELAGENT_REV`
(the build fails unless `git rev-parse HEAD` matches the pin), applies
the compatibility patches from `containerfiles/kernelagent/patches/`,
builds a wheel, and the final image stage installs that wheel into the
locked Python environment. A fresh clone of this repository reproduces
the tested image with no local KernelAgent source, and no source
checkout ships inside the image — only the installed package and its
resources.

The compatibility patches:

- `0001-openai-base-url-provider.patch` — honors `OPENAI_BASE_URL` and picks
  the provider for every model id, registered or not: the Anthropic provider
  when `LLM_BACKEND=anthropic` (which the entrypoint derives from a
  `claude-*` model id, so any current `claude-*` id works), the
  OpenAI-compatible provider when `LLM_BACKEND=openai` or `OPENAI_BASE_URL`
  is set (so public and local servers both work for any id, including
  registry names served by a proxy), and the internal registry with its
  relay fallback otherwise

- `0002-openai-max-completion-tokens.patch` — sends `max_completion_tokens`
  and no default temperature for o-series and GPT models with major version
  5 or newer, which the Chat Completions API requires instead of
  `max_tokens`; Azure's legacy `gpt-35-*` ids are GPT-3.5 and keep
  `max_tokens`

- `0003-unknown-gpu-specs.patch` — proceeds with name-only GPU specs when
  the GPU is missing from the optimizer's specs catalog instead of crashing
  every optimization worker, and adds the `KA_GPU_SPECS_FILE` override (see
  `Unknown GPU: ...` under Troubleshooting): a JSON file of GPU name →
  specs mappings merged over the shipped catalog, so an uncatalogued GPU
  can optimize with real peak figures instead of the name-only fallback

- `0004-installed-wheel-resources.patch` — ships the resources a
  wheel-installed KernelAgent needs but upstream's packaging left out:
  the `utils.providers` subpackage (without it even
  `import triton_kernel_agent` fails), the optimizer's NCU wrapper
  template, and a package-relative default path for the RAG optimization
  database

- `0005-reasoning-effort-env.patch` — reads `KA_REASONING_EFFORT`
  (`low | medium | high`) in the provider layer: set, it overrides the
  model-name reasoning gating for any model id on the OpenAI-compatible
  path and enables adaptive thinking with an explicit effort level on
  the Anthropic path (no explicit temperature, and a response with no
  text block raises with its stop reason); unset, behavior is unchanged.
  Invalid values raise instead of silently running at the wrong level

- `0006-rtx-5000-ada-laptop-specs.patch` — adds this repository's target
  laptop GPU (`NVIDIA RTX 5000 Ada Generation Laptop GPU`) to the shipped GPU
  specs catalog, so the optimizer uses real peak bandwidth/FLOPs figures by
  default instead of the name-only fallback. `KA_GPU_SPECS_FILE` still takes
  precedence when overriding those figures or adding another GPU

- `0007-optimizer-test-timeout.patch` — threads the per-test subprocess
  timeout through optimization: `OptimizationWorker` and the manager-level
  `NvidiaVerifier` now accept `test_timeout_s` and pass it to every
  `VerificationWorker` they build. The CLI's `--test-timeout-s`
  (`KA_TEST_TIMEOUT` on the make targets) therefore covers optimization and
  its initial correctness check instead of upstream's fixed 30 seconds

- `0008-rtx-5090-specs.patch` — adds the desktop RTX 5090
  (`NVIDIA GeForce RTX 5090`, Blackwell, sm_120) to the shipped GPU specs
  catalog, keyed by its exact CUDA/nvidia-smi device name, so the optimizer
  uses real peak bandwidth/FLOPs figures by default instead of the name-only
  fallback. `KA_GPU_SPECS_FILE` still takes precedence when overriding those
  figures

The Python dependency set is locked
(`containerfiles/kernelagent/requirements.lock`, generated from
`requirements.in` with the command in that file's header): torch (the
CUDA 12.9 build), triton, the profiling and notebook stack, and every
KernelAgent runtime dependency, all at the exact versions the image was
tested with — a rebuild cannot silently pick up a newer release. The
committed lock is generated and tested for CUDA 12.9
(`CUDA_VERSION=12-9`), Python 3.12, linux x86_64. Building for another
CUDA or Python version needs a matching lock; see
[AGENTS.md](../../AGENTS.md) for the regeneration procedure and the
build-failure behavior.

What went into a specific image is recorded in the image labels, and
the exact patch set and dependency lock live in this repository:

```bash
podman inspect triton-profiling/cuda-profiling:12-9 \
  --format '{{ index .Labels "kernelagent-rev" }}'
```

The image build itself validates the result
(`containerfiles/kernelagent/check_install.py` runs in the final stage):
the installed package imports cleanly, its prompt and NCU templates and
RAG optimization database resolve from the install location, a
shipped catalog covers this repository's target laptop GPU and the desktop
RTX 5090 with real peak figures, a supplemental `KA_GPU_SPECS_FILE` entry
round-trips through the
installed catalog with its figures reaching the optimization and
bottleneck-analysis prompts, the
`KA_REASONING_EFFORT` knob gates and validates in the installed
providers, no
builder-source paths leaked into site-packages, and the CLI helpers and
problem specs are in place. A broken KernelAgent revision, patch, or
dependency set fails the build instead of a user's first generation run.

### Updating the pinned revision or dependencies

The maintainer procedure for bumping `KERNELAGENT_REV`, adapting the
compatibility patches, and regenerating or revalidating the dependency lock
is in [AGENTS.md](../../AGENTS.md).

The image builds against CUDA 12.9 (`CUDA_VERSION`) — the same toolkit
variable, image, and tag the CUDA blog targets use, so one build serves
every NVIDIA workflow:

```bash
make kernelagent-image CUDA_VERSION=12-8   # needs a lock generated for cu128
```

### 1. Smoke-test the provider

```bash
make kernelagent-test OPENAI_MODEL=gpt-5.2
```

Prints `VERIFY_OK` when the model routes to the right provider and a
minimal completion round-trips.

### 2. Generate the kernels

```bash
make kernelagent-generate KA_PROBLEM=matmul_bias OPENAI_MODEL=gpt-5.2
make kernelagent-generate KA_PROBLEM=flash_attention OPENAI_MODEL=claude-sonnet-4-6
```

`KA_PROBLEM` selects the problem (default `matmul_bias`). `KA_WORKERS` is the
number of initial kernel candidates the agent generates and verifies in
parallel (candidates and workers are coupled), and `KA_GEN_ROUNDS` is the
refinement-round budget:

```bash
make kernelagent-generate KA_PROBLEM=matmul_bias OPENAI_MODEL=gpt-5.2 KA_WORKERS=2 KA_GEN_ROUNDS=8
```

Outputs land under `results/<problem>/`:

- `final_kernel.py` — the verified kernel
- `generate_result.json` — model, rounds, elapsed time
- `session_*/` — full KernelAgent session (seeds, worker logs, round history)
- copies of `problem.txt`, `test_fixed.py`, `bench.py`, `spec.py`, and the
  shared `_harness.py` used for the run

### 3. Inspect the generated kernel

Read `results/<problem>/final_kernel.py` before trusting it. The fixed test
checks numerical correctness, shapes, and dtypes; the Triton-only compute
and flash-attention algorithm requirements are only enforceable by review —
see the review checklist in [AGENTS.md](../../AGENTS.md).

### 4. Verify and benchmark

Re-run the fixed test and the benchmark against one problem's saved kernel
at any time — no LLM needed. `KA_PROBLEM` selects the problem (default
`matmul_bias`), and `KA_IMPL` selects which saved kernel to verify: `kernel`
(the generated `final_kernel.py`, default) or `optimized`
(`optimized_kernel.py`, see the next section):

```bash
make kernelagent-verify KA_PROBLEM=matmul_bias
make kernelagent-verify KA_PROBLEM=matmul_bias KA_IMPL=optimized
cat results/matmul_bias/verify.json
cat results/matmul_bias/verify_optimized.json
```

Baselines: `torch.addmm` for matmul+bias, torch SDPA for attention.

Each record (`verify.json` for the generated kernel,
`verify_optimized.json` for the optimized one) records per-case latency
and speedup, the SHA-256 hash of the verified kernel, and an overall
`success` flag; it describes exactly the artifact on disk and is dropped
whenever that verification restarts or that artifact is replaced. Failed
attempts are recorded too, with the failing stage (`test` or `benchmark`)
and reason. Publishing a new kernel and verifying the saved one serialize
on a per-problem lock file under the results directory
(`results/<problem>/.results.lock`), so concurrent runs cannot interleave;
kernel files are also atomically replaced, so readers such as the profiler
that do not take the lock see a complete old or new file. A generation
publishes `final_kernel.py` and its `generate_result.json` summary in that
same critical section (an optimization does the same for
`optimized_kernel.py`, its record, and `optimize_result.json`), so the
shared summary always describes the kernel currently on disk; a failed
run replaces the shared summary under the lock while leaving the saved
kernel untouched.

### 5. Optimize a saved kernel

KernelAgent's optimizer refines a *generated* kernel with hardware
feedback: it NCU-profiles the kernel, asks the LLM to rewrite it against
the top bottlenecks, re-verifies with the fixed test, and benchmarks —
for several rounds. Optimization targets one benchmark case (`KA_CASE`) of
one problem:

```bash
make kernelagent-optimize KA_PROBLEM=matmul_bias KA_CASE=3 OPENAI_MODEL=gpt-5.2
```

`KA_OPT_ROUNDS` is the optimization-round budget. `KA_WORKERS` sets how many
rewrite candidates are explored in parallel each round: 1 (default) runs
the greedy strategy; more fan each round out across that many ranked
bottlenecks of the current best kernel:

```bash
make kernelagent-optimize KA_PROBLEM=flash_attention KA_CASE=3 KA_WORKERS=2 KA_OPT_ROUNDS=6 \
  OPENAI_MODEL=claude-sonnet-4-6
```

GPU selection follows `CUDA_VISIBLE_DEVICES`. Select whole GPUs by numeric
index or GPU UUID (full UUID or unambiguous prefix); MIG instance
selectors are rejected. The whole selection is resolved and validated,
then the session runs on **one target GPU — the first selected device**:
the optimizer ranks candidate kernels by raw latency and shares timings,
profiling data, and hardware specifications across its workers and
rounds, so measurements from mixed GPU models would corrupt the search
(the final benchmark gate cannot recover candidates a mismatched ranking
discarded). `KA_WORKERS` is unaffected: it sets the search width, not the
GPU count. A numeric index is an ordinal of CUDA's own device ordering —
which can differ from `nvidia-smi`'s — so `generate-kernel.py` probes
CUDA in an isolated subprocess before any worker forks (the parent
process never initializes CUDA) whenever the selection contains one: the
probe validates every index and names the GPU it actually selects, so
the optimizer's hardware specifications always describe the target the
workers pin; UUID-only selections skip the probe. UUID selectors are
resolved against `nvidia-smi` to their full UUID and remain UUIDs when
the worker pins its own `CUDA_VISIBLE_DEVICES`, which sidesteps the
ordering mismatch entirely. Selectors that resolve to the same physical
GPU — a numeric index alongside its UUID, for example — collapse to one
entry, keeping the recorded selection exact. `KA_GPU_NAME` overrides
only the name used to look up GPU specifications; it does not change
which GPU workers use, and it names the GPU when neither the CUDA probe
nor `nvidia-smi` is available.

The optimizer drives kernels through a Python problem module instead of
the text spec, so each session starts by generating a self-contained
`problem.py` pinned to the selected case (under
`results/<problem>/optimize_session_*/`): a stateless `Model` — the
PyTorch baseline — plus `get_inputs()` matching the `kernel_function`
argument order. Its docstring embeds `problem.txt` with `@GPU_TARGET@`
resolved, so rewrites keep honoring every written requirement, including
the Triton-only compute rule and the tuning target for this run's GPU.

Acceptance is deliberately stricter than the optimizer's own timing: a
candidate is saved only when it passes the **full** fixed test, completes
the benchmark, and is **strictly faster on the selected case** than the
source kernel — both sides measured by the fixed `bench.py`, the only
timing that runs the exact shapes and dtype. (The optimizer's internal
timings approximate the case and may run a dtype inferred from the kernel
source.)

Accepted results are kept **separate** from the generated kernel:

- `results/<problem>/optimized_kernel.py` — the accepted kernel
- `results/<problem>/verify_optimized.json` — its verification record
  (kernel hash, full benchmark data)
- `results/<problem>/optimize_result.json` — the latest completed attempt's
  summary (model, strategy, rounds, per-case timings, speedup); successful
  runs publish it atomically under the results lock with the kernel and
  verification record it describes
- `results/<problem>/optimize_session_*/` — the full optimizer session
  (NCU profiles, worker logs, program database), including that run's own
  `optimize_result.json`

A failed attempt — optimizer error, failed test, or no speedup — leaves
any previously accepted `optimized_kernel.py` and its record untouched;
the shared summary then describes the failed attempt, while each session's
summary preserves its own outcome. As with generation, review an accepted
kernel before trusting it — see the review checklist in
[AGENTS.md](../../AGENTS.md).

Verify or profile the optimized kernel with `KA_IMPL=optimized` (or the
explicit target below):

```bash
make kernelagent-verify KA_PROBLEM=matmul_bias KA_IMPL=optimized
make kernelagent-profile KA_PROBLEM=matmul_bias KA_CASE=3 KA_IMPL=optimized KA_PROFILER=proton
make kernelagent-profile-optimized KA_PROBLEM=matmul_bias KA_CASE=3
```

### 6. Profile

Profile one benchmark case of the generated kernel, the optimized kernel,
or the PyTorch baseline (`KA_IMPL=kernel|optimized|ref`) with Proton, Nsight
Systems, or Nsight Compute:

```bash
make kernelagent-profile KA_PROBLEM=matmul_bias KA_CASE=3 KA_PROFILER=proton
make kernelagent-profile KA_PROBLEM=matmul_bias KA_CASE=3 KA_IMPL=ref KA_PROFILER=proton
make kernelagent-profile KA_PROBLEM=matmul_bias KA_CASE=3 KA_IMPL=optimized KA_PROFILER=proton
make kernelagent-profile KA_PROBLEM=flash_attention KA_CASE=3 KA_PROFILER=nsys
make kernelagent-profile KA_PROBLEM=matmul_bias KA_CASE=0 KA_PROFILER=ncu
```

Any other contract-compliant kernel file can be profiled the same way
with `KA_KERNEL_SOURCE` and `KA_LABEL` instead of `KA_IMPL` (see
[External baseline kernels](#external-baseline-kernels)):

```bash
make kernelagent-profile KA_PROBLEM=matmul_bias KA_CASE=0 KA_PROFILER=proton \
  KA_KERNEL_SOURCE=matmul_bias/baselines/rocm_blog_fusion.py KA_LABEL=rocm_blog_fusion
```

`KA_KERNEL_SOURCE` accepts the same host, problems-relative, and container
paths as the baseline flow below; a host file outside the mounted trees is
copied into `results/.kernel-sources/` for the run.

For a before/after comparison of the two saved kernels, use the explicit
targets — they fix `KA_IMPL` for you, `kernelagent-profile-unoptimized`
profiling the generated `final_kernel.py` and
`kernelagent-profile-optimized` the accepted `optimized_kernel.py`. They
take the same `KA_PROBLEM`, `KA_CASE`, `KA_PROFILER`, `KA_PROFILE_ITERS`,
`KA_PROFILE_WARMUP`, and `KA_NCU_SET` variables (and no `KA_IMPL`):

```bash
make kernelagent-profile-unoptimized KA_PROBLEM=matmul_bias KA_CASE=3
make kernelagent-profile-optimized KA_PROBLEM=matmul_bias KA_CASE=3

# Nsight Compute comparison of the same pair (small --iters; ncu replays
# every profiled launch)
make kernelagent-profile-unoptimized KA_PROBLEM=matmul_bias KA_CASE=3 KA_PROFILER=ncu KA_PROFILE_ITERS=3 KA_NCU_SET=detailed
make kernelagent-profile-optimized KA_PROBLEM=matmul_bias KA_CASE=3 KA_PROFILER=ncu KA_PROFILE_ITERS=3 KA_NCU_SET=detailed
```

`kernelagent-profile-optimized` needs a completed optimization first (see
the previous section); both targets need the generated kernel.

Reports land under `profiles/<problem>/` as
`<impl>_<profiler>_case<N>.{hatchet,nsys-rep,ncu-rep}` with a `.json`
metadata file recording the case, GPU, software versions, and kernel
identity — `kernel_`-prefixed for the generated kernel (the unoptimized
target), `optimized_`-prefixed for the optimized one, `ref_` for the
PyTorch baseline, `<label>_`-prefixed for a `KA_KERNEL_SOURCE`/`KA_LABEL` run. View
Proton's `.hatchet` output with `proton-viewer`, or open the
`.nsys-rep` / `.ncu-rep` reports in the Nsight UIs launched from this
workspace's own container — the reports appear at
`/workspace/user/profiles/<problem>/`:

```bash
make kernelagent-systems
make kernelagent-compute
```

With a custom `KERNELAGENT_WORKSPACE`, the UI targets mount that path, so
the reports follow automatically.

The runner warms up JIT compilation and autotuning first, then delimits the
measured region with `cudaProfilerStart`/`Stop`, so Nsight captures only the
steady-state kernel launches. With `ncu`, prefer a small `--iters` (make
variable `KA_PROFILE_ITERS`), since Nsight Compute replays every profiled
launch, and a lean metric set (make variable `KA_NCU_SET`, default `basic`;
use `detailed` or `full` for deeper analysis):

```bash
make kernelagent-profile KA_PROBLEM=matmul_bias KA_PROFILER=ncu KA_PROFILE_ITERS=3 KA_NCU_SET=full
```

Proton prints a per-kernel timing tree when the run finishes; compare the
`kernel`, `optimized`, and `ref` captures to see where the generated and
optimized kernels spend their time relative to the library baseline.

## Comparison experiments

The single-kernel workflow above optimizes one saved kernel and replaces
`optimized_kernel.py` on each accepted run. Comparing optimization
settings — for a webinar, a blog post, or a tuning study — needs every
variant kept side by side, which is what the experiment workflow does:

```bash
make kernelagent-experiment KA_EXPERIMENT=webinar-v1 \
  KA_CASE=3 KA_OPT_ROUNDS="5 10" KA_PROFILER=ncu
```

One experiment generates **one baseline kernel per problem**, then
optimizes that exact same source **once per budget in `KA_OPT_ROUNDS`**
(each budget's staged baseline is hash-checked against the generated
kernel), verifies every kernel with the fixed test and benchmark, and
captures every verified variant with every requested profiler
(`KA_PROFILER`, any of `ncu nsys proton`). With no `KA_EXPERIMENT` the runner
picks a unique id and prints it; an existing id is never overwritten.

| Variable | Meaning |
|---|---|
| `KA_EXPERIMENT` | Experiment id (auto-generated when omitted at creation) |
| `KA_PROBLEM` | Problems in the experiment: one or more space-separated names (default: every problem) |
| `KA_CASE` | The benchmark case every stage targets |
| `KA_GEN_ROUNDS` | Refinement budget for the **baseline generation** |
| `KA_OPT_ROUNDS` | Space-separated optimization budgets, e.g. `"5 10"` |
| `KA_GENERATION_MODEL`, `KA_OPTIMIZATION_MODEL` | Per-stage LLM spec as `OPENAI_MODEL[:EFFORT]` (default: the run-wide `OPENAI_MODEL` and `KA_REASONING_EFFORT`; each stage's backend follows its model id) |
| `KA_REASONING_EFFORT` | Run-wide reasoning level for both stages' models (low / medium / high; empty = model default) |
| `KA_WORKERS` | Search width for generation and optimization |
| `KA_PROFILER` | Capture profilers: any of `ncu nsys proton` (default `proton`) |
| `KA_PROFILE_ITERS`, `KA_PROFILE_WARMUP` | Measured/warmup iterations per capture, shared by every profiler (ncu replays every launch, so prefer a small `KA_PROFILE_ITERS` such as 3 for ncu captures) |
| `KA_NCU_SET`, `KA_TEST_TIMEOUT` | ncu metric set; per-test subprocess timeout |

Both stages use the run-wide `OPENAI_MODEL` (with its backend inferred from the
model id) unless overridden. Each stage takes one `OPENAI_MODEL[:EFFORT]` spec;
the suffix is a reasoning effort only when it is `low`, `medium` or
`high`, so colon-bearing model ids (Ollama tags, local-server ids) pass
through whole, and an omitted model or effort falls back to the run-wide
`OPENAI_MODEL` / `KA_REASONING_EFFORT`. The backend follows the model id, so a
per-stage `claude-*` model puts that stage on the Anthropic API in an
otherwise-OpenAI run. Only an explicitly supplied `OPENAI_BASE_URL`
reroutes `claude-*`; the public URL the container defaults to does not
count:

```bash
make kernelagent-experiment KA_EXPERIMENT=mixed-v1 KA_CASE=3 \
  KA_GENERATION_MODEL=gpt-6.1-sol \
  KA_OPTIMIZATION_MODEL=claude-sonnet-4-6:high
```

When `OPENAI_MODEL` is empty, set both stages — the Makefile checks that before
starting.

Experiment runs forward both providers' credentials into the container,
so either direction works; the manifest and report record the resolved
backend and model per stage. Smoke-test each model you use first
([the provider check](#1-smoke-test-the-provider) with `OPENAI_MODEL=...`) — a
stage whose backend has no usable credential or model aborts the run
before anything is created.

Everything the experiment produced is preserved under
`results/experiments/<id>/`:

```text
results/experiments/webinar-v1/
  manifest.json            Authoritative index: config, spec and
                           shared-harness hashes, outcomes, capture batches
                           (paths relative to this root)
  environment.json         GPU, driver, CUDA, torch, triton, profilers,
                           KernelAgent, repository revision
  specs/                   Snapshot of the problem specs the run used
  sources/                 Host kernel source copied in for an
                           add-baseline run (the archived copy lives under
                           problems/<P>/baselines/<label>/)
  problems/matmul_bias/
    generated/matmul_bias/         final_kernel.py, verify.json, session_*/
    case_3/rounds_5/matmul_bias/   staged baseline final_kernel.py beside
                                   optimized_kernel.py, optimize_result.json,
                                   verify_optimized.json, optimize_session_*/
    case_3/rounds_10/matmul_bias/  …same, independent run from the same source
    baselines/rocm_blog_fusion/matmul_bias/
                                   external baselines added after the run:
                                   archived kernel.py beside
                                   verify_rocm_blog_fusion.json
  captures/<capture-id>/matmul_bias/
    generated_ncu_case3.ncu-rep    one stem per variant and profiler, with
    rounds_5_ncu_case3.ncu-rep     .details.csv/.raw.csv (ncu) or
    rounds_10_ncu_case3.ncu-rep    nsys stats CSVs exported beside them
  reports/
    comparison.md                 The rendered comparison report
    comparison.json               The same data, machine-readable
    source-diffs/                 Pairwise diffs of the archived kernels
```

Failure semantics are the experiment's core contract: a failed
generation, a crashed optimization, a candidate that did not beat its
source, or a kernel that fails verification is **recorded and reported**
— never silently replaced by another variant's kernel — and the run
continues with the remaining budgets and problems. Captures profile only
verified sources, and a capture batch refuses an archived kernel whose
hash no longer matches its record. `KA_GEN_ROUNDS` and each `KA_OPT_ROUNDS`
budget are maxima; the report shows completed rounds against the
requested budget. An experiment is otherwise immutable: its problem set,
case, budgets, and LLM configuration are fixed at creation. Retrying or
extending an optimization means creating a new experiment id; profiling
archives can still be added to the saved experiment (see
[Additional capture batches](#additional-capture-batches)).

### The report

The report is rendered offline from the saved artifacts — no GPU, no
LLM, no container (host Python 3 standard library only):

```bash
make kernelagent-report KA_EXPERIMENT=webinar-v1

# Write it straight into a blog or slides tree (links follow the file)
make kernelagent-report KA_EXPERIMENT=webinar-v1 KA_REPORT_OUT=slides/comparison.md

# Machine-readable data instead of Markdown
python3 workspace/KernelAgent/scripts/report-experiment.py webinar-v1 \
  --results-root workspace/KernelAgent/results --format json
```

It contains, per problem: the fixed-benchmark comparison table (latency,
speedup vs the generated kernel and the PyTorch baseline, completed
rounds, correctness and acceptance status), the optimizer's own
acceptance timings, one section per profiler (ncu duration and SOL
metrics aggregated over the profiled launches, nsys kernel and CUDA API
summaries, Proton timing entries), artifact links (kernel sources,
records, native captures, source diffs), and explicit failure rows.
Profiler timings stay in their own sections — they measure captured
launches, not the benchmark's `do_bench` means — and missing metrics
render as `—`.

### Additional capture batches

Add captures to a saved experiment without repeating any LLM work:

```bash
make kernelagent-experiment-profile KA_EXPERIMENT=webinar-v1 \
  KA_PROFILER="nsys proton" KA_PROFILE_ITERS=20
```

Each invocation appends one capture batch (the report defaults to the
latest; `make kernelagent-report KA_CAPTURE_BATCH=<id>` selects another)
and re-profiles the archived, verified kernels only. Captures can be
added only to a terminal experiment (`completed`, `partial`, or
`failed`); let a run still in progress finish first.

### External baseline kernels

An experiment's variants all descend from one LLM-generated kernel. To
compare a hand-written kernel against them — one lifted from a blog
post or a library, or from your own tuning study — add it as an
**external baseline**. The repository ships two: the ROCm blog's fused
matmul + bias kernel, adapted to the problem contract at
`problems/matmul_bias/baselines/rocm_blog_fusion.py`, and vLLM v0.32.0's
Triton prefill-attention kernel at
`problems/flash_attention/baselines/vllm_triton_prefill.py`.

An external kernel must satisfy the same contract as a generated one —
a top-level `kernel_function(A, B, bias)` computing `(A @ B) + bias`
with the output dtype following `A`, across the fp16/bf16 shapes the
fixed test and benchmark run — because it is verified by exactly that
test and benchmark. Adapting the blog kernel took two changes: the
wrapper renamed to `kernel_function`, and the output dtype made to
follow the inputs (the blog hardcodes fp16, which fails the bf16
cases). The kernel body, autotune config space, and launch metadata
are unchanged, so the performance comparison stays honest.

The vLLM baseline is already a contiguous flash-attention forward with
online softmax, causal and non-causal, and arbitrary sequence lengths, so
it needs only vLLM-import/platform shims and a wrapper. The wrapper maps
the problem's `(B, H, S, D)` tensors onto the kernel's `[b*s, head, d]`
layout with a zero-copy view (batch and heads are flattened into the
kernel's batch axis with head = 1). The kernel body, masking, block-size
heuristic, and launch parameters are unchanged.

Add it to a saved experiment — one in a terminal state (`completed` or
`partial`); no LLM is involved:

```bash
make kernelagent-experiment-baseline KA_EXPERIMENT=webinar-v1 \
  KA_PROBLEM=matmul_bias KA_LABEL=rocm_blog_fusion \
  KA_KERNEL_SOURCE=matmul_bias/baselines/rocm_blog_fusion.py
```

The flash-attention baseline is added the same way. Matching the
experiment's existing captures keeps the report comparable:

```bash
make kernelagent-experiment-baseline KA_EXPERIMENT=flashattention-ada5000_v1 \
  KA_PROBLEM=flash_attention KA_LABEL=vllm_triton_prefill \
  KA_KERNEL_SOURCE=flash_attention/baselines/vllm_triton_prefill.py \
  KA_PROFILER=ncu KA_NCU_SET=full
```

`KA_KERNEL_SOURCE` is a host path (absolute or relative to the repository
root), a path relative to the problems mount, or a `/workspace/user/...`
path copied from inside the container. A host file outside the container
mounts is copied into the experiment's `sources/` directory and that
container path is passed to the verifier; the original path is kept as
the baseline's source. A missing path is rejected before the container
starts. `KA_LABEL` names the baseline everywhere it appears —
manifest, capture stems, report (letters, digits, `.`, `_`, `-`; it
must not collide with a variant name or the reserved `kernel`,
`optimized`, `ref` stems). The command archives and hashes the kernel
inside the experiment, verifies the archived copy against the
experiment's problem snapshot, records it in the manifest, and — with
the default `KA_PROFILER=proton` — captures it in its own batch;
`KA_PROFILER=` (empty) records the verified baseline without profiling
it. A kernel that fails verification records nothing. Re-running with the
same source and `KA_LABEL` is a no-op; a different source under an
existing label is verified first, so a kernel that does not meet the
problem contract reports that failure before the label collision, and the
recorded baseline is left untouched. Use a new label (or a new
experiment) to add a second kernel.

From there the baseline flows through the rest of the workflow for
free: `kernelagent-experiment-profile` captures it alongside the
variants, and the report renders it as a row marked
*(external baseline)* with its benchmark numbers, its source path, and
source diffs against every other kernel in the experiment.

The same kernel can be inspected outside any experiment:

```bash
# Inside the container (make kernelagent-console): verify + benchmark
# any contract-compliant kernel file. The record lands beside — never
# over — the generated kernel's verify.json, as
# results/matmul_bias/verify_rocm_blog_fusion.json:
generate-kernel.py verify matmul_bias \
  --kernel /workspace/user/problems/matmul_bias/baselines/rocm_blog_fusion.py \
  --label rocm_blog_fusion --test-timeout-s 600

# Standalone profiling (KA_KERNEL_SOURCE takes the same paths):
make kernelagent-profile KA_PROBLEM=matmul_bias KA_CASE=0 KA_PROFILER=proton \
  KA_KERNEL_SOURCE=matmul_bias/baselines/rocm_blog_fusion.py KA_LABEL=rocm_blog_fusion
```

The blog kernel re-tunes its large (MI300X-sized) autotune space for
every shape the test and benchmark run, so give its verification a
generous timeout — `--test-timeout-s 600` above, or
`KA_TEST_TIMEOUT=600` on the make targets.

## Adapting the problems

The final requirement in each `problem.txt` requires wrapping the kernel in
`@triton.autotune` with a small `triton.Config` space sized for the target
GPU. It carries an `@GPU_TARGET@` token, resolved at generate and optimize
time to the detected GPU's name, memory, compute capability, catalogued
architecture, and `sm_<major><minor>` — for example `NVIDIA GeForce RTX 5090
(32GB, compute capability 12.0, Blackwell, sm_120)`. A GPU absent from the
optimizer's specs catalog gets the same description without the architecture.
`KA_GPU_TARGET` replaces the whole description, which is how you state details
the detector cannot report (the chip codename) or target hardware you are not
running on:

```bash
make kernelagent-generate KA_PROBLEM=matmul_bias KA_GPU_TARGET="NVIDIA RTX 5000 Ada Generation \
Laptop GPU (AD103, 16GB, compute capability 8.9, sm_89)"
```

The optimizer's generated `problem.py` embeds the resolved text, so the
requirement steers optimization rewrites; the `problem.txt` copied beside each
published kernel records the resolved text. The pinned KernelAgent generation
template does not interpolate `problem_description`, so this text does not
reach the initial generator — generation follows KernelAgent's own
`triton_guidelines.j2`, which recommends `@triton.autotune`.

The benchmark cases in `bench.py` and the tolerance checks in
`test_fixed.py` are hand-written; extend them (and re-run
`make kernelagent-verify KA_PROBLEM=<problem>`) to cover shapes you care
about. Optimization
targets one case at a time (`KA_CASE` selects it), so the cases you add are
the cases you can optimize for. A hand-written kernel you want compared
against the generated ones follows the same contract — see
[External baseline kernels](#external-baseline-kernels).

## Troubleshooting

**Nsight Compute prints `LookupError: unknown encoding: utf-8-sig` when
closing the UI** — rebuild with `make kernelagent-image`, then relaunch
`make kernelagent-compute`. The image removes the base Python setuptools
`.pth` hook and installs `nsys-ui`/`ncu-ui` launchers that isolate Nsight's
embedded interpreter from the workspace venv's `.pth` files. The launchers
live on `PATH`, so the same fix applies when you type `nsys-ui` or `ncu-ui`
in an interactive container.

**`ERROR: OPENAI_MODEL is required ...`** — set `OPENAI_MODEL` to a public model id or,
for a local server, the id it reports at `/v1/models` (plus
`OPENAI_BASE_URL`).

**Provider errors against a local server** — make sure `OPENAI_API_KEY` is
set to any non-empty value and `OPENAI_BASE_URL` uses
`host.containers.internal` (not `127.0.0.1`, which refers to the container
itself).

**`ERROR: no provider for model ...`** — the API key for the selected
backend is missing or empty.

**`ERROR: could not detect the GPU name ...`** — the optimizer learns
the GPU name before any CUDA context exists in the parent process (it
forks workers): numeric selections are probed in an isolated CUDA
subprocess and UUID selections through `nvidia-smi`. When neither can
run or name the selected device (for example, the probe subprocess
failed), the numeric selection still reaches the workers and the name
must be passed explicitly:

```bash
make kernelagent-optimize KA_PROBLEM=matmul_bias KA_CASE=3 \
  KA_GPU_NAME="NVIDIA RTX 5000 Ada Generation Laptop GPU" OPENAI_MODEL=gpt-5.2
```

**`ERROR: CUDA_VISIBLE_DEVICES=...`** — the optimizer could not resolve
the selection. Use a numeric GPU index or a full GPU UUID; unknown or
ambiguous UUIDs, indices beyond the CUDA device count, and MIG instance
selectors are rejected rather than widening the worker selection.

**Optimizer logs `Unknown GPU: ...`** — upstream's GPU specs catalog only
covers a handful of datacenter and GeForce parts. The compatibility patches
add this repository's target GPUs: `0006-rtx-5000-ada-laptop-specs.patch`
adds the laptop (`NVIDIA RTX 5000 Ada Generation Laptop GPU`) and
`0008-rtx-5090-specs.patch` adds the desktop RTX 5090 under the exact key
`NVIDIA GeForce RTX 5090` (the `GeForce` is part of the CUDA/nvidia-smi
device name, and lookups are exact-name membership). Any other GPU missing
from the catalog still optimizes: the
`0003-unknown-gpu-specs.patch` compatibility patch proceeds with
name-only specs, so the bottleneck-analysis prompts carry no peak
bandwidth/FLOPs numbers for that GPU. To give the optimizer real figures
for an uncatalogued GPU — or to override a shipped entry — write a JSON
file mapping its device-reported name to a specs dict shaped like a
catalog entry and pass it as `KA_GPU_SPECS_FILE` — the make targets mount
it into the container read-only and the optimizer merges it over the
shipped catalog (a supplemental entry takes precedence over a shipped
one). For the target laptop GPU, whose figures now ship via 0006 (the RTX
5090 figures ship via 0008):

```bash
cat > ~/gpu_specs.json <<'EOF'
{
  "NVIDIA RTX 5000 Ada Generation Laptop GPU": {
    "name": "NVIDIA RTX 5000 Ada Generation Laptop GPU",
    "architecture": "Ada Lovelace",
    "peak_fp32_tflops": 42.6,
    "peak_fp16_tflops": 42.6,
    "peak_bf16_tflops": 42.6,
    "peak_memory_bw_gbps": 576,
    "sm_count": 76,
    "max_threads_per_sm": 1536,
    "l1_cache_kb": 128,
    "l2_cache_mb": 64,
    "memory_gb": 16,
    "memory_type": "GDDR6"
  }
}
EOF
make kernelagent-optimize KA_PROBLEM=matmul_bias KA_CASE=3 \
  KA_GPU_SPECS_FILE=~/gpu_specs.json OPENAI_MODEL=gpt-5.2
```

The key must match the name exactly as CUDA and nvidia-smi report it —
or, when `KA_GPU_NAME` overrides detection (see
`ERROR: could not detect the GPU name ...` above), the name you set
there. These example figures are boost-clock reference peaks, and laptop
power and thermal limits keep sustained performance below them. A
missing, unreadable, or malformed file only logs a warning; optimization
proceeds without the supplement.

**ncu reports permission errors** — the make targets already run privileged
with `SYS_ADMIN`; if you run `ncu` manually inside `make
kernelagent-console`, keep those flags. (The optimizer profiles every
candidate with ncu, so the same applies to
`make kernelagent-optimize`.)

**CUDA OOM during generation** — kernel tests run alongside nothing else in
this container; if you still hit OOM on a small GPU, shrink the benchmark
cases in the problem specs.

## References

- [KernelAgent](https://github.com/meta-pytorch/KernelAgent) — the kernel
  generation agent used here
- [Triton](https://github.com/triton-lang/triton) — language and compiler
  for the generated kernels; Proton is its profiler
- [Nsight Systems](https://developer.nvidia.com/nsight-systems) /
  [Nsight Compute](https://developer.nvidia.com/nsight-compute) — the
  timeline and kernel profilers used in the CUDA blog
