# Triton GPU profiling and KernelAgent

Demo materials for profiling Triton kernels on NVIDIA CUDA and AMD ROCm,
and generating, optimizing, and profiling Triton kernels with KernelAgent.

## Requirements

- Podman or Docker and `make`
- An NVIDIA GPU for CUDA and KernelAgent workflows
- An AMD GPU for ROCm workflows (ROCm Compute profiling requires CDNA GPUs)
- An LLM endpoint for KernelAgent generation and optimization only

## Build an image

```bash
make cuda-image   # CUDA profiling, Nsight, Jupyter, and KernelAgent
make rocm-image   # AMD ROCm profiling
```

The CUDA image builds KernelAgent inside the container build from a
pinned upstream commit, on a locked Python dependency set — no local
KernelAgent source is needed or staged (see the
[KernelAgent guide](workspace/KernelAgent/README.md)).

The `nsight-*` targets use the CUDA image without passing through a GPU.
`kernelagent-image` is an alias for `cuda-image`.

## Common workflows

```bash
make cuda-console
make cuda-jupyter
make cuda-systems
make cuda-compute

make rocm-console
make rocm-jupyter

make kernelagent-test OPENAI_MODEL=gpt-5.2
make kernelagent-generate-kernels OPENAI_MODEL=gpt-5.2
make kernelagent-verify-kernels
make kernelagent-optimize KA_PROBLEM=matmul_bias KA_CASE=3 OPENAI_MODEL=gpt-5.2
make kernelagent-profile KA_PROBLEM=matmul_bias KA_CASE=3 KA_PROFILER=proton
make kernelagent-profile-unoptimized KA_PROBLEM=matmul_bias KA_CASE=3
make kernelagent-profile-optimized KA_PROBLEM=matmul_bias KA_CASE=3

make kernelagent-experiment KA_EXPERIMENT=webinar-v1 KA_CASE=3 KA_OPT_ROUNDS="5 10" KA_PROFILER=ncu
make kernelagent-experiment-profile KA_EXPERIMENT=webinar-v1 KA_PROFILER="nsys proton"
make kernelagent-experiment-baseline KA_EXPERIMENT=webinar-v1 KA_PROBLEM=matmul_bias KA_LABEL=rocm_blog_fusion KA_KERNEL_SOURCE=matmul_bias/baselines/rocm_blog_fusion.py
make kernelagent-report KA_EXPERIMENT=webinar-v1
```

`OPENAI_MODEL` selects both the model and its backend (`claude-*` ids use the
Anthropic API, anything else the OpenAI API or an OpenAI-compatible local
server via `OPENAI_BASE_URL`). The default is `claude-sonnet-4-6` on both the
make command line and a bare container, so it needs `ANTHROPIC_API_KEY` unless
you point `OPENAI_BASE_URL` at a server that serves it. Add
`KA_REASONING_EFFORT=low|medium|high` to
any kernelagent target to set the selected model's reasoning level.
See the [KernelAgent guide](workspace/KernelAgent/README.md) for provider
configuration, the pinned in-image KernelAgent build, problem specs,
optimization, profiling, and the comparison-experiment workflow
(`kernelagent-experiment` / `kernelagent-report`, including external
baseline kernels such as the ROCm blog's).

Run `make help` for the complete target and variable list.
