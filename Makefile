IMAGE_REPO ?= triton-profiling
WORKSPACE ?= $(PWD)/workspace
CUDA_EPEL_VERSION ?= 9
CUDA_IMAGE_NAME ?= cuda-profiling
CUDA_VERSION ?= 12-9
CUDA_PYTHON_VERSION ?= 3.12
CUDA_VISIBLE_DEVICES ?= 0
CUDA_WORKSPACE ?= ${WORKSPACE}/CUDA
ROCM_EPEL_VERSION ?= 10
ROCM_IMAGE_NAME ?= rocm-profiling
ROCM_VERSION ?= 7.0.3
ROCM_PYTHON_VERSION ?= 3.12
ROCM_WORKSPACE ?= ${WORKSPACE}/ROCm
ROCR_VISIBLE_DEVICES ?= 0
CTR_CMD := $(or $(shell command -v podman), $(shell command -v docker))

# Container Run
define ctr_run
	$(CTR_CMD) run -it --rm \
	$(1) \
	$(IMAGE_REPO)/$(2) \
	$(3)
endef

# CUDA runtime arguments
ifeq ($(CTR_CMD),/usr/bin/podman)
	cuda_dev_args=--device nvidia.com/gpu=all
else ifeq ($(CTR_CMD),/usr/bin/docker)
	cuda_dev_args=--runtime=nvidia --gpus all
endif

# Shared NVIDIA runtime arguments (cuda-*, nsight-*, and kernelagent-*
# targets). SYS_ADMIN/privileged are required for ncu; host IPC allows
# CUDA inter-process communication.
define cuda_args
	--security-opt label=disable \
	--privileged \
	--cap-add=SYS_ADMIN \
	--ipc=host \
	-e CUDA_VERSION=$(CUDA_VERSION) \
	-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)
endef

define cuda_vol
	-v "${CUDA_WORKSPACE}:/workspace/user:Z"
endef

# Runtime Nsight arguments
define nsight_args
	-e DISPLAY=${DISPLAY} \
	-e "WAYLAND_DISPLAY=${WAYLAND_DISPLAY}" \
	-e XDG_RUNTIME_DIR=/tmp \
	-e PULSE_SERVER=${XDG_RUNTIME_DIR}/pulse/native \
	-e QT_QPA_PLATFORM=wayland-egl \
	-e WAYLAND_DISPLAY=wayland-0 \
	-v "${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY}:/tmp/${WAYLAND_DISPLAY}:ro" \
	-v "${XDG_RUNTIME_DIR}/pulse:/tmp/pulse:ro"
endef

# ROCm runtime arguments
define rocm_args
	--device=/dev/kfd \
	--device=/dev/dri \
	--cap-add=SYS_PTRACE \
	--group-add keep-groups \
	--ipc=host \
	--security-opt seccomp=unconfined \
	-e ROCM_VERSION=$(ROCM_VERSION) \
	-e ROCR_VISIBLE_DEVICES=$(ROCR_VISIBLE_DEVICES)
endef

define rocm_vol
	-v "${ROCM_WORKSPACE}:/workspace/user:Z"
endef

# Runtime Jupyter notebook arguments
define notebook_args
	-e NOTEBOOK_PORT=$(NOTEBOOK_PORT) \
	-e NOTEBOOK_DIR=/workspace/user \
	-p $(NOTEBOOK_PORT):$(NOTEBOOK_PORT)
endef


# Build the shared NVIDIA image: Triton profiling (Nsight, Proton,
# Jupyter) plus KernelAgent and its LLM provider deps. KernelAgent is
# cloned inside the build at KERNELAGENT_REV from KERNELAGENT_UPSTREAM,
# patched, built into a wheel, and installed on top of the locked Python
# dependency set — a fresh clone of this repository builds the tested
# image with no local KernelAgent source (see
# containerfiles/Dockerfile.cuda).
.PHONY: cuda-image
cuda-image: NOTEBOOK_PORT ?= 8888
cuda-image: containerfiles/Dockerfile.cuda
	$(CTR_CMD) build \
	--build-arg "CUDA_VERSION=$(CUDA_VERSION)" \
	--build-arg "EPEL_VERSION=$(CUDA_EPEL_VERSION)" \
	--build-arg "KERNELAGENT_REV=$(KERNELAGENT_REV)" \
	--build-arg "KERNELAGENT_UPSTREAM=$(KERNELAGENT_UPSTREAM)" \
	--build-arg "NOTEBOOK_PORT=$(NOTEBOOK_PORT)" \
	--build-arg "PYTHON_VERSION=$(CUDA_PYTHON_VERSION)" \
	-t $(IMAGE_REPO)/$(CUDA_IMAGE_NAME):$(CUDA_VERSION) \
	-f $< .


# Run the NVIDIA Nsight Systems UI (with no-GPU support)
.PHONY: nsight-systems
nsight-systems: cuda-image
	$(call ctr_run,$(cuda_args)$(cuda_vol)$(nsight_args),$(CUDA_IMAGE_NAME):$(CUDA_VERSION),nsys-ui)

# Run the NVIDIA Nsight Compute UI (with no-GPU support)
.PHONY: nsight-compute
nsight-compute: cuda-image
	$(call ctr_run,$(cuda_args)$(cuda_vol)$(nsight_args),$(CUDA_IMAGE_NAME):$(CUDA_VERSION),ncu-ui)

# Run a Jupyter Notebook Server (with no-GPU support)
.PHONY: nsight-jupyter
nsight-jupyter: NOTEBOOK_PORT ?= 8888
nsight-jupyter: cuda-image
	$(call ctr_run,$(cuda_args)$(cuda_vol)$(nsight_args)$(notebook_args),$(CUDA_IMAGE_NAME):$(CUDA_VERSION),start_jupyter)

# Open a shell in the NVIDIA Nsight container (with no-GPU support)
.PHONY: nsight-console
nsight-console: NOTEBOOK_PORT ?= 8889
nsight-console: cuda-image
	$(call ctr_run,$(cuda_args)$(cuda_vol)$(nsight_args)$(notebook_args),$(CUDA_IMAGE_NAME):$(CUDA_VERSION),)


# Run the NVIDIA Nsight Systems UI (with GPU support)
.PHONY: cuda-systems
cuda-systems: cuda-image
	$(call ctr_run,$(cuda_dev_args)$(cuda_args)$(cuda_vol)$(nsight_args),$(CUDA_IMAGE_NAME):$(CUDA_VERSION),nsys-ui)

# Run the NVIDIA Nsight Systems UI (with GPU support)
.PHONY: cuda-compute
cuda-compute: cuda-image
	$(call ctr_run,$(cuda_dev_args)$(cuda_args)$(cuda_vol)$(nsight_args),$(CUDA_IMAGE_NAME):$(CUDA_VERSION),ncu-ui)

# Run a Jupyter Notebook Server (with GPU support)
.PHONY: cuda-jupyter
cuda-jupyter: NOTEBOOK_PORT ?= 8888
cuda-jupyter: cuda-image
	$(call ctr_run,$(cuda_dev_args)$(cuda_args)$(cuda_vol)$(nsight_args)$(notebook_args),$(CUDA_IMAGE_NAME):$(CUDA_VERSION),start_jupyter)

# Open a shell in the NVIDIA Nsight container (with GPU support)
.PHONY: cuda-console
cuda-console: NOTEBOOK_PORT ?= 8889
cuda-console: cuda-image
	$(call ctr_run,$(cuda_dev_args)$(cuda_args)$(cuda_vol)$(nsight_args)$(notebook_args),$(CUDA_IMAGE_NAME):$(CUDA_VERSION),)


.PHONY: rocm-image
rocm-image: NOTEBOOK_PORT ?= 8888
rocm-image: containerfiles/Dockerfile.rocm
	$(CTR_CMD) build \
	--build-arg "EPEL_VERSION=$(ROCM_EPEL_VERSION)" \
	--build-arg "ROCM_VERSION=$(ROCM_VERSION)" \
	--build-arg "NOTEBOOK_PORT=$(NOTEBOOK_PORT)" \
	--build-arg "PYTHON_VERSION=$(ROCM_PYTHON_VERSION)" \
	-t $(IMAGE_REPO)/$(ROCM_IMAGE_NAME):$(ROCM_VERSION) \
	-f $< .


# Run a Jupyter Notebook Server (with GPU support)
.PHONY: rocm-jupyter
rocm-jupyter: NOTEBOOK_PORT ?= 8888
rocm-jupyter: rocm-image
	$(call ctr_run,$(rocm_args)$(rocm_vol)$(notebook_args),$(ROCM_IMAGE_NAME):$(ROCM_VERSION),start_jupyter)

# Open a shell in the AMD ROCm container (with GPU support)
.PHONY: rocm-console
rocm-console: NOTEBOOK_PORT ?= 8889
rocm-console: rocm-image
	$(call ctr_run,$(rocm_args)$(rocm_vol)$(notebook_args),$(ROCM_IMAGE_NAME):$(ROCM_VERSION),)


# KernelAgent — agentic Triton kernel generation (third blog).
#
# CUDA-only workflow running in the shared CUDA image (see cuda-image):
# generate Triton kernels with KernelAgent from the fixed problem specs
# in workspace/KernelAgent/problems, then verify, benchmark, optimize,
# and profile them with Proton / Nsight. The kernelagent-* targets mount
# the KernelAgent workspace and select the KernelAgent entrypoint
# workflow (LLM backend setup); see workspace/KernelAgent/README.md for
# the full guide.
#
# LLM (only needed for generation, optimization, and the smoke test):
#   OPENAI_MODEL selects the model and its backend — claude-* ids use the
#                         Anthropic API (ANTHROPIC_API_KEY); everything
#                         else the OpenAI API or any OpenAI-compatible
#                         server (OPENAI_API_KEY; OPENAI_BASE_URL for
#                         local servers, which also reroutes claude-* ids
#                         to that server). The default is
#                         claude-sonnet-4-6, matching the entrypoint.
#                         This is also the container's
#                         model variable, so it keeps its provider name.
#   KA_REASONING_EFFORT (low | medium | high, optional) sets the model's
#                         reasoning level; experiments override it per
#                         stage in KA_GENERATION_MODEL /
#                         KA_OPTIMIZATION_MODEL
# Pinned upstream KernelAgent source, cloned and verified inside the
# image build (the kernelagent-builder stage of Dockerfile.cuda). The
# bump procedure and revalidation checklist are in AGENTS.md.
KERNELAGENT_WORKSPACE    ?= ${WORKSPACE}/KernelAgent
KERNELAGENT_UPSTREAM     ?= https://github.com/meta-pytorch/KernelAgent.git
KERNELAGENT_REV          ?= e0647170da36ef9b059ac0bd3d60103aa4ed378b
OPENAI_MODEL             ?= claude-sonnet-4-6
KA_WORKERS               ?= 1
# Refinement-round budgets. KA_GEN_ROUNDS applies to baseline generation;
# KA_OPT_ROUNDS is one budget for the single-kernel optimizers and a
# space-separated list for kernelagent-experiment.
KA_GEN_ROUNDS            ?= 5
KA_OPT_ROUNDS            ?= 5
# Reasoning effort requested from the model on either backend
# (low | medium | high; empty = each provider's model-name gating, i.e.
# current behavior)
KA_REASONING_EFFORT      ?=
# Per-stage LLM specs for kernelagent-experiment only, as
# OPENAI_MODEL[:EFFORT] (empty = both stages use OPENAI_MODEL and
# KA_REASONING_EFFORT; the stage's backend is inferred from its model
# id). The stages may differ: the experiment target forwards both
# providers' credentials.
KA_GENERATION_MODEL      ?=
KA_OPTIMIZATION_MODEL    ?=
# GPU name for the optimizer (detected via an isolated CUDA probe and
# nvidia-smi when empty); selects the entry in the optimizer's GPU specs
# catalog
KA_GPU_NAME              ?=
# Hardware description substituted for the @GPU_TARGET@ token in each
# problem.txt (detected via the isolated CUDA probe when empty): name,
# memory, compute capability, catalogued architecture, and sm_<major><minor>.
# Set it to override the whole description or to target hardware you are
# not running on
KA_GPU_TARGET            ?=
# Host-side JSON file of supplemental optimizer GPU specs (empty = the
# image's shipped catalog only): a JSON object mapping the GPU name
# exactly as CUDA and nvidia-smi report it to a specs dict shaped like a
# catalog entry, giving the optimizer real peak figures for GPUs the
# shipped catalog does not cover. Mounted read-only into the container
# and pointed at by KA_GPU_SPECS_FILE there. See the KernelAgent README
# for the format and an example
KA_GPU_SPECS_FILE        ?=
KA_PROBLEM               ?= matmul_bias
KA_CASE                  ?= 0
KA_IMPL                  ?= kernel
# One profiler, or a space-separated list for the experiment targets
KA_PROFILER              ?= proton
# Measured capture iterations; every profiler uses it (ncu replays every
# profiled launch, so prefer a small value such as 3 for ncu captures)
KA_PROFILE_ITERS         ?= 10
# ncu metric set: basic | detailed | full
KA_NCU_SET               ?= basic
# The kernelagent-* targets run the shared CUDA image
KA_IMAGE_TAG             := $(CUDA_IMAGE_NAME):$(CUDA_VERSION)
# The problem list is derived from the workspace, not hardcoded: every
# problems/ subdirectory holding spec.py is a problem (check_install.py
# verifies the full problem.txt/test_fixed.py/bench.py/spec.py set the
# CLIs require when the image is built). wildcard sorts, so the list is
# stable.
KA_ALL_PROBLEMS          := $(patsubst $(KERNELAGENT_WORKSPACE)/problems/%/spec.py,%,$(wildcard $(KERNELAGENT_WORKSPACE)/problems/*/spec.py))
# Problems selected by an explicitly given KA_PROBLEM. Unset keeps each
# target's documented default: matmul_bias for the single-problem
# targets, every problem for kernelagent-experiment.
KA_SELECTED_PROBLEMS     := $(if $(filter command line environment,$(origin KA_PROBLEM)),$(KA_PROBLEM),$(KA_ALL_PROBLEMS))
# Experiment optimization budgets: unset keeps the documented "5 10"
# (origin distinguishes it from the single-optimizer default).
KA_EXPERIMENT_OPT_ROUNDS := $(if $(filter command line environment,$(origin KA_OPT_ROUNDS)),$(KA_OPT_ROUNDS),5 10)
# Verification record written by kernelagent-verify for the selected
# KA_IMPL (kernel -> final_kernel.py, optimized -> optimized_kernel.py)
KA_VERIFY_RECORD         := $(if $(filter optimized,$(KA_IMPL)),verify_optimized.json,verify.json)
# External kernel source for kernelagent-profile and
# kernelagent-experiment-baseline. Accepts a host path (absolute or
# repo-relative), the historical problems-mount-relative form (e.g.
# matmul_bias/baselines/rocm_blog_fusion.py), or a /workspace/user/...
# path copied from inside the container (mapped back to its host mount).
# KA_LABEL names the kernel in verification records, capture filenames,
# and reports.
KA_KERNEL_SOURCE         ?=
KA_LABEL                 ?=
# Host file the source resolves to (empty when it does not exist; the
# parse-time check below turns that into a clear error).
KA_KERNEL_HOST_PATH = $(if $(strip $(KA_KERNEL_SOURCE)),$(firstword \
	$(wildcard $(KA_KERNEL_SOURCE)) \
	$(wildcard $(patsubst /workspace/user/%,$(KERNELAGENT_WORKSPACE)/%,$(KA_KERNEL_SOURCE))) \
	$(wildcard $(KERNELAGENT_WORKSPACE)/problems/$(KA_KERNEL_SOURCE))))
# A source already under one of the container mounts needs no copy: map
# its host path back to the container path that mount already provides.
KA_KERNEL_MOUNTED_CONTAINER_PATH = $(patsubst $(KERNELAGENT_WORKSPACE)/%,/workspace/user/%,$(filter \
	$(KERNELAGENT_WORKSPACE)/problems/% \
	$(KERNELAGENT_WORKSPACE)/results/%,$(KA_KERNEL_HOST_PATH)))
# Sources outside those mounts are copied into a mounted tree before the
# container runs; kernelagent-experiment-baseline overrides the directory
# with the experiment tree.
KA_KERNEL_STAGE_DIR = $(KERNELAGENT_WORKSPACE)/results/.kernel-sources
KA_KERNEL_STAGE_HOST = $(KA_KERNEL_STAGE_DIR)/$(notdir $(KA_KERNEL_HOST_PATH))
KA_KERNEL_CONTAINER_PATH = $(or $(KA_KERNEL_MOUNTED_CONTAINER_PATH),$(patsubst $(KERNELAGENT_WORKSPACE)/%,/workspace/user/%,$(KA_KERNEL_STAGE_HOST)))

# Comparison experiments (kernelagent-experiment-*): one experiment
# generates a baseline kernel per selected problem (KA_GEN_ROUNDS is that
# generation's refinement budget), independently optimizes the same
# baseline once per budget in KA_OPT_ROUNDS (space-separated, e.g.
# "5 10"), then profiles and reports. See workspace/KernelAgent/README.md.
KA_EXPERIMENT            ?=
# Experiment directory for KA_EXPERIMENT (an id, an absolute/repo-relative
# directory, or a /workspace/user/... path). The parse-time checks require
# its manifest.
KA_EXPERIMENT_DIR = $(if $(filter /workspace/user/%,$(KA_EXPERIMENT)),$(patsubst /workspace/user/%,$(KERNELAGENT_WORKSPACE)/%,$(KA_EXPERIMENT)),$(if $(wildcard $(KA_EXPERIMENT)/manifest.json),$(abspath $(KA_EXPERIMENT)),$(KERNELAGENT_WORKSPACE)/results/experiments/$(KA_EXPERIMENT)))
# The same directory as the container sees it; used by the targets whose
# CLI accepts an experiment id or a directory.
KA_EXPERIMENT_CONTAINER_PATH = $(if $(filter $(KERNELAGENT_WORKSPACE)/%,$(KA_EXPERIMENT_DIR)),$(patsubst $(KERNELAGENT_WORKSPACE)/%,/workspace/user/%,$(KA_EXPERIMENT_DIR)),$(KA_EXPERIMENT_DIR))
# Warmup iterations before the measured capture region (profiling and
# experiment captures)
KA_PROFILE_WARMUP        ?= 10
# Per-test subprocess timeout forwarded to generation/optimization/verify
KA_TEST_TIMEOUT          ?= 180
# Optional destination for kernelagent-report's Markdown
KA_REPORT_OUT            ?=
# Capture batch reported by kernelagent-report (empty: the latest)
KA_CAPTURE_BATCH         ?=

# KA_IMPL is shared by targets with different valid values: verify takes
# the saved kernels (kernel | optimized), profile additionally takes the
# PyTorch baseline (ref). Reject the mismatch at parse time instead of
# letting the container's argparse fail after a slow image check.
KA_VERIFY_IMPLS          := kernel optimized
KA_PROFILE_IMPLS         := kernel optimized ref
# Every target that profiles a saved kernel or the PyTorch baseline.
KA_PROFILE_TARGETS       := kernelagent-profile kernelagent-profile-unoptimized \
	kernelagent-profile-optimized
KA_EXPERIMENT_TARGETS    := kernelagent-experiment kernelagent-experiment-profile \
	kernelagent-experiment-baseline
KA_PROFILER_CHOICES      := ncu nsys proton
KA_NCU_SETS              := basic detailed full
# Every target that takes exactly one KA_PROBLEM.
KA_SINGLE_PROBLEM_TARGETS := $(KA_PROFILE_TARGETS) kernelagent-generate \
	kernelagent-optimize kernelagent-verify kernelagent-experiment-baseline
# Every target that takes one or more problems: an explicit KA_PROBLEM
# selects them, otherwise every problem runs.
KA_MULTI_PROBLEM_TARGETS  := kernelagent-experiment

# Legacy names from before the KA_ namespace. Reject them on any
# kernelagent-* goal so an old command or a typo can never be silently
# ignored. Gating the check on the goals keeps a globally exported MODEL
# or PROBLEM from breaking the cuda-*/rocm-* targets.
KA_LEGACY_VARS := MODEL REASONING_EFFORT GENERATION OPTIMIZATION PROBLEM \
	PROBLEMS PROFILE_PROBLEMS CASE IMPL PROFILER PROFILERS PROFILE_ITERS \
	PROFILE_WARMUP NCU_SET KERNEL LABEL EXPERIMENT OPT_ROUNDS REPORT_OUT \
	CAPTURE_BATCH KA_ROUNDS
KA_LEGACY_SET := $(strip $(foreach v,$(KA_LEGACY_VARS),$(if $(filter command line environment,$(origin $(v))),$(v))))
ifneq (,$(filter kernelagent-%,$(MAKECMDGOALS)))
ifneq (,$(KA_LEGACY_SET))
$(error $(KA_LEGACY_SET) is no longer a KernelAgent option; use OPENAI_MODEL for MODEL and the KA_-prefixed name for everything else (see make help))
endif
endif

# Per-stage model overrides belong to kernelagent-experiment only.
ifeq (,$(filter kernelagent-experiment,$(MAKECMDGOALS)))
ifneq (,$(filter kernelagent-%,$(MAKECMDGOALS)))
ifneq (,$(filter command line environment,$(origin KA_GENERATION_MODEL)))
$(error KA_GENERATION_MODEL is only supported by kernelagent-experiment)
endif
ifneq (,$(filter command line environment,$(origin KA_OPTIMIZATION_MODEL)))
$(error KA_OPTIMIZATION_MODEL is only supported by kernelagent-experiment)
endif
endif
endif

ifneq (,$(filter kernelagent-verify,$(MAKECMDGOALS)))
ifneq ($(KA_IMPL),$(filter $(KA_VERIFY_IMPLS),$(KA_IMPL)))
$(error unknown KA_IMPL '$(KA_IMPL)' for kernelagent-verify; valid choices: $(KA_VERIFY_IMPLS) — ref is kernelagent-profile only)
endif
endif
ifneq (,$(filter kernelagent-profile,$(MAKECMDGOALS)))
ifneq ($(KA_IMPL),$(filter $(KA_PROFILE_IMPLS),$(KA_IMPL)))
$(error unknown KA_IMPL '$(KA_IMPL)' for kernelagent-profile; valid choices: $(KA_PROFILE_IMPLS))
endif
ifneq ($(strip $(KA_KERNEL_SOURCE)),)
ifeq ($(strip $(KA_LABEL)),)
$(error KA_LABEL=<name> is required with KA_KERNEL_SOURCE for kernelagent-profile)
endif
ifeq ($(KA_IMPL),ref)
$(error KA_KERNEL_SOURCE=<path> cannot be combined with KA_IMPL=ref for kernelagent-profile — the PyTorch baseline has no kernel source)
endif
endif
endif
ifneq (,$(filter kernelagent-profile-unoptimized kernelagent-profile-optimized,$(MAKECMDGOALS)))
ifneq ($(strip $(KA_KERNEL_SOURCE)),)
$(error KA_KERNEL_SOURCE is not supported by kernelagent-profile-unoptimized/-optimized; they profile the saved artifacts — use kernelagent-profile)
endif
ifneq (,$(filter command line environment,$(origin KA_IMPL)))
$(error KA_IMPL is fixed by kernelagent-profile-unoptimized/-optimized; use kernelagent-profile to select an impl)
endif
endif

ifneq (,$(filter $(KA_PROFILE_TARGETS),$(MAKECMDGOALS)))
ifeq ($(strip $(KA_PROFILER)),)
$(error KA_PROFILER must name one of $(KA_PROFILER_CHOICES) for $(KA_PROFILE_TARGETS))
endif
ifneq ($(words $(KA_PROFILER)),1)
$(error KA_PROFILER must be exactly one of $(KA_PROFILER_CHOICES) for $(KA_PROFILE_TARGETS), not '$(KA_PROFILER)')
endif
ifneq ($(KA_PROFILER),$(filter $(KA_PROFILER_CHOICES),$(KA_PROFILER)))
$(error unknown KA_PROFILER '$(KA_PROFILER)' for $(KA_PROFILE_TARGETS); valid choices: $(KA_PROFILER_CHOICES))
endif
endif

ifneq (,$(filter $(KA_SINGLE_PROBLEM_TARGETS),$(MAKECMDGOALS)))
ifneq ($(KA_PROBLEM),$(filter $(KA_ALL_PROBLEMS),$(KA_PROBLEM)))
$(error unknown KA_PROBLEM '$(KA_PROBLEM)' for $(KA_SINGLE_PROBLEM_TARGETS); valid choices: $(KA_ALL_PROBLEMS))
endif
ifneq ($(words $(KA_PROBLEM)),1)
$(error KA_PROBLEM must be exactly one of $(KA_ALL_PROBLEMS), not '$(KA_PROBLEM)')
endif
endif

ifneq (,$(filter $(KA_MULTI_PROBLEM_TARGETS),$(MAKECMDGOALS)))
ifeq ($(strip $(KA_SELECTED_PROBLEMS)),)
$(error KA_PROBLEM must name at least one problem for $(KA_MULTI_PROBLEM_TARGETS); valid choices: $(KA_ALL_PROBLEMS))
endif
ifneq ($(KA_SELECTED_PROBLEMS),$(filter $(KA_ALL_PROBLEMS),$(KA_SELECTED_PROBLEMS)))
$(error unknown KA_PROBLEM '$(KA_SELECTED_PROBLEMS)' for $(KA_MULTI_PROBLEM_TARGETS); valid choices: $(KA_ALL_PROBLEMS) (space-separated))
endif
endif

ifneq (,$(filter kernelagent-optimize,$(MAKECMDGOALS)))
ifneq ($(words $(KA_OPT_ROUNDS)),1)
$(error KA_OPT_ROUNDS must be exactly one optimization budget for $(MAKECMDGOALS), not '$(KA_OPT_ROUNDS)')
endif
endif

ifneq (,$(filter $(KA_EXPERIMENT_TARGETS),$(MAKECMDGOALS)))
ifeq ($(strip $(KA_PROFILER)),)
ifneq (,$(filter kernelagent-experiment kernelagent-experiment-profile,$(MAKECMDGOALS)))
$(error KA_PROFILER must name at least one profiler for $(MAKECMDGOALS); use KA_PROFILER= only with kernelagent-experiment-baseline)
endif
endif
ifneq ($(KA_PROFILER),$(filter $(KA_PROFILER_CHOICES),$(KA_PROFILER)))
$(error unknown KA_PROFILER '$(KA_PROFILER)' for $(KA_EXPERIMENT_TARGETS); valid choices: $(KA_PROFILER_CHOICES))
endif
ifneq ($(words $(KA_EXPERIMENT)),1)
ifneq ($(strip $(KA_EXPERIMENT)),)
$(error KA_EXPERIMENT must be a single id, not '$(KA_EXPERIMENT)')
endif
endif
endif

ifneq (,$(filter kernelagent-experiment-profile,$(MAKECMDGOALS)))
ifneq (,$(filter command line environment,$(origin KA_PROBLEM)))
ifeq ($(strip $(KA_PROBLEM)),)
$(error KA_PROBLEM must name at least one problem for kernelagent-experiment-profile)
endif
endif
endif

ifneq (,$(filter $(KA_PROFILE_TARGETS) $(KA_EXPERIMENT_TARGETS),$(MAKECMDGOALS)))
ifneq ($(KA_NCU_SET),$(filter $(KA_NCU_SETS),$(KA_NCU_SET)))
$(error unknown KA_NCU_SET '$(KA_NCU_SET)'; valid choices: $(KA_NCU_SETS))
endif
endif

ifneq (,$(filter kernelagent-experiment-profile kernelagent-experiment-baseline kernelagent-report,$(MAKECMDGOALS)))
ifeq ($(strip $(KA_EXPERIMENT)),)
$(error KA_EXPERIMENT=<id> is required for kernelagent-experiment-profile, kernelagent-experiment-baseline, and kernelagent-report)
endif
endif
ifneq (,$(filter kernelagent-experiment-baseline,$(MAKECMDGOALS)))
ifeq ($(strip $(KA_LABEL)),)
$(error KA_LABEL=<name> is required for kernelagent-experiment-baseline)
endif
ifeq ($(strip $(KA_KERNEL_SOURCE)),)
$(error KA_KERNEL_SOURCE=<path> is required for kernelagent-experiment-baseline)
endif
endif

# Host file/path inputs are checked before any image build or run: podman
# silently creates a directory for a missing bind-mount source, and a
# missing kernel source otherwise surfaces only after the container starts.
ifneq (,$(filter kernelagent-%,$(MAKECMDGOALS)))
ifneq ($(strip $(KA_GPU_SPECS_FILE)),)
ifeq (,$(shell test -f "$(KA_GPU_SPECS_FILE)" && echo yes))
$(error KA_GPU_SPECS_FILE '$(KA_GPU_SPECS_FILE)' not found)
endif
endif
endif

ifneq (,$(filter kernelagent-profile kernelagent-experiment-baseline,$(MAKECMDGOALS)))
ifneq ($(strip $(KA_KERNEL_SOURCE)),)
ifeq (,$(shell test -f "$(KA_KERNEL_HOST_PATH)" && echo yes))
$(error KA_KERNEL_SOURCE '$(KA_KERNEL_SOURCE)' not found — pass a host path, a path under the problems mount, or a /workspace/user/... path)
endif
endif
endif

ifneq (,$(filter kernelagent-experiment-profile kernelagent-experiment-baseline kernelagent-report,$(MAKECMDGOALS)))
ifneq ($(strip $(KA_EXPERIMENT)),)
ifeq (,$(shell test -f "$(KA_EXPERIMENT_DIR)/manifest.json" && echo yes))
$(error no experiment '$(KA_EXPERIMENT)' at $(KA_EXPERIMENT_DIR))
endif
endif
endif

# The container can only see an experiment that lives under the results
# mount (the report target runs on the host, so it is exempt).
ifneq (,$(filter kernelagent-experiment-profile kernelagent-experiment-baseline,$(MAKECMDGOALS)))
ifneq ($(strip $(KA_EXPERIMENT)),)
ifeq (,$(filter $(KERNELAGENT_WORKSPACE)/results/%,$(KA_EXPERIMENT_DIR)))
$(error experiment directory '$(KA_EXPERIMENT_DIR)' is outside the results mount $(KERNELAGENT_WORKSPACE)/results/)
endif
endif
endif

# LLM environment forwarded into the container: the model travels as
# OPENAI_MODEL, the same name the container and both providers read, and
# the reasoning effort as KA_REASONING_EFFORT (the provider patch's). The
# backend is derived inside the container from the model id, so both
# providers' credentials always travel along — secrets by reference so
# their values never appear in make output or process lists.
ka_llm_env := -e OPENAI_MODEL=$(OPENAI_MODEL) -e OPENAI_API_KEY -e OPENAI_BASE_URL \
	-e ANTHROPIC_API_KEY \
	$(if $(KA_REASONING_EFFORT),-e KA_REASONING_EFFORT=$(KA_REASONING_EFFORT))

# LLM environment for the experiment target only: unlike ka_llm_env
# (one model for the whole container), the per-stage OPENAI_MODEL[:EFFORT]
# specs also travel — as KA_GENERATION_MODEL / KA_OPTIMIZATION_MODEL env
# vars the runner resolves per stage (an empty part falls back to
# OPENAI_MODEL / KA_REASONING_EFFORT, and each stage's backend is inferred
# from its model id).
ka_experiment_env := $(ka_llm_env) \
	$(if $(strip $(KA_GENERATION_MODEL)),-e "KA_GENERATION_MODEL=$(KA_GENERATION_MODEL)") \
	$(if $(strip $(KA_OPTIMIZATION_MODEL)),-e "KA_OPTIMIZATION_MODEL=$(KA_OPTIMIZATION_MODEL)")

# KernelAgent runtime arguments, added to the shared cuda_args by the
# kernelagent-* targets: the entrypoint workflow selector, plus
# host-gateway entries that let OPENAI_BASE_URL point at model servers
# running on the host while the container stays on the default bridge.
# The LLM backend is derived by the entrypoint from the model id, not
# forwarded here. A supplemental GPU specs file (KA_GPU_SPECS_FILE, a
# host path) is mounted read-only at a fixed container path so every
# kernelagent target — and the experiment runner's optimize children,
# which inherit the container environment — resolves it by the same env
# var the optimizer reads.
define ka_args
	--add-host=host.containers.internal:host-gateway \
	--add-host=host.docker.internal:host-gateway \
	-e CONTAINER_WORKFLOW=kernelagent \
	$(if $(KA_GPU_SPECS_FILE),$(ka_gpu_specs))
endef

# Only expanded when KA_GPU_SPECS_FILE is set (a separate define so the
# commas in the mount's SELinux option stay out of $(if)'s argument
# parsing)
define ka_gpu_specs
	-v "$(KA_GPU_SPECS_FILE):/workspace/user/gpu_specs.json:ro,Z" \
	-e KA_GPU_SPECS_FILE=/workspace/user/gpu_specs.json
endef

define ka_vol
	-v "${KERNELAGENT_WORKSPACE}/problems:/workspace/user/problems:Z" \
	-v "${KERNELAGENT_WORKSPACE}/results:/workspace/user/results:Z"
endef

KA_RUN := $(CTR_CMD) run --rm $(cuda_dev_args)$(cuda_args)$(ka_args)$(ka_vol)
KA_GENERATE_RUN = $(KA_RUN) $(ka_llm_env) -e KA_WORKERS=$(KA_WORKERS) -e KA_ROUNDS=$(KA_GEN_ROUNDS) \
	-e "KA_GPU_TARGET=$(KA_GPU_TARGET)" $(IMAGE_REPO)/$(KA_IMAGE_TAG) generate-kernel.py generate \
	--test-timeout-s $(KA_TEST_TIMEOUT)
KA_OPTIMIZE_RUN = $(KA_RUN) $(ka_llm_env) -e KA_WORKERS=$(KA_WORKERS) -e KA_ROUNDS=$(KA_OPT_ROUNDS) \
	-e "KA_GPU_NAME=$(KA_GPU_NAME)" -e "KA_GPU_TARGET=$(KA_GPU_TARGET)" \
	$(IMAGE_REPO)/$(KA_IMAGE_TAG) generate-kernel.py optimize \
	--test-timeout-s $(KA_TEST_TIMEOUT)
KA_VERIFY_RUN = $(KA_RUN) $(IMAGE_REPO)/$(KA_IMAGE_TAG) generate-kernel.py verify \
	--test-timeout-s $(KA_TEST_TIMEOUT)

# Validate the LLM configuration before generation or smoke tests, using
# the same model-id/OPENAI_BASE_URL rule as the entrypoint (the Makefile
# cannot see a container-side LLM_BACKEND override).
# Secrets are read from the host environment at run time, never stored here.
.PHONY: _ka_require_llm
_ka_require_llm:
ifeq ($(strip $(OPENAI_MODEL)),)
	@echo "ERROR: OPENAI_MODEL is required (Anthropic API: OPENAI_MODEL=claude-sonnet-4-6 ANTHROPIC_API_KEY=sk-ant-...; OpenAI or local server: OPENAI_MODEL=<model id> OPENAI_API_KEY=... [OPENAI_BASE_URL=...])" >&2; \
	exit 1
else ifneq ($(OPENAI_BASE_URL),)
	@if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "ERROR: OPENAI_BASE_URL is set, so the OpenAI-compatible client needs OPENAI_API_KEY (any non-empty value works for local servers)" >&2; \
		exit 1; \
	fi
else ifneq ($(filter claude-%,$(OPENAI_MODEL)),)
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then \
		echo "ERROR: OPENAI_MODEL=$(OPENAI_MODEL) uses the Anthropic API; set ANTHROPIC_API_KEY (or point OPENAI_BASE_URL at an OpenAI-compatible server serving it)" >&2; \
		exit 1; \
	fi
else
	@if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "ERROR: OPENAI_MODEL=$(OPENAI_MODEL) uses the OpenAI API; set OPENAI_API_KEY" >&2; \
		exit 1; \
	fi
endif

# kernelagent-experiment may run each stage on its own model, so when the
# run-wide OPENAI_MODEL is empty, both stage specs must be set instead; the
# runner then resolves and validates each stage's model and credentials
# before it touches the filesystem.
ka_experiment_llm_hint := for example KA_GENERATION_MODEL=gpt-6.1-sol KA_OPTIMIZATION_MODEL=claude-sonnet-4-6:high
.PHONY: _ka_require_experiment_llm
_ka_require_experiment_llm:
ifeq ($(strip $(KA_GENERATION_MODEL)),)
	@echo "ERROR: OPENAI_MODEL is empty, so KA_GENERATION_MODEL must be set as OPENAI_MODEL[:EFFORT] ($(ka_experiment_llm_hint))" >&2; \
	exit 1
else ifeq ($(strip $(KA_OPTIMIZATION_MODEL)),)
	@echo "ERROR: OPENAI_MODEL is empty, so KA_OPTIMIZATION_MODEL must be set as OPENAI_MODEL[:EFFORT] ($(ka_experiment_llm_hint))" >&2; \
	exit 1
endif

# Build the shared CUDA image; KernelAgent is part of it. Alias kept for
# the KernelAgent workflow commands — it is the same image and the same
# build as cuda-image.
.PHONY: kernelagent-image
kernelagent-image: cuda-image

# Open a shell in the KernelAgent container (with GPU support). The LLM
# environment is forwarded so verify-llm.py and generation work from inside
# the console exactly as they do through the make targets.
.PHONY: kernelagent-console
kernelagent-console: kernelagent-image
	$(call ctr_run,$(cuda_dev_args)$(cuda_args)$(ka_args) $(ka_llm_env)$(ka_vol),$(KA_IMAGE_TAG),)

# Run the Nsight Systems UI in the KernelAgent container (with GPU
# support); the mounted results tree holds the saved profiles
.PHONY: kernelagent-systems
kernelagent-systems: kernelagent-image
	$(call ctr_run,$(cuda_dev_args)$(cuda_args)$(ka_args)$(ka_vol)$(nsight_args),$(KA_IMAGE_TAG),nsys-ui)

# Run the Nsight Compute UI in the KernelAgent container (with GPU
# support); the mounted results tree holds the saved profiles
.PHONY: kernelagent-compute
kernelagent-compute: kernelagent-image
	$(call ctr_run,$(cuda_dev_args)$(cuda_args)$(ka_args)$(ka_vol)$(nsight_args),$(KA_IMAGE_TAG),ncu-ui)

# Smoke-test the configured LLM provider (prints VERIFY_OK on success)
.PHONY: kernelagent-test
kernelagent-test: kernelagent-image _ka_require_llm
	$(KA_RUN) $(ka_llm_env) $(IMAGE_REPO)/$(KA_IMAGE_TAG) verify-llm.py

# Generate one problem's Triton kernel from its fixed spec, selected with
# KA_PROBLEM (default matmul_bias). Mirrors kernelagent-optimize.
.PHONY: kernelagent-generate
kernelagent-generate: kernelagent-image _ka_require_llm
	@echo "Kernel session will be saved under $(KERNELAGENT_WORKSPACE)/results/$(KA_PROBLEM)/"
	$(KA_GENERATE_RUN) $(KA_PROBLEM)
	@echo "Results: $(KERNELAGENT_WORKSPACE)/results/$(KA_PROBLEM)/ (final_kernel.py)"

# Optimize the saved final_kernel.py of one problem for one benchmark case
# (KA_CASE) with KernelAgent's hardware-guided optimizer (NCU profiling plus
# LLM rewrites). An accepted result — fixed test PASS and strictly faster
# on the selected case — is saved separately as optimized_kernel.py; a
# failed attempt leaves any previously accepted kernel in place.
.PHONY: kernelagent-optimize
kernelagent-optimize: kernelagent-image _ka_require_llm
	@echo "Optimization session will be saved under $(KERNELAGENT_WORKSPACE)/results/$(KA_PROBLEM)/"
	$(KA_OPTIMIZE_RUN) $(KA_PROBLEM) --case $(KA_CASE)
	@echo "Results: $(KERNELAGENT_WORKSPACE)/results/$(KA_PROBLEM)/ (optimized_kernel.py, optimize_result.json)"

# Re-run the fixed correctness tests and benchmarks on one problem's saved
# kernel (no LLM access needed). KA_IMPL selects which saved kernel to
# verify: kernel (final_kernel.py, default) or optimized (optimized_kernel.py).
.PHONY: kernelagent-verify
kernelagent-verify: kernelagent-image
	@echo "Verification record will be saved under $(KERNELAGENT_WORKSPACE)/results/$(KA_PROBLEM)/"
	$(KA_VERIFY_RUN) $(KA_PROBLEM) --impl $(KA_IMPL)
	@echo "Results: $(KERNELAGENT_WORKSPACE)/results/$(KA_PROBLEM)/ ($(KA_VERIFY_RECORD))"

# Kernel sources outside the mounted trees are copied into a mounted tree
# (already labeled for the container) before the container runs; the
# recipe passes the copy's container path. Sources already under problems/
# or results/ need no copy.
define ka_stage_kernel_source
	set -e; if [ -n "$(KA_KERNEL_HOST_PATH)" ] && [ -z "$(KA_KERNEL_MOUNTED_CONTAINER_PATH)" ]; then mkdir -p "$(dir $(KA_KERNEL_STAGE_HOST))"; cp -p "$(KA_KERNEL_HOST_PATH)" "$(KA_KERNEL_STAGE_HOST)"; fi;
endef

# Shared recipe for profiling one benchmark case of a saved kernel or the
# PyTorch baseline; $(1) is the KA_IMPL (kernel | optimized | ref).
# KA_KERNEL_SOURCE and KA_LABEL (both required together, checked at parse
# time) profile an external kernel source instead of the saved artifact.
define ka_profile
	$(ka_stage_kernel_source) \
	$(KA_RUN) $(IMAGE_REPO)/$(KA_IMAGE_TAG) profile-kernel.py $(KA_PROBLEM) \
		--case $(KA_CASE) --impl $(1) --profiler $(KA_PROFILER) --iters $(KA_PROFILE_ITERS) \
		--warmup $(KA_PROFILE_WARMUP) --ncu-set $(KA_NCU_SET) \
		$(if $(KA_KERNEL_SOURCE),--kernel $(KA_KERNEL_CONTAINER_PATH) --label $(KA_LABEL))
	@echo "Profiles: $(KERNELAGENT_WORKSPACE)/results/$(KA_PROBLEM)/profiles/"
endef

# Profile a saved kernel — or the PyTorch baseline with KA_IMPL=ref, or an
# external kernel source with KA_KERNEL_SOURCE and KA_LABEL — for one
# benchmark case. KA_IMPL: kernel | optimized | ref. KA_PROFILER: proton |
# nsys | ncu. KA_KERNEL_SOURCE: a host path (absolute or repo-relative), a
# path relative to the problems mount, or a /workspace/user/... path. No
# LLM access needed.
.PHONY: kernelagent-profile
kernelagent-profile: kernelagent-image
	$(call ka_profile,$(KA_IMPL))

# Profile the generated, unoptimized kernel (results/<problem>/
# final_kernel.py) for one benchmark case — the before half of a
# before/after comparison with kernelagent-profile-optimized. Takes the
# same variables as kernelagent-profile except KA_IMPL, which is fixed to
# the generated kernel.
.PHONY: kernelagent-profile-unoptimized
kernelagent-profile-unoptimized: kernelagent-image
	$(call ka_profile,kernel)

# Profile the accepted optimized kernel (results/<problem>/
# optimized_kernel.py) for one benchmark case — the after half of a
# before/after comparison with kernelagent-profile-unoptimized. Requires
# a completed kernelagent-optimize first. Takes the same variables as
# kernelagent-profile except KA_IMPL, which is fixed to the optimized kernel.
.PHONY: kernelagent-profile-optimized
kernelagent-profile-optimized: kernelagent-image
	$(call ka_profile,optimized)

# Shared capture-flag forwarding for the experiment targets: profiler
# selection and the capture batch settings. $(1) guards the batch
# settings — pass 1 where the target always captures
# (kernelagent-experiment, kernelagent-experiment-profile), $(KA_PROFILER)
# for the baseline target, where an empty KA_PROFILER records the result
# without capturing it. $(2) wedges target-specific flags between the
# batch settings and the trailing argument slot.
define ka_capture_flags
	$(if $(1),--profilers "$(KA_PROFILER)" --profile-iters $(KA_PROFILE_ITERS) --profile-warmup $(KA_PROFILE_WARMUP) --ncu-set $(KA_NCU_SET)) $(2)
endef

# Run a comparison experiment: generate one baseline kernel per selected
# problem, independently optimize it once per KA_OPT_ROUNDS budget, verify
# every kernel, capture every requested KA_PROFILER, and write the
# comparison report. Results land under results/experiments/<KA_EXPERIMENT>/;
# with no KA_EXPERIMENT the runner picks a unique id and prints it. Takes
# the LLM variables plus KA_WORKERS, KA_GEN_ROUNDS (generation budget),
# KA_OPT_ROUNDS, KA_CASE, KA_PROBLEM, KA_PROFILER, KA_PROFILE_ITERS,
# KA_PROFILE_WARMUP, KA_NCU_SET, and KA_TEST_TIMEOUT.
# KA_GENERATION_MODEL / KA_OPTIMIZATION_MODEL override the LLM per stage as
# OPENAI_MODEL[:EFFORT] (empty: the run-wide OPENAI_MODEL and
# KA_REASONING_EFFORT).
.PHONY: kernelagent-experiment
kernelagent-experiment: $(if $(strip $(OPENAI_MODEL)),_ka_require_llm,_ka_require_experiment_llm) kernelagent-image
	$(KA_RUN) $(ka_experiment_env) $(IMAGE_REPO)/$(KA_IMAGE_TAG) \
		experiment-kernels.py run $(KA_EXPERIMENT) \
		--problems "$(KA_SELECTED_PROBLEMS)" --case $(KA_CASE) \
		--generation-rounds $(KA_GEN_ROUNDS) --opt-rounds "$(KA_EXPERIMENT_OPT_ROUNDS)" \
		--workers $(KA_WORKERS) \
		$(call ka_capture_flags,1,--test-timeout-s $(KA_TEST_TIMEOUT))
	@echo "Experiment: $(KERNELAGENT_WORKSPACE)/results/experiments/$(if $(KA_EXPERIMENT),$(KA_EXPERIMENT),(auto-generated id — see the run output))"

# Add one capture batch to a saved experiment — re-profile its archived,
# verified kernels with any subset of KA_PROFILER, then refresh the report.
# No LLM needed. An explicit KA_PROBLEM restricts the batch (unset: every
# problem in the experiment).
.PHONY: kernelagent-experiment-profile
kernelagent-experiment-profile: kernelagent-image
	$(KA_RUN) $(IMAGE_REPO)/$(KA_IMAGE_TAG) \
		experiment-kernels.py profile $(KA_EXPERIMENT_CONTAINER_PATH) \
		$(call ka_capture_flags,1,$(if $(filter command line environment,$(origin KA_PROBLEM)),--problems "$(KA_PROBLEM)"))
	@echo "Experiment: $(KA_EXPERIMENT_DIR)"

# Add an external kernel to a saved experiment as a baseline: archive and
# hash it inside the experiment, verify it against the experiment's
# problem snapshot, record it in the manifest, then refresh the report.
# With KA_PROFILER at its default (proton) the new baseline is also
# captured in its own batch; KA_PROFILER= (empty) records the verified
# baseline without profiling it. No LLM needed. KA_KERNEL_SOURCE is a host
# path (absolute or repo-relative), a path relative to the problems mount,
# or a /workspace/user/... path; a host source outside the mounts is copied
# into the experiment's sources/ directory first, and --source-path keeps
# the original path in the manifest and report. KA_LABEL names the
# baseline.
.PHONY: kernelagent-experiment-baseline
kernelagent-experiment-baseline: KA_KERNEL_STAGE_DIR = $(KA_EXPERIMENT_DIR)/sources
kernelagent-experiment-baseline: kernelagent-image
	$(ka_stage_kernel_source) \
	$(KA_RUN) $(IMAGE_REPO)/$(KA_IMAGE_TAG) \
		experiment-kernels.py add-baseline $(KA_EXPERIMENT_CONTAINER_PATH) \
		--problem $(KA_PROBLEM) --label $(KA_LABEL) --kernel $(KA_KERNEL_CONTAINER_PATH) \
		--source-path "$(KA_KERNEL_SOURCE)" \
		--test-timeout-s $(KA_TEST_TIMEOUT) \
		$(call ka_capture_flags,$(KA_PROFILER))
	@echo "Experiment: $(KA_EXPERIMENT_DIR)"

# Render (or refresh) an experiment's Markdown comparison report on the
# host — no container, GPU, or LLM: the report is built from the saved
# manifest, benchmark records, and profiler exports. KA_REPORT_OUT writes
# elsewhere (links follow), KA_CAPTURE_BATCH selects a batch (default:
# latest).
.PHONY: kernelagent-report
kernelagent-report:
	python3 $(KERNELAGENT_WORKSPACE)/scripts/report-experiment.py $(KA_EXPERIMENT) \
		--results-root $(KERNELAGENT_WORKSPACE)/results \
		$(if $(KA_REPORT_OUT),--out $(KA_REPORT_OUT)) \
		$(if $(KA_CAPTURE_BATCH),--capture-batch $(KA_CAPTURE_BATCH))

.PHONY: clean
clean:
	rm -rf workspace/.ipynb_checkpoints

# Print target usage (not the default; invoke explicitly as `make help`)
.PHONY: help
help:
	@echo "Usage: make <target> [VARIABLE=value ...]  (default target: cuda-image)"
	@echo ""
	@echo "CUDA / Nsight (NVIDIA):"
	@echo "  cuda-image                         Build the shared CUDA image (profiling + KernelAgent)"
	@echo "  nsight-systems                     Run the Nsight Systems UI (no GPU)"
	@echo "  nsight-compute                     Run the Nsight Compute UI (no GPU)"
	@echo "  nsight-jupyter                     Run a Jupyter notebook server (no GPU)"
	@echo "  nsight-console                     Open a shell in the container (no GPU)"
	@echo "  cuda-systems                       Run the Nsight Systems UI (with GPU)"
	@echo "  cuda-compute                       Run the Nsight Compute UI (with GPU)"
	@echo "  cuda-jupyter                       Run a Jupyter notebook server (with GPU)"
	@echo "  cuda-console                       Open a shell in the container (with GPU)"
	@echo ""
	@echo "ROCm (AMD):"
	@echo "  rocm-image                         Build the ROCm profiling image"
	@echo "  rocm-jupyter                       Run a Jupyter notebook server (with GPU)"
	@echo "  rocm-console                       Open a shell in the container (with GPU)"
	@echo ""
	@echo "KernelAgent (agentic Triton kernel generation, runs the shared CUDA image):"
	@echo "  kernelagent-image                  Build the shared CUDA image (alias for cuda-image)"
	@echo "  kernelagent-console                Open a shell in the KernelAgent container"
	@echo "  kernelagent-test                   Smoke-test the configured LLM provider (prints VERIFY_OK)"
	@echo "  kernelagent-generate               Generate one problem's Triton kernel (KA_PROBLEM)"
	@echo "  kernelagent-optimize               Optimize a saved kernel for one case (KA_PROBLEM, KA_CASE)"
	@echo "  kernelagent-verify                 Re-run correctness tests and benchmarks on one saved kernel (KA_PROBLEM, KA_IMPL)"
	@echo "  kernelagent-profile                Profile a saved kernel, the PyTorch baseline, or an external kernel (KA_KERNEL_SOURCE, KA_LABEL)"
	@echo "  kernelagent-profile-unoptimized    Profile the generated kernel (KA_IMPL fixed to kernel)"
	@echo "  kernelagent-profile-optimized      Profile the optimized kernel (KA_IMPL fixed to optimized)"
	@echo "  kernelagent-systems                Run the Nsight Systems UI (with GPU)"
	@echo "  kernelagent-compute                Run the Nsight Compute UI (with GPU)"
	@echo "  kernelagent-experiment             Run a comparison experiment (generate, optimize per KA_OPT_ROUNDS, profile, report)"
	@echo "  kernelagent-experiment-profile     Add a capture batch to an experiment (KA_EXPERIMENT, KA_PROFILER)"
	@echo "  kernelagent-experiment-baseline    Add an external kernel baseline to an experiment (KA_EXPERIMENT, KA_PROBLEM, KA_LABEL, KA_KERNEL_SOURCE)"
	@echo "  kernelagent-report                 Render an experiment's Markdown report on the host (KA_EXPERIMENT)"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean                              Remove notebook checkpoints"
	@echo ""
	@echo "Common variables (override with VARIABLE=value):"
	@echo "  CUDA_VERSION=$(CUDA_VERSION)  ROCM_VERSION=$(ROCM_VERSION)  CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"
	@echo "  NOTEBOOK_PORT=8888  OPENAI_MODEL=<model> (default claude-sonnet-4-6)  KA_REASONING_EFFORT=low|medium|high"
	@echo "  OPENAI_API_KEY / ANTHROPIC_API_KEY  OPENAI_BASE_URL=<local server /v1>"
	@echo "  KA_GENERATION_MODEL / KA_OPTIMIZATION_MODEL=<model>[:low|medium|high]  per-stage model and effort (experiments only)"
	@echo "  KA_PROBLEM=<problem> (single-problem targets, default matmul_bias) or '<p1> <p2>' (kernelagent-experiment; default: every problem)"
	@echo "  KA_CASE=$(KA_CASE)  KA_IMPL=kernel|optimized (profile also: ref)  KA_PROFILER=$(KA_PROFILER)  KA_NCU_SET=$(KA_NCU_SET)"
	@echo "  KA_EXPERIMENT=<id>  KA_GEN_ROUNDS=$(KA_GEN_ROUNDS)  KA_OPT_ROUNDS=<budget or '5 10'>  KA_PROFILER='ncu nsys proton'"
	@echo "  KA_PROFILE_ITERS=$(KA_PROFILE_ITERS)  KA_PROFILE_WARMUP=$(KA_PROFILE_WARMUP)  KA_TEST_TIMEOUT=$(KA_TEST_TIMEOUT)  KA_REPORT_OUT=<path>  (list values are space-separated and must be quoted)"
	@echo "  KA_KERNEL_SOURCE=<host path, path relative to problems/, or /workspace/user/... path>  KA_LABEL=<name>  (kernelagent-profile and kernelagent-experiment-baseline)"
	@echo ""
	@echo "Examples:"
	@echo "  make cuda-jupyter NOTEBOOK_PORT=8888"
	@echo ""
	@echo "  Single-kernel workflow:"
	@echo "  make kernelagent-generate KA_PROBLEM=matmul_bias OPENAI_MODEL=gpt-5.2 OPENAI_API_KEY=sk-..."
	@echo "  make kernelagent-optimize KA_PROBLEM=matmul_bias KA_CASE=3 OPENAI_MODEL=gpt-5.2"
	@echo "  make kernelagent-verify KA_PROBLEM=matmul_bias KA_IMPL=kernel"
	@echo "  make kernelagent-profile KA_PROBLEM=flash_attention KA_CASE=0 KA_IMPL=ref KA_PROFILER=proton"
	@echo "  make kernelagent-profile-unoptimized KA_PROBLEM=matmul_bias KA_CASE=3"
	@echo "  make kernelagent-profile-optimized KA_PROBLEM=matmul_bias KA_CASE=3 KA_PROFILER=ncu"
	@echo "  make kernelagent-profile KA_PROBLEM=matmul_bias KA_CASE=0 KA_PROFILER=proton KA_KERNEL_SOURCE=matmul_bias/baselines/rocm_blog_fusion.py KA_LABEL=rocm_blog_fusion"
	@echo ""
	@echo "  Comparison-experiment flow (same KA_EXPERIMENT id; each step refreshes the report):"
	@echo "  make kernelagent-experiment KA_EXPERIMENT=webinar-v1 KA_CASE=3 KA_OPT_ROUNDS='5 10' KA_PROFILER=ncu"
	@echo "  make kernelagent-experiment KA_EXPERIMENT=mixed-v1 KA_CASE=3 KA_GENERATION_MODEL=gpt-6.1-sol KA_OPTIMIZATION_MODEL=claude-sonnet-4-6:high  # one model per stage"
	@echo "  make kernelagent-experiment-baseline KA_EXPERIMENT=webinar-v1 KA_PROBLEM=matmul_bias KA_LABEL=rocm_blog_fusion KA_KERNEL_SOURCE=matmul_bias/baselines/rocm_blog_fusion.py"
	@echo "  make kernelagent-experiment-profile KA_EXPERIMENT=webinar-v1 KA_PROFILER='nsys proton'  # optional: more capture batches"
	@echo "  make kernelagent-report KA_EXPERIMENT=webinar-v1 KA_REPORT_OUT=slides/comparison.md  # optional: re-render / relocate"
