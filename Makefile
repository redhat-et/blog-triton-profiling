IMAGE_REPO ?= triton-profiling
IMAGE_NAME ?= triton-profiling
CUDA_RELEASE ?= 12-8
WORKSPACE ?= $(PWD)/workspace
NOTEBOOK_PORT ?= 8080


# Jupyter Notebook arguments
define notebook_args
	-e NOTEBOOK_PORT=$(NOTEBOOK_PORT) \
	-p $(NOTEBOOK_PORT):$(NOTEBOOK_PORT)
endef

# Podman Run
define podman_run
	podman run -it --rm \
	--privileged \
	--cap-add=SYS_ADMIN \
	-e DISPLAY=${DISPLAY} \
	-e "WAYLAND_DISPLAY=${WAYLAND_DISPLAY}" \
	-e XDG_RUNTIME_DIR=/tmp \
	-e PULSE_SERVER=${XDG_RUNTIME_DIR}/pulse/native \
	-e QT_QPA_PLATFORM=wayland-egl \
	-e WAYLAND_DISPLAY=wayland-0 \
	-v "${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY}:/tmp/${WAYLAND_DISPLAY}:ro" \
	-v "${XDG_RUNTIME_DIR}/pulse:/tmp/pulse:ro" \
	--ipc host \
	-v "${WORKSPACE}:/workspace:Z"
endef


# Podman Runtime CUDA GPU arguments
define cuda_args
	--device nvidia.com/gpu=all \
	--security-opt label=disable
endef


# Podman build
define podman-build
	$(eval $@_IMAGE_NAME_PREFIX = $(1))
	$(eval $@_CONTAINERFILE = $(2))
	podman build \
	--build-arg "CUDA_RELEASE=$(CUDA_RELEASE)" \
	--build-arg "NOTEBOOK_PORT=$(NOTEBOOK_PORT)" \
	-t $(IMAGE_REPO)/$(IMAGE_NAME)-${$@_IMAGE_NAME_PREFIX}:$(CUDA_RELEASE) \
	-f ${$@_CONTAINERFILE} .
endef


# Container build.
.PHONY: nsight-image
nsight-image: containerfiles/Containerfile.nsight
	@$(call podman-build, "nsight", $<)

.PHONY: cuda-image
cuda-image: containerfiles/Containerfile.cuda
	@$(call podman-build, "cuda", $<)


# Generate the NVIDIA CDI for the container toolkit
.PHONY: nvidia-cdi
nvidia-cdi:
	sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml


# NVIDIA Nsight Tools no-CUDA Support
define nsight-run
	$(podman_run) $(IMAGE_REPO)/$(IMAGE_NAME)-nsight:$(CUDA_RELEASE) $(1)
endef


# Run the NVIDIA Nsight Systems UI (no-CUDA)
.PHONY: nsight-systems
nsight-systems: nsight-image
	@$(call nsight-run, "nsys-ui")

# Run the NVIDIA Nsight Compute UI (no-CUDA)
.PHONY: nsight-compute
nsight-compute: nsight-image
	@$(call nsight-run, "ncu-ui")

# Run a Jupyter Notebook Server (no-CUDA)
.PHONY: nsight-jupyter
nsight-jupyter: nsight-image
	$(podman_run) $(notebook_args) $(IMAGE_REPO)/$(IMAGE_NAME)-nsight:$(CUDA_RELEASE) start_jupyter

# Open a shell in the NVIDIA Nsight container (no-CUDA)
.PHONY: nsight-console
nsight-console: nsight-image
	@$(call nsight-run, "")


# NVIDIA Nsight Tools w/CUDA Support
define cuda-run
	$(podman_run) $(cuda_args) $(IMAGE_REPO)/$(IMAGE_NAME)-cuda:$(CUDA_RELEASE) $(1)
endef

# Run the NVIDIA Nsight Systems UI (w/CUDA)
.PHONY: cuda-systems
cuda-systems: cuda-image
	@$(call cuda-run, "nsys-ui")

# Run the NVIDIA Nsight Systems UI (w/CUDA)
.PHONY: cuda-compute
cuda-compute: cuda-image
	@$(call cuda-run, "ncu-ui")

# Run a Jupyter Notebook Server
.PHONY: cuda-jupyter
cuda-jupyter: cuda-image
	$(podman_run) $(notebook_args) $(cuda_args) $(IMAGE_REPO)/$(IMAGE_NAME)-cuda:$(CUDA_RELEASE) start_jupyter

# Open a shell in the NVIDIA Nsight container (CUDA support)
.PHONY: cuda-console
cuda-console: cuda-image
	@$(call cuda-run, "")
