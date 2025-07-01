IMAGE_REPO ?= triton-profiling
IMAGE_NAME ?= triton-profiling
CUDA_RELEASE ?= 12-8
WORKSPACE ?= $(PWD)/workspace


# Podman build
define podman-build
	podman build \
	--build-arg "CUDA_RELEASE=$(CUDA_RELEASE)" \
	--build-arg "NOTEBOOK_PORT=$(NOTEBOOK_PORT)" \
	-t $(IMAGE_REPO)/$(IMAGE_NAME)-$(1):$(CUDA_RELEASE) \
	-f $(2) .
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
	-v "${WORKSPACE}:/workspace:Z" \
	$(1) \
	$(IMAGE_REPO)/$(IMAGE_NAME)-$(2):$(CUDA_RELEASE) \
	$(3)
endef

# Podman Runtime Jupyter Notebook arguments
define notebook_args
	-e NOTEBOOK_PORT=$(NOTEBOOK_PORT) \
	-p $(NOTEBOOK_PORT):$(NOTEBOOK_PORT)
endef

# Podman Runtime CUDA GPU arguments
define cuda_args
	--device nvidia.com/gpu=all \
	--security-opt label=disable
endef


# NVIDIA Nsight Profiling image without CUDA support
.PHONY: nsight-image
nsight-image: containerfiles/Containerfile.nsight
	@$(call podman-build,nsight,$<)

# Generate the NVIDIA CDI for the container toolkit
.PHONY: nvidia-cdi
nvidia-cdi:
	sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Run the NVIDIA Nsight Systems UI (no-CUDA)
.PHONY: nsight-systems
nsight-systems: nsight-image
	$(call podman_run,,nsight,nsys-ui)

# Run the NVIDIA Nsight Compute UI (no-CUDA)
.PHONY: nsight-compute
nsight-compute: nsight-image
	$(call podman_run,, nsight,ncu-ui)

# Run a Jupyter Notebook Server (no-CUDA)
.PHONY: nsight-jupyter
nsight-jupyter: NOTEBOOK_PORT ?= 8888
nsight-jupyter: nsight-image
	$(call podman_run,$(notebook_args),nsight,start_jupyter)
	$(podman_run) $(notebook_args) $(IMAGE_REPO)/$(IMAGE_NAME)-nsight:$(CUDA_RELEASE) start_jupyter

# Open a shell in the NVIDIA Nsight container (no-CUDA)
.PHONY: nsight-console
nsight-console: NOTEBOOK_PORT ?= 8889
nsight-console: nsight-image
	$(call podman_run,$(notebook_args),nsight,)

# NVIDIA Nsight Profiling image with CUDA support
.PHONY: cuda-image
cuda-image: containerfiles/Containerfile.cuda
	@$(call podman-build,cuda,$<)

# Run the NVIDIA Nsight Systems UI (w/CUDA)
.PHONY: cuda-systems
cuda-systems: cuda-image
	$(call podman_run,$(cuda_args),cuda,nsys-ui)

# Run the NVIDIA Nsight Systems UI (w/CUDA)
.PHONY: cuda-compute
cuda-compute: cuda-image
	$(call podman_run,$(cuda_args),cuda,ncu-ui)

# Run a Jupyter Notebook Server
.PHONY: cuda-jupyter
cuda-jupyter: NOTEBOOK_PORT ?= 8888
cuda-jupyter: cuda-image
	$(call podman_run,$(cuda_args) $(notebook_args),cuda,start_jupyter)

# Open a shell in the NVIDIA Nsight container (CUDA support)
.PHONY: cuda-console
cuda-console: NOTEBOOK_PORT ?= 8889
cuda-console: cuda-image
	$(call podman_run,$(cuda_args) $(notebook_args),cuda,)
