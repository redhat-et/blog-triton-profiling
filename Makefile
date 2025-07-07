IMAGE_REPO ?= triton-profiling
CUDA_IMAGE_NAME ?= cuda-profiling
CUDA_RELEASE ?= 12-8
WORKSPACE ?= $(PWD)/workspace


# Podman Run
define podman_run_cuda
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
	$(IMAGE_REPO)/$(CUDA_IMAGE_NAME):$(CUDA_RELEASE) \
	$(2)
endef

# Podman Runtime CUDA GPU arguments
define cuda_args
	--device nvidia.com/gpu=all \
	--security-opt label=disable
endef

# Podman Runtime Jupyter Notebook arguments
define notebook_args
	-e NOTEBOOK_PORT=$(NOTEBOOK_PORT) \
	-p $(NOTEBOOK_PORT):$(NOTEBOOK_PORT)
endef


# Build NVIDIA Nsight profiling image
.PHONY: cuda-image
cuda-image: NOTEBOOK_PORT ?= 8888
cuda-image: containerfiles/Containerfile.cuda
	podman build \
	--build-arg "CUDA_RELEASE=$(CUDA_RELEASE)" \
	--build-arg "NOTEBOOK_PORT=$(NOTEBOOK_PORT)" \
	-t $(IMAGE_REPO)/$(CUDA_IMAGE_NAME):$(CUDA_RELEASE) \
	-f $< .


# Generate the NVIDIA CDI for the container toolkit
.PHONY: nvidia-cdi
nvidia-cdi:
	sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml


# Run the NVIDIA Nsight Systems UI (with no-GPU support)
.PHONY: nsight-systems
nsight-systems: cuda-image
	$(call podman_run_cuda,,nsys-ui)

# Run the NVIDIA Nsight Compute UI (with no-GPU support)
.PHONY: nsight-compute
nsight-compute: cuda-image
	$(call podman_run_cuda,,ncu-ui)

# Run a Jupyter Notebook Server (with no-GPU support)
.PHONY: nsight-jupyter
nsight-jupyter: NOTEBOOK_PORT ?= 8888
nsight-jupyter: cuda-image
	$(call podman_run_cuda,$(notebook_args),start_jupyter)

# Open a shell in the NVIDIA Nsight container (with no-GPU support)
.PHONY: nsight-console
nsight-console: NOTEBOOK_PORT ?= 8889
nsight-console: cuda-image
	$(call podman_run_cuda,$(notebook_args),)


# Run the NVIDIA Nsight Systems UI (with GPU support)
.PHONY: cuda-systems
cuda-systems: cuda-image
	$(call podman_run_cuda,$(cuda_args),nsys-ui)

# Run the NVIDIA Nsight Systems UI (with GPU support)
.PHONY: cuda-compute
cuda-compute: cuda-image
	$(call podman_run_cuda,$(cuda_args),ncu-ui)

# Run a Jupyter Notebook Server (with GPU support)
.PHONY: cuda-jupyter
cuda-jupyter: NOTEBOOK_PORT ?= 8888
cuda-jupyter: cuda-image
	$(call podman_run_cuda,$(cuda_args)$(notebook_args),start_jupyter)

# Open a shell in the NVIDIA Nsight container (with GPU support)
.PHONY: cuda-console
cuda-console: NOTEBOOK_PORT ?= 8889
cuda-console: cuda-image
	$(call podman_run_cuda,$(cuda_args)$(notebook_args),)


.PHONY: clean
clean:
	rm -rf workspace/.ipynb_checkpoints
