# blog-triton-profiling

Repo containing blog demo materials for profiling Triton GPU kernels on
- CUDA
- ROCm

=======
## Triton Profiling

Triton profiling introduction demo materials and runtime container images.

### Requirements

- NVIDIA GPU for the NVIDIA blog
- AMD GPU for the AMD blog
- Podman or Docker
- make

NOTE: ROCm's Compute profiler only runs on AMD CDNA GPUs, i.e. MI300X (02-12-2026).

### Images

Container images that provide a runtime environment for the target GPU

- cuda                  - Requires an NVIDIA GPU
- nsight                - Does not require an NVIDIA GPU
- rocm                  - Requires an AMD GPU (i.e. MI300X)

NOTE: The nsight target only provides an environment to run the Nsight tools.

### Instructions

#### Build

##### With an NVIDIA GPU

```bash
make cuda-image
```

##### Without an NVIDIA GPU

```bash
make nsight-image
```

##### With an AMD GPU

```bash
make rocm-image
```

#### Runtime

All targets will start a new container and remove it when you exit.

##### CUDA/Nsight

###### Console

Runs the target image and leaves the user inside it at a bash prompt.

```bash
make [cuda | nsight]-console
```

###### Jupyter Notebook

The nsight image can only be used to view the Jupyter notebook,
an NVIDIA GPU and the cuda image are required to run it.

```bash
make [cuda | nsight]-jupyter
```

###### Nsight Systems

Runs the Nsight Systems UI.

```bash
make [cuda | nsight]-systems
```

###### Nsight Compute

Runs the Nsight Compute UI.

```bash
make [cuda | nsight]-compute
```

##### AMD ROCm

###### Console

Runs the target image and leaves the user inside it at a bash prompt.

```bash
make rocm-console
```

###### Jupyter Notebook

```bash
make rocm-jupyter
```
