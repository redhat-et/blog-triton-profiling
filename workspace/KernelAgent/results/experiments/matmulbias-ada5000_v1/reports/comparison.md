# Kernel comparison experiment: `matmulbias-ada5000_v1`

- Status: **completed**
- Created: 2026-10-26T14:09:36.676368
- Report generated: 2026-10-26T15:36:03.260624
- Capture batch: `capture_20261026_153544_0129` (ncu, completed)

## Configuration

| Key | Value |
|---|---|
| Problems | matmul_bias |
| Benchmark case | 0 |
| Generation rounds (budget) | 5 |
| Optimization budgets | 5, 10 |
| Workers | 1 |
| Generation LLM | openai / gpt-6-luna |
| Optimization LLM | openai / gpt-6.1-sol |
| Profilers | ncu |

## Environment

| Component | Version |
|---|---|
| GPU | NVIDIA RTX 5000 Ada Generation Laptop GPU |
| GPU memory | 16376 MiB |
| Driver | 615.78.08 |
| CUDA | nvcc: NVIDIA (R) Cuda compiler driver |
| PyTorch | 2.13.0+cu129 |
| Triton | 3.7.1 |
| KernelAgent | 0.1.0 |
| Nsight Compute | NVIDIA (R) Nsight Compute Command Line Profiler — 2025.2.1.0 |
| Nsight Systems | NVIDIA Nsight Systems version 2024.6.2.225-246235244400v0 |
| Python | 3.12.15 |
| Repository | — |

## matmul_bias — benchmark case 0

Case: `(1024, 1024, 1024, 'float16')`

### Benchmark (fixed harness, `do_bench` mean)

| Variant | Rounds used/budget | Fixed test | Status | Kernel (ms) | PyTorch (ms) | vs PyTorch | vs generated |
|---|---:|---|---|---:|---:|---:|---:|
| `generated` | 1/5 | PASS | verified | 0.0404 | 0.0262 | 0.65x | 1.00x |
| `rounds_5` | 5/5 | PASS | accepted | 0.0253 | 0.0241 | 0.96x | 1.60x |
| `rounds_10` | 8/10 | PASS | accepted | 0.0249 | 0.0243 | 0.97x | 1.62x |
| `rocm_blog_fusion` *(external baseline)* | — | PASS | verified | 0.0429 | 0.1819 | 4.24x | 0.94x |

Optimizer acceptance timings (the comparison each optimization session gated on, case 0):

| Variant | Source (ms) | Candidate (ms) | Accepted speedup |
|---|---:|---:|---:|
| `rounds_5` | 0.0403 | 0.0253 | 1.59x |
| `rounds_10` | 0.0404 | 0.0249 | 1.62x |

### Nsight Compute (`ncu`)

`ncu --set full`, 10 measured iterations; duration is the mean over profiled launches, throughput and occupancy are per-launch means (% of peak).

| Variant | Kernel | Launches | Duration avg (µs) | SM % | Memory % | DRAM % | Occupancy % | Reg/thread |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `generated` | not captured | | | | | | | |
| `rounds_5` | not captured | | | | | | | |
| `rounds_10` | not captured | | | | | | | |
| `rocm_blog_fusion` | `matmulbias_kernel` | 10 | 34.96 | 59.4 | 136.1 | 24.2 | 8.3 | 255 |

#### Artifacts

- `generated` · kernel: [final_kernel.py](../problems/matmul_bias/generated/matmul_bias/final_kernel.py) · sha256 `5ba2fb340021…` · record: [verify.json](../problems/matmul_bias/generated/matmul_bias/verify.json) · summary: [generate_result.json](../problems/matmul_bias/generated/matmul_bias/generate_result.json)
- `rounds_5` · kernel: [optimized_kernel.py](../problems/matmul_bias/case_0/rounds_5/matmul_bias/optimized_kernel.py) · sha256 `e71fe54ed298…` · record: [verify_optimized.json](../problems/matmul_bias/case_0/rounds_5/matmul_bias/verify_optimized.json) · summary: [optimize_result.json](../problems/matmul_bias/case_0/rounds_5/matmul_bias/optimize_result.json)
- `rounds_10` · kernel: [optimized_kernel.py](../problems/matmul_bias/case_0/rounds_10/matmul_bias/optimized_kernel.py) · sha256 `30091c7800b1…` · record: [verify_optimized.json](../problems/matmul_bias/case_0/rounds_10/matmul_bias/verify_optimized.json) · summary: [optimize_result.json](../problems/matmul_bias/case_0/rounds_10/matmul_bias/optimize_result.json)
- `rocm_blog_fusion` · kernel: [kernel.py](../problems/matmul_bias/baselines/rocm_blog_fusion/matmul_bias/kernel.py) · sha256 `ef1f4165e29f…` · source: `matmul_bias/baselines/rocm_blog_fusion.py` · record: [verify_rocm_blog_fusion.json](../problems/matmul_bias/baselines/rocm_blog_fusion/matmul_bias/verify_rocm_blog_fusion.json) · ncu: [rocm_blog_fusion_ncu_case0.ncu-rep](../captures/capture_20261026_153544_0129/matmul_bias/rocm_blog_fusion_ncu_case0.ncu-rep)

## Source diffs

- matmul_bias: `generated` → `rounds_5`: [matmul_bias__generated_to_rounds_5.diff](source-diffs/matmul_bias__generated_to_rounds_5.diff)
- matmul_bias: `generated` → `rounds_10`: [matmul_bias__generated_to_rounds_10.diff](source-diffs/matmul_bias__generated_to_rounds_10.diff)
- matmul_bias: `generated` → `rocm_blog_fusion`: [matmul_bias__generated_to_rocm_blog_fusion.diff](source-diffs/matmul_bias__generated_to_rocm_blog_fusion.diff)
- matmul_bias: `rounds_5` → `rounds_10`: [matmul_bias__rounds_5_to_rounds_10.diff](source-diffs/matmul_bias__rounds_5_to_rounds_10.diff)
- matmul_bias: `rounds_5` → `rocm_blog_fusion`: [matmul_bias__rounds_5_to_rocm_blog_fusion.diff](source-diffs/matmul_bias__rounds_5_to_rocm_blog_fusion.diff)
- matmul_bias: `rounds_10` → `rocm_blog_fusion`: [matmul_bias__rounds_10_to_rocm_blog_fusion.diff](source-diffs/matmul_bias__rounds_10_to_rocm_blog_fusion.diff)

## Interpreting the numbers

- Benchmark latencies come from the fixed `bench.py` harness (`triton.testing.do_bench` means over the exact case shapes and dtype); profiler durations come from each tool's own measurement of the captured launches and are not interchangeable with them.
- `rounds used/budget` reports the optimizer's completed rounds against the requested budget (`KA_ROUNDS` is a maximum; a run may stop earlier).
- Every optimization budget started from the same generated kernel (hash-checked); rejected or failed attempts keep their outcome and are never replaced by another variant's kernel.

