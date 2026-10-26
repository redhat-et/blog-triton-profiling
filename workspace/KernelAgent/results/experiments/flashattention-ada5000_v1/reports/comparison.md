# Kernel comparison experiment: `flashattention-ada5000_v1`

- Status: **completed**
- Created: 2026-10-26T14:41:49.397753
- Report generated: 2026-10-26T17:22:56.657620
- Capture batch: `capture_20261026_212117_4e7d` (ncu, completed)

## Configuration

| Key | Value |
|---|---|
| Problems | flash_attention |
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

## flash_attention — benchmark case 0

Case: `(2, 8, 1024, 64, False, 'float16')`

### Benchmark (fixed harness, `do_bench` mean)

| Variant | Rounds used/budget | Fixed test | Status | Kernel (ms) | PyTorch (ms) | vs PyTorch | vs generated |
|---|---:|---|---|---:|---:|---:|---:|
| `generated` | 1/5 | PASS | verified | 0.1338 | 0.0677 | 0.51x | 1.00x |
| `rounds_5` | 5/5 | PASS | accepted | 0.4374 | 0.3296 | 0.75x | 0.31x |
| `rounds_10` | 10/10 | PASS | accepted | 0.3072 | 0.3540 | 1.15x | 0.44x |
| `vllm_triton_prefill` *(external baseline)* | — | PASS | verified | 0.0818 | 0.0638 | 0.78x | 1.64x |

Optimizer acceptance timings (the comparison each optimization session gated on, case 0):

| Variant | Source (ms) | Candidate (ms) | Accepted speedup |
|---|---:|---:|---:|
| `rounds_5` | 0.4419 | 0.4374 | 1.01x |
| `rounds_10` | 0.3816 | 0.3072 | 1.24x |

### Nsight Compute (`ncu`)

`ncu --set full`, 10 measured iterations; duration is the mean over profiled launches, throughput and occupancy are per-launch means (% of peak).

| Variant | Kernel | Launches | Duration avg (µs) | SM % | Memory % | DRAM % | Occupancy % | Reg/thread |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `generated` | `_flash_attention_kernel` | 10 | 185.45 | 41.8 | 40.7 | 7.2 | 16.2 | 64 |
| `rounds_5` | `_flash_attention_kernel` | 10 | 72.62 | 54.6 | 93.8 | 16.7 | 13.0 | 248 |
| `rounds_10` | `_flash_attention` | 10 | 68.60 | 58.1 | 96.4 | 17.1 | 13.3 | 244 |
| `vllm_triton_prefill` | `_fwd_kernel` | 10 | 80.74 | 49.5 | 90.4 | 16.1 | 13.9 | 255 |

#### Artifacts

- `generated` · kernel: [final_kernel.py](../problems/flash_attention/generated/flash_attention/final_kernel.py) · sha256 `d1c93f8eb483…` · record: [verify.json](../problems/flash_attention/generated/flash_attention/verify.json) · summary: [generate_result.json](../problems/flash_attention/generated/flash_attention/generate_result.json) · ncu: [generated_ncu_case0.ncu-rep](../captures/capture_20261026_212117_4e7d/flash_attention/generated_ncu_case0.ncu-rep)
- `rounds_5` · kernel: [optimized_kernel.py](../problems/flash_attention/case_0/rounds_5/flash_attention/optimized_kernel.py) · sha256 `138f765d4df3…` · record: [verify_optimized.json](../problems/flash_attention/case_0/rounds_5/flash_attention/verify_optimized.json) · summary: [optimize_result.json](../problems/flash_attention/case_0/rounds_5/flash_attention/optimize_result.json) · ncu: [rounds_5_ncu_case0.ncu-rep](../captures/capture_20261026_212117_4e7d/flash_attention/rounds_5_ncu_case0.ncu-rep)
- `rounds_10` · kernel: [optimized_kernel.py](../problems/flash_attention/case_0/rounds_10/flash_attention/optimized_kernel.py) · sha256 `9b1431e00661…` · record: [verify_optimized.json](../problems/flash_attention/case_0/rounds_10/flash_attention/verify_optimized.json) · summary: [optimize_result.json](../problems/flash_attention/case_0/rounds_10/flash_attention/optimize_result.json) · ncu: [rounds_10_ncu_case0.ncu-rep](../captures/capture_20261026_212117_4e7d/flash_attention/rounds_10_ncu_case0.ncu-rep)
- `vllm_triton_prefill` · kernel: [kernel.py](../problems/flash_attention/baselines/vllm_triton_prefill/flash_attention/kernel.py) · sha256 `2c38b86851dd…` · source: `flash_attention/baselines/vllm_triton_prefill.py` · record: [verify_vllm_triton_prefill.json](../problems/flash_attention/baselines/vllm_triton_prefill/flash_attention/verify_vllm_triton_prefill.json) · ncu: [vllm_triton_prefill_ncu_case0.ncu-rep](../captures/capture_20261026_212117_4e7d/flash_attention/vllm_triton_prefill_ncu_case0.ncu-rep)

## Source diffs

- flash_attention: `generated` → `rounds_5`: [flash_attention__generated_to_rounds_5.diff](source-diffs/flash_attention__generated_to_rounds_5.diff)
- flash_attention: `generated` → `rounds_10`: [flash_attention__generated_to_rounds_10.diff](source-diffs/flash_attention__generated_to_rounds_10.diff)
- flash_attention: `generated` → `vllm_triton_prefill`: [flash_attention__generated_to_vllm_triton_prefill.diff](source-diffs/flash_attention__generated_to_vllm_triton_prefill.diff)
- flash_attention: `rounds_5` → `rounds_10`: [flash_attention__rounds_5_to_rounds_10.diff](source-diffs/flash_attention__rounds_5_to_rounds_10.diff)
- flash_attention: `rounds_5` → `vllm_triton_prefill`: [flash_attention__rounds_5_to_vllm_triton_prefill.diff](source-diffs/flash_attention__rounds_5_to_vllm_triton_prefill.diff)
- flash_attention: `rounds_10` → `vllm_triton_prefill`: [flash_attention__rounds_10_to_vllm_triton_prefill.diff](source-diffs/flash_attention__rounds_10_to_vllm_triton_prefill.diff)

## Interpreting the numbers

- Benchmark latencies come from the fixed `bench.py` harness (`triton.testing.do_bench` means over the exact case shapes and dtype); profiler durations come from each tool's own measurement of the captured launches and are not interchangeable with them.
- `rounds used/budget` reports the optimizer's completed rounds against the requested budget (`KA_ROUNDS` is a maximum; a run may stop earlier).
- Every optimization budget started from the same generated kernel (hash-checked); rejected or failed attempts keep their outcome and are never replaced by another variant's kernel.

