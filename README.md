# gpu-util-demo

Small experiments for understanding what “GPU utilization” really means, and why different tools answer different questions.

This repo is organized around several measurement layers:

- **NVML / `nvidia-smi`** for coarse device-level busy-time signals
- **DCGM** for aggregated hardware telemetry such as `SM_ACTIVE` and `SM_OCCUPANCY`
- **Nsight Systems** for CPU–GPU timeline structure
- **Nsight Compute** for kernel-level efficiency and bottleneck analysis

The repository currently contains the top-level directories `.vscode`, `DCGM`, `cpu_util`, `nsight_compute`, `nsight_systems`, and `nvml`.  [oai_citation:0‡GitHub](https://github.com/852866031/gpu-util-demo)

## Repository layout

### `cpu_util/`
Contains a simple CPU utilization demo (`cpu_util_demo.py`) used as a conceptual baseline before moving to GPU metrics.  [oai_citation:1‡GitHub](https://github.com/852866031/gpu-util-demo/tree/main/cpu_util)

### `nvml/`
Contains a CUDA workload (`two_cases.cu`), a small NVML Python example (`nvml.py`), and a `Makefile`. This directory is used to demonstrate how coarse GPU-util numbers can hide very different workload structures.  [oai_citation:2‡GitHub](https://github.com/852866031/gpu-util-demo/tree/main/nvml)

### `DCGM/`
Contains `dcgm_trace.py` plus a directory README. This part of the repo shows how to install DCGM and how to trace a real workload while collecting metrics such as `SM_ACTIVE`, `SM_OCCUPANCY`, `TENSOR_ACTIVE`, and `DRAM_ACTIVE`.  [oai_citation:3‡GitHub](https://github.com/852866031/gpu-util-demo/tree/main/DCGM)

### `nsight_compute/`
Contains a two-GPU kernel experiment (`workload_two_gpu.py`), DCGM helpers, a tuner, plotting code, a `Makefile`, and a local README. The goal is to show that similar aggregated GPU activity can still correspond to very different kernel efficiency.  [oai_citation:4‡GitHub](https://github.com/852866031/gpu-util-demo/tree/main/nsight_compute)

### `nsight_systems/`
Contains a two-GPU inference-pipeline example (`inf_sys.py`), NVML/DCGM runners, combined plotting code, a `Makefile`, and a local README. The goal is to show that similar GPU activity does not necessarily imply similar productive or service-level utilization.  [oai_citation:5‡GitHub](https://github.com/852866031/gpu-util-demo/tree/main/nsight_systems)

## Create a conda environment

A minimal environment for the Python parts of this repo:

```bash
conda create -n gpu-util-demo python=3.11 -y
conda activate gpu-util-demo
pip install torch matplotlib nvidia-ml-py

For directories that use custom CUDA kernels from Python, also install CuPy:

pip install cupy-cuda13x

If your system is on CUDA 12 instead of CUDA 13, replace cupy-cuda13x with cupy-cuda12x.

System tools you may also need

Some subdirectories require NVIDIA system tools in addition to Python packages:
	•	DCGM (dcgmi) for the DCGM/, nsight_compute/, and nsight_systems/ monitoring scripts. The DCGM directory includes Ubuntu 24.04 installation notes.  ￼
	•	Nsight Systems (nsys) for nsight_systems/.  ￼
	•	Nsight Compute (ncu) for nsight_compute/.  ￼

On many systems, Nsight Compute also requires permission to access NVIDIA performance counters.

Suggested workflow

1. Start with the coarse demos
	•	cpu_util/ to ground the CPU-side idea of utilization
	•	nvml/ to see how one coarse GPU-util number can hide very different cases

2. Move to aggregated hardware telemetry
	•	DCGM/ to collect SM_ACTIVE, SM_OCCUPANCY, and related metrics on a real workload

3. Compare kernel efficiency
	•	nsight_compute/ to see why similar aggregated activity does not imply similar kernel productivity

4. Compare pipeline efficiency
	•	nsight_systems/ to see why similar GPU activity does not imply similar throughput or latency at the pipeline level

Examples

Run the Nsight Systems inference example directly:

cd nsight_systems
make run

Collect NVML and DCGM measurements for that example:

make nvml
make dcgm
make plot-all

Run the two-kernel Nsight Compute example:

cd ../nsight_compute
make run
make plot
make ncu-gather
make ncu-dense
