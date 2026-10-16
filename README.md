# Orchestra

## Environment Setup

This project requires Python 3.10 and PyTorch 2.9.1. Install dependencies using pip:

```bash
pip install -r requirements.txt
```

## Workload Simulation

The `workload/gpu_thermostat.py` script provides GPU load simulation capabilities for testing and benchmarking. It supports two control modes:

- **bysize**: Controls GPU utilization by adjusting tensor sizes adaptively
- **bytime**: Controls GPU utilization using a busy-idle pattern with adjustable compute time ratios

### Example Usage

1. **Adaptive mode (bysize):**
```bash
python workload/gpu_thermostat.py --gpu_index 0 --target_util 60 --control_type bysize --allowable_error 3
```

2. **Fixed-size mode (bysize):**
```bash
python workload/gpu_thermostat.py --gpu_index 0 --target_util 60 --control_type bysize --fixed_size 1 --data_size 3000000
```

3. **Adaptive mode (bytime):**
```bash
python workload/gpu_thermostat.py --gpu_index 0 --target_util 60 --control_type bytime --allowable_error 3 --data_size 3000000
```

4. **Fixed-ratio mode (bytime):**
```bash
python workload/gpu_thermostat.py --gpu_index 0 --target_util 60 --control_type bytime --fixed_ratio 1 --compute_ratio 0.85 --compute_period 0.1 --data_size 3000000
```

We recommend using the first method to simulate the load. If it is not stable enough, you can change the parameters to test the settings that best suit your hardware, and then use the other three methods to simulate the load.

## Running Inference

The `scripts/example.sh` script demonstrates how to run SDXL inference with automatic GPU selection and distributed execution.

### Example Usage

```bash
bash scripts/example.sh
```

The script will:
1. Automatically detect available GPUs with utilization below 85% using `orchestra/monitor/gpu_monitor.py`
2. Run SDXL inference with the selected GPUs
3. Support different parallelism strategies: `orche` (Orchestra), `patch` (Distrifusion), or `tensor` (Tensor Parallelism)

You can modify the script to:
- Change the model path
- Adjust GPU selection threshold (default: 85%)
- Switch between parallelism strategies