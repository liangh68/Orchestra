import pynvml
import argparse
import time
import sys

class GPUMonitor:
    def __init__(self):
        self.gpu_data = []
        self.total_gpus = 0
        
    def collect_gpu_info(self, duration=0, interval=1.0, monitor_gpus=None):
        
        pynvml.nvmlInit()
        self.total_gpus = pynvml.nvmlDeviceGetCount()
        if monitor_gpus is None or len(monitor_gpus) == 0:
            gpu_indices = list(range(self.total_gpus))
        else:
            gpu_indices = [idx for idx in monitor_gpus if 0 <= idx < self.total_gpus]
            invalid_gpus = set(monitor_gpus) - set(gpu_indices)
            if invalid_gpus:
                sys.stderr.write(f"Warning: Invalid GPU indices ignored: {', '.join(map(str, invalid_gpus))}\n")

        gpu_samples = [[] for _ in range(self.total_gpus)]
        
        if duration > 0 and interval > 0:
            sample_count = max(1, int(duration / interval))
            start_time = time.time()
            
            for _ in range(sample_count):
                for i in range(self.total_gpus):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    gpu_samples[i].append(util)
                
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - (elapsed % interval))
                time.sleep(sleep_time)
        
        for i in range(self.total_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            compute_mode = pynvml.nvmlDeviceGetComputeMode(handle)
            
            if gpu_samples[i]:
                avg_util = sum(gpu_samples[i]) / len(gpu_samples[i])
            else:
                avg_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            
            self.gpu_data.append({
                'index': i,
                'available': compute_mode != pynvml.NVML_COMPUTEMODE_PROHIBITED,
                'utilization': avg_util
            })
            
        pynvml.nvmlShutdown()
    
    def get_filtered_gpus(self, with_filter=False, threshold=70, monitor_gpus=None):
        if not self.gpu_data:
            return 0, "", ""  
            
        filtered = []
        for gpu in self.gpu_data:
            if gpu['available']:
                if with_filter:
                    if gpu['utilization'] <= threshold and gpu['index'] in monitor_gpus:
                        filtered.append(gpu)
                else:
                    filtered.append(gpu)
        
        if not filtered:
            return 0, "", ""  
        
        gpu_num = len(filtered)
        indices = ",".join(str(gpu['index']) for gpu in filtered)
        utils = ",".join(str(round(gpu['utilization'])) for gpu in filtered)
        return gpu_num, indices, utils

def main():
    parser = argparse.ArgumentParser(description='Enhanced GPU Monitor')
    parser.add_argument('--with_filter', action='store_true', 
                        help='Enable filtering by utilization threshold')
    parser.add_argument('--threshold', type=int, default=68, 
                        help='Utilization threshold (0-100)')
    parser.add_argument('--duration', type=float, default=0,
                        help='Sampling duration in seconds (0 for instant sampling)')
    parser.add_argument('--interval', type=float, default=0.5,
                        help='Sampling interval in seconds (default: 0.5)')
    parser.add_argument('--gpus', type=str, default="", 
                        help='Comma-separated list of GPU indices to monitor (e.g. "0,1,3")')
    
    args = parser.parse_args()
    
    if args.duration < 0 or args.interval <= 0:
        sys.stderr.write("Invalid arguments: duration must be >=0 and interval must be >0\n")
        print(0)
        print("")
        print("")
        return
    
    monitor_gpus = []
    if args.gpus:
        try:
            monitor_gpus = [int(idx.strip()) for idx in args.gpus.split(',')]
        except ValueError:
            sys.stderr.write(f"Invalid GPU list format: {args.gpus}\n")
            print(0)
            print("")
            print("")
            return

    monitor = GPUMonitor()
    error = monitor.collect_gpu_info(args.duration, args.interval, monitor_gpus)
    
    if error:
        print(0)
        print("")
        print("")
        return
    
    gpu_num, indices, utils = monitor.get_filtered_gpus(args.with_filter, args.threshold, monitor_gpus)
    
    print(gpu_num)
    print(indices)
    print(utils)

if __name__ == "__main__":
    main()