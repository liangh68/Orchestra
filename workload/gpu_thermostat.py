import torch
import pynvml
import time
import threading
import random
import argparse

# Initialize NVML
pynvml.nvmlInit()

class GPULoadControllerBysize:
    # Configuration constants
    CHECK_INTERVAL = 0.5  # Check interval (seconds)
    STABILIZATION_TIME = 5.0  # Stabilization duration (seconds)
    ERROR_HISTORY_SIZE = 10  # Error history size
    MIN_SIZE = 512  # Minimum tensor size
    MAX_STEP = 32768  # Maximum adjustment step
    MIN_STEP = 32  # Minimum adjustment step
    STEP_MULTIPLIER = 64  # Step multiplier
    def __init__(self, gpu_index, target_util, allowable_error, fixed_size=0, data_size=None):
        """
        gpu_index: GPU index
        target_util: Target utilization
        data_size: Fixed data size, if provided use fixed size, no adaptive adjustment
        """
        self.gpu_index = gpu_index
        self.target_util = target_util
        self.allowable_error = allowable_error
        self.fixed_size = fixed_size
        if fixed_size == 1:
            assert data_size
            self.data_size = data_size
        else:
            self.data_size = None
        
        # Check GPU availability
        device_count = pynvml.nvmlDeviceGetCount()
        if gpu_index >= device_count:
            raise ValueError(f"GPU index {gpu_index} exceeds available devices count ({device_count})")
        
        # Initialize controller state
        initial_size = self.data_size if self.data_size is not None else 4096
        self.controller = {
            'target_util': target_util,
            'size': initial_size,
            'data': None,
            'stabilization_achieved': False,
            'stable_start_time': None,
            'best_size': None,
            'computation_factor': int(target_util * 0.8),
            'error_history': [],
            'adjustment_lock': threading.Lock()
        }
        self.thread = None
    
    def _gpu_worker(self):
        """GPU load control thread"""
        ctrl = self.controller
        
        try:
            # Set GPU for current thread
            torch.cuda.set_device(self.gpu_index)
            device = torch.device(f"cuda:{self.gpu_index}")
            
            # Get GPU handle
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            if self.data_size is not None:
                print(f"🚀 GPU {self.gpu_index}: Starting fixed-size computation (size: {ctrl['size']})")
            else:
                print(f"🚀 GPU {self.gpu_index}: Starting load adjustment to {ctrl['target_util']}%")
            
            # Initialize tensor
            with ctrl['adjustment_lock']:
                ctrl['data'] = torch.randn(ctrl['size'], device=device, dtype=torch.float32).abs()
        
            allowable_error = self.allowable_error
            consecutive_count = 0
            last_state = 1
            check_last = time.time()
            
            while not ctrl.get('stop_signal', False):
                # Execute compute-intensive operations (not in lock, improve concurrency)
                with ctrl['adjustment_lock']:
                    data = ctrl['data']
                
                result = data
                for _ in range(ctrl['computation_factor']):
                    # Mix multiple operations
                    rand_val = random.random()
                    if rand_val < 0.3:
                        result = torch.sin(result) * torch.cos(result)
                    elif rand_val < 0.6:
                        result = result * torch.exp(result * 0.1)
                    else:
                        result = result * 0.8 + torch.tanh(result) * 0.2
                
                # Stability detection and adjustment (state update in lock)
                current_time = time.time()
                if current_time - check_last > self.CHECK_INTERVAL:
                    try:
                        # Get current utilization
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        error = abs(util - ctrl['target_util'])
                    except Exception as e:
                        print(f"⚠️ GPU {self.gpu_index}: Failed to get utilization: {e}")
                        check_last = current_time
                        continue
                    
                    with ctrl['adjustment_lock']:
                        # If fixed size is provided, only print utilization, no adaptive adjustment
                        if self.data_size is not None:
                            print(f"🟢 GPU {self.gpu_index}:  \t{ctrl['size']} \t({util}% / {ctrl['target_util']}%)")
                            # Periodically regenerate tensor to simulate memory allocation overhead in adaptive mode
                            # This maintains similar GPU utilization as in adaptive mode
                            ctrl['data'] = torch.randn(ctrl['size'], device=device, dtype=torch.float32).abs()
                        else:
                            # Record error history (for adaptive adjustment)
                            ctrl['error_history'].append(error)
                            if len(ctrl['error_history']) > self.ERROR_HISTORY_SIZE:
                                ctrl['error_history'].pop(0)
                            
                            # Stability detection
                            if error <= allowable_error:
                                if ctrl['stable_start_time'] is None:
                                    ctrl['stable_start_time'] = current_time
                                    ctrl['best_size'] = ctrl['size']
                                elif current_time - ctrl['stable_start_time'] >= self.STABILIZATION_TIME and not ctrl['stabilization_achieved']:
                                    ctrl['stabilization_achieved'] = True
                                    ctrl['best_size'] = ctrl['size']
                                    print(f"\n⭐ GPU {self.gpu_index}: STABILIZED at {util:.1f}% "
                                        f"(size: {ctrl['size']}, target: {ctrl['target_util']}%) ⭐\n")
                            else:
                                ctrl['stable_start_time'] = None
                            
                            # Adjustment logic (only when not stabilized)
                            if not ctrl['stabilization_achieved']:
                                # Adaptive step adjustment
                                if ctrl['error_history']:
                                    # Copy list to avoid modifying original data
                                    sorted_errors = sorted(ctrl['error_history'])
                                    # Remove one highest and one lowest value
                                    if len(sorted_errors) >= 3:
                                        trimmed_errors = sorted_errors[1:-1]  # Remove first and last elements
                                    else:
                                        trimmed_errors = sorted_errors  # Don't remove if less than 3 elements
                                    # Calculate trimmed mean
                                    avg_error = sum(trimmed_errors) / len(trimmed_errors)
                                else:
                                    avg_error = 0
                                adjust_step = min(self.MAX_STEP, max(self.MIN_STEP, int(avg_error) * self.STEP_MULTIPLIER))
                                
                                if util < ctrl['target_util'] - allowable_error:
                                    consecutive_count = consecutive_count + 1 if last_state == 1 else 0
                                    ctrl['size'] += int(adjust_step * 2 ** consecutive_count)
                                    last_state = 1
                                    print(f"🔴 GPU {self.gpu_index}:  \t{ctrl['size']} \t({util}% / {ctrl['target_util']}%)")
                                elif util > ctrl['target_util'] + allowable_error:
                                    consecutive_count = consecutive_count + 1 if last_state == -1 else 1
                                    ctrl['size'] = max(self.MIN_SIZE, ctrl['size'] - int(adjust_step * 2 ** consecutive_count))
                                    last_state = -1
                                    print(f"🔵 GPU {self.gpu_index}:  \t{ctrl['size']} \t({util}% / {ctrl['target_util']}%)")
                                else:
                                    consecutive_count = 0
                                    last_state = 0
                                    print(f"🟢 GPU {self.gpu_index}:  \t{ctrl['size']} \t({util}% / {ctrl['target_util']}%)")
                                
                                # Update tensor
                                ctrl['data'] = torch.randn(ctrl['size'], device=device, dtype=torch.float32).abs()
                    
                    check_last = current_time
        except Exception as e:
            print(f"❌ GPU {self.gpu_index}: Worker thread error: {e}")
            
            
    
    def start(self):
        """Start GPU load thread"""
        thread = threading.Thread(target=self._gpu_worker)
        thread.daemon = True
        thread.start()
        self.thread = thread
    
    def stop(self):
        """Stop GPU load thread"""
        self.controller['stop_signal'] = True
        if self.thread:
            self.thread.join(timeout=2.0)
        print("✅ GPU load controller stopped")
    
    def get_status(self):
        """Get GPU status information"""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        except Exception as e:
            print(f"⚠️ GPU {self.gpu_index}: Failed to get status: {e}")
            util = 0
        
        with self.controller['adjustment_lock']:
            return {
                "gpu_index": self.gpu_index,
                "current_util": util,
                "target_util": self.controller['target_util'],
                "size": self.controller['size'],
                "status": "STABILIZED" if self.controller['stabilization_achieved'] else "ADJUSTING",
                "error": round(abs(util - self.controller['target_util']), 1)
            }


class GPULoadController2:
    # Configuration constants
    CHECK_INTERVAL = 0.5  # Check interval (seconds)
    STABILIZATION_TIME = 5.0  # Stabilization duration (seconds)
    ERROR_HISTORY_SIZE = 10  # Error history size
    PERIOD = 0.1  # Period (seconds), default 100ms
    MIN_COMPUTE_TIME = 0.001  # Minimum compute time (seconds), 1ms
    MAX_COMPUTE_TIME_RATIO = 0.99  # Maximum compute time ratio, avoid fully occupying period
    MIN_STEP_MS = 1  # Minimum adjustment step (milliseconds)
    MAX_STEP_MS = 10  # Maximum adjustment step (milliseconds)
    STEP_MULTIPLIER = 0.5  # Step multiplier
    
    def __init__(self, gpu_index, target_util, allowable_error, period=None, data_size=None, fixed_ratio=0, compute_ratio=None):
        """
        gpu_index: GPU index
        target_util: Target utilization
        allowable_error: Allowable error range
        period: Period time (seconds), if provided use it, otherwise use default
        data_size: Fixed tensor size for computation
        fixed_ratio: Whether to use fixed ratio, 1 means use fixed ratio, 0 means adaptive adjustment
        compute_ratio: Fixed compute ratio (0-1), must be provided when fixed_ratio=1
        """
        self.gpu_index = gpu_index
        self.target_util = target_util
        self.allowable_error = allowable_error
        self.period = period if period is not None else self.PERIOD
        self.fixed_ratio = fixed_ratio
        if fixed_ratio == 1:
            assert compute_ratio is not None, "compute_ratio must be provided when fixed_ratio=1"
            assert 0 < compute_ratio <= 1, "compute_ratio must be between 0 and 1"
            self.compute_ratio = compute_ratio
        else:
            self.compute_ratio = None
        assert data_size
        self.data_size = data_size
        
        # Check GPU availability
        device_count = pynvml.nvmlDeviceGetCount()
        if gpu_index >= device_count:
            raise ValueError(f"GPU index {gpu_index} exceeds available devices count ({device_count})")
        
        # Initialize controller state
        # If using fixed ratio, compute_time = compute_ratio * period
        # Otherwise, initial compute_time = target utilization * period
        if self.fixed_ratio == 1:
            initial_compute_time = self.compute_ratio * self.period
        else:
            initial_compute_time = (target_util / 100.0) * self.period
        initial_compute_time = max(self.MIN_COMPUTE_TIME, 
                                  min(initial_compute_time, self.period * self.MAX_COMPUTE_TIME_RATIO))
        
        self.controller = {
            'target_util': target_util,
            'compute_time': initial_compute_time,  # Compute time per period (seconds)
            'period': self.period,  # Period time (seconds)
            'data': None,
            'stabilization_achieved': False,
            'stable_start_time': None,
            'best_compute_time': None,
            'computation_factor': int(target_util * 0.8),
            'error_history': [],
            'adjustment_lock': threading.Lock()
        }
        self.thread = None
    
    def _gpu_worker(self):
        """GPU load control thread"""
        ctrl = self.controller
        
        try:
            # Set GPU for current thread
            torch.cuda.set_device(self.gpu_index)
            device = torch.device(f"cuda:{self.gpu_index}")
            
            # Get GPU handle
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            if self.fixed_ratio == 1:
                compute_ratio = (ctrl['compute_time'] / ctrl['period']) * 100
                print(f"🚀 GPU {self.gpu_index}: Starting fixed-ratio computation "
                      f"(period: {ctrl['period']*1000:.1f}ms, compute_ratio: {compute_ratio:.1f}%)")
            else:
                print(f"🚀 GPU {self.gpu_index}: Starting busy-idle pattern adjustment to {ctrl['target_util']}% "
                      f"(period: {ctrl['period']*1000:.1f}ms)")
            
            # Initialize tensor
            with ctrl['adjustment_lock']:
                ctrl['data'] = torch.randn(self.data_size, device=device, dtype=torch.float32).abs()
            
            allowable_error = self.allowable_error
            consecutive_count = 0
            last_state = 1
            check_last = time.time()
            
            while not ctrl.get('stop_signal', False):
                # Busy-idle alternating pattern: compute first, then sleep in each period
                period_start = time.time()
                
                # Compute phase
                compute_start = time.time()
                with ctrl['adjustment_lock']:
                    data = ctrl['data']
                    compute_time = ctrl['compute_time']
                
                result = data
                # Continue computation within compute time
                while (time.time() - compute_start) < compute_time:
                    for _ in range(ctrl['computation_factor']):
                        # Mix multiple operations
                        rand_val = random.random()
                        if rand_val < 0.3:
                            result = torch.sin(result) * torch.cos(result)
                        elif rand_val < 0.6:
                            result = result * torch.exp(result * 0.1)
                        else:
                            result = result * 0.8 + torch.tanh(result) * 0.2
                
                # Idle phase (sleep)
                elapsed = time.time() - period_start
                sleep_time = ctrl['period'] - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Stability detection and adjustment (state update in lock)
                current_time = time.time()
                if current_time - check_last > self.CHECK_INTERVAL:
                    try:
                        # Get current utilization
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        error = abs(util - ctrl['target_util'])
                    except Exception as e:
                        print(f"⚠️ GPU {self.gpu_index}: Failed to get utilization: {e}")
                        check_last = current_time
                        continue
                    
                    with ctrl['adjustment_lock']:
                        # If fixed ratio is used, only print utilization, no adaptive adjustment
                        if self.fixed_ratio == 1:
                            compute_ratio = (ctrl['compute_time'] / ctrl['period']) * 100
                            print(f"🟢 GPU {self.gpu_index}:  \t{ctrl['compute_time']*1000:.2f}ms/{ctrl['period']*1000:.1f}ms "
                                f"({compute_ratio:.1f}%) \t({util}% / {ctrl['target_util']}%)")
                        else:
                            # Record error history (for adaptive adjustment)
                            ctrl['error_history'].append(error)
                            if len(ctrl['error_history']) > self.ERROR_HISTORY_SIZE:
                                ctrl['error_history'].pop(0)
                            
                            # Stability detection
                            if error <= allowable_error:
                                if ctrl['stable_start_time'] is None:
                                    ctrl['stable_start_time'] = current_time
                                    ctrl['best_compute_time'] = ctrl['compute_time']
                                elif current_time - ctrl['stable_start_time'] >= self.STABILIZATION_TIME and not ctrl['stabilization_achieved']:
                                    ctrl['stabilization_achieved'] = True
                                    ctrl['best_compute_time'] = ctrl['compute_time']
                                    compute_ratio = (ctrl['compute_time'] / ctrl['period']) * 100
                                    print(f"\n⭐ GPU {self.gpu_index}: STABILIZED at {util:.1f}% "
                                        f"(compute_time: {ctrl['compute_time']*1000:.2f}ms, "
                                        f"ratio: {compute_ratio:.1f}%, target: {ctrl['target_util']}%) ⭐\n")
                            else:
                                ctrl['stable_start_time'] = None
                            
                            # Adjustment logic (only when not stabilized)
                            if not ctrl['stabilization_achieved']:
                                # Adaptive step adjustment
                                if ctrl['error_history']:
                                    # Copy list to avoid modifying original data
                                    sorted_errors = sorted(ctrl['error_history'])
                                    # Remove one highest and one lowest value
                                    if len(sorted_errors) >= 3:
                                        trimmed_errors = sorted_errors[1:-1]  # Remove first and last elements
                                    else:
                                        trimmed_errors = sorted_errors  # Don't remove if less than 3 elements
                                    # Calculate trimmed mean
                                    avg_error = sum(trimmed_errors) / len(trimmed_errors)
                                else:
                                    avg_error = 0
                                
                                # Calculate adjustment step (ms to seconds)
                                adjust_step_ms = min(self.MAX_STEP_MS, max(self.MIN_STEP_MS, int(avg_error * self.STEP_MULTIPLIER)))
                                adjust_step = adjust_step_ms / 1000.0  # Convert to seconds
                                
                                if util < ctrl['target_util'] - allowable_error:
                                    # Utilization too low, increase compute time
                                    consecutive_count = consecutive_count + 1 if last_state == 1 else 0
                                    new_compute_time = ctrl['compute_time'] + adjust_step * (2 ** consecutive_count)
                                    ctrl['compute_time'] = min(new_compute_time, 
                                                              ctrl['period'] * self.MAX_COMPUTE_TIME_RATIO)
                                    last_state = 1
                                    compute_ratio = (ctrl['compute_time'] / ctrl['period']) * 100
                                    print(f"🔴 GPU {self.gpu_index}:  \t{ctrl['compute_time']*1000:.2f}ms/{ctrl['period']*1000:.1f}ms "
                                        f"({compute_ratio:.1f}%) \t({util}% / {ctrl['target_util']}%)")
                                elif util > ctrl['target_util'] + allowable_error:
                                    # Utilization too high, decrease compute time
                                    consecutive_count = consecutive_count + 1 if last_state == -1 else 1
                                    new_compute_time = ctrl['compute_time'] - adjust_step * (2 ** consecutive_count)
                                    ctrl['compute_time'] = max(self.MIN_COMPUTE_TIME, new_compute_time)
                                    last_state = -1
                                    compute_ratio = (ctrl['compute_time'] / ctrl['period']) * 100
                                    print(f"🔵 GPU {self.gpu_index}:  \t{ctrl['compute_time']*1000:.2f}ms/{ctrl['period']*1000:.1f}ms "
                                        f"({compute_ratio:.1f}%) \t({util}% / {ctrl['target_util']}%)")
                                else:
                                    consecutive_count = 0
                                    last_state = 0
                                    compute_ratio = (ctrl['compute_time'] / ctrl['period']) * 100
                                    print(f"🟢 GPU {self.gpu_index}:  \t{ctrl['compute_time']*1000:.2f}ms/{ctrl['period']*1000:.1f}ms "
                                        f"({compute_ratio:.1f}%) \t({util}% / {ctrl['target_util']}%)")
                    
                    check_last = current_time
        except Exception as e:
            print(f"❌ GPU {self.gpu_index}: Worker thread error: {e}")
    
    def start(self):
        """Start GPU load thread"""
        thread = threading.Thread(target=self._gpu_worker)
        thread.daemon = True
        thread.start()
        self.thread = thread
    
    def stop(self):
        """Stop GPU load thread"""
        self.controller['stop_signal'] = True
        if self.thread:
            self.thread.join(timeout=2.0)
        print("✅ GPU load controller stopped")
    
    def get_status(self):
        """Get GPU status information"""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        except Exception as e:
            print(f"⚠️ GPU {self.gpu_index}: Failed to get status: {e}")
            util = 0
        
        with self.controller['adjustment_lock']:
            compute_ratio = (self.controller['compute_time'] / 
                           self.controller['period']) * 100
            return {
                "gpu_index": self.gpu_index,
                "current_util": util,
                "target_util": self.controller['target_util'],
                "compute_time": self.controller['compute_time'],
                "period": self.controller['period'],
                "compute_ratio": compute_ratio,
                "status": "STABILIZED" if self.controller['stabilization_achieved'] else "ADJUSTING",
                "error": round(abs(util - self.controller['target_util']), 1)
            }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Single GPU load controller')
    parser.add_argument('--gpu_index', type=int, required=True,
                        help='GPU index')
    parser.add_argument('--target_util', type=int, required=True,
                        help='Target utilization (percentage)')
    parser.add_argument('--control_type', type=str, choices=['bytime', 'bysize'], default='bytime',
                        help='Control type: bytime (time-based) or bysize (size-based)')
    parser.add_argument('--allowable_error', type=int, default=0,
                        help='Allowable error range (percentage)')

    parser.add_argument('--fixed_ratio', type=int, choices=[0, 1], default=0,
                        help='Whether to use fixed ratio (only for bytime mode)')
    parser.add_argument('--compute_ratio', type=float, default=0.85,
                        help='Fixed compute ratio (0-1, only for bytime mode when fixed_ratio=1)')
    parser.add_argument('--compute_period', type=float, default=0.1,
                        help='Compute period (seconds, only for bytime mode)')
                        
    parser.add_argument('--fixed_size', type=int, choices=[0, 1], default=0,
                        help='Whether to use fixed size (only for bysize mode)')
    parser.add_argument('--data_size', type=int, default=3000000,
                        help='Fixed data size (for fixed size in bysize mode, or tensor size in bytime mode)')
    args = parser.parse_args()
    
    # Create controller
    if args.control_type == 'bysize':
        controller = GPULoadControllerBysize(args.gpu_index, args.target_util, args.allowable_error, 
                                       fixed_size=args.fixed_size, data_size=args.data_size)
    elif args.control_type == 'bytime':
        controller = GPULoadController2(args.gpu_index, args.target_util, args.allowable_error, 
                                       period=args.compute_period, 
                                       data_size=args.data_size, fixed_ratio=args.fixed_ratio, 
                                       compute_ratio=args.compute_ratio)
    
    try:
        # Start load generation
        controller.start()
        
        # If using fixed size or fixed ratio, directly enter monitoring loop; otherwise wait for stabilization
        if (args.control_type == 'bysize' and args.fixed_size == 1) or \
           (args.control_type == 'bytime' and args.fixed_ratio == 1):
            if args.control_type == 'bysize':
                print("\n🔥 Using fixed data size. Press Ctrl+C to stop.")
            else:
                print("\n🔥 Using fixed compute ratio. Press Ctrl+C to stop.")
            # Keep running
            while True:
                time.sleep(1)
        else:
            # Monitor status
            while not controller.controller['stabilization_achieved']:
                time.sleep(1)
                status = controller.get_status()
                # print(f"\nCurrent Status: GPU {status['gpu_index']}: {status['current_util']}% "
                #       f"(target: {status['target_util']}%, {status['status']}, "
                #       f"error: {status['error']}%)")
            
            print("\n🔥 GPU has stabilized! Press Ctrl+C to stop.")
            
            # Keep running
            while True:
                time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        controller.stop()
        try:
            pynvml.nvmlShutdown()
        except:
            pass
