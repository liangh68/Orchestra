#!/bin/bash
output=$(python gpu_monitor.py --duration 3 --interval 0.5 --with_filter --threshold 60)

gpu_num=$(echo "$output" | sed -n '1p')
gpu_ids=$(echo "$output" | sed -n '2p')
gpu_utils=$(echo "$output" | sed -n '3p')

echo "Available GPU num: $gpu_num"
echo "Available GPU id: $gpu_ids"
echo "GPU utils: $gpu_utils"
