output=$(python ./orchestra/monitor/gpu_monitor.py --with_filter --threshold 85 --gpus 0,1)
gpu_num=$(echo "$output" | sed -n '1p')
gpu_ids=$(echo "$output" | sed -n '2p')
gpu_utils=$(echo "$output" | sed -n '3p')

model_path="path/to/your/SDXL" 

echo "***************** Bash Print *****************"
echo "Available GPU num: $gpu_num"
echo "Available GPU id: $gpu_ids"
echo "GPU utilizations: $gpu_utils"
echo "***************** Bash Print *****************"


echo "Orchestra"
CUDA_VISIBLE_DEVICES=$gpu_ids \
    python -W ignore -m torch.distributed.run --nproc_per_node=$gpu_num scripts/run_sdxl.py \
    --model_path $model_path\
    --output_path ./generated_images/example.png \
    --parallelism 'orche' \
    # --no_cuda_graph \
    # --no_split_batch \


# echo "Distrifusion"
# CUDA_VISIBLE_DEVICES=$gpu_ids \
#     python -W ignore -m torch.distributed.run --nproc_per_node=$gpu_num scripts/run_sdxl.py \
#     --model_path $model_path\
#     --output_path ./generated_images/example.png \
#     --parallelism 'patch' \
#     --no_cuda_graph \
#     --no_split_batch


# echo "Tensor Parallelism"
# CUDA_VISIBLE_DEVICES=$gpu_ids \
#     python -W ignore -m torch.distributed.run --nproc_per_node=$gpu_num scripts/run_sdxl.py \
#     --model_path $model_path\
#     --output_path ./generated_images/example.png \
#     --parallelism 'tensor' \
#     --no_cuda_graph \
#     --no_split_batch 
