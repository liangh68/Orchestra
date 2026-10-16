import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    

import argparse
import os
import time

import torch
from torch import distributed as dist

from diffusers import DDIMScheduler
from tqdm import trange

from orchestra.pipelines import OrcheSDXLPipeline
from orchestra.config import OrcheConfig
from orchestra.coordinator import OrcheCoordinator
from orchestra.schedulers.orche_scheduling_ddim import OrcheDDIMScheduler


def modify_output_path(args):
    base_name, ext = os.path.splitext(args.output_path)
    if args.image_size == [1024, 1024]:
        if isinstance(args.image_size, int):
            args.image_size = [args.image_size, args.image_size]
            size_name = f"_HW{args.image_size}-{args.image_size}"
        else:
            if len(args.image_size) == 1:
                args.image_size = [args.image_size[0], args.image_size[0]]
                size_name = f"_HW{args.image_size[0]}-{args.image_size[0]}"
            else:
                assert len(args.image_size) == 2
                size_name = f"_HW{args.image_size[0]}-{args.image_size[1]}"
    else:
        size_name = ''
    step_name = f"_T{args.num_inference_steps}"
    GPU_num = f"_GPU{dist.get_world_size()}"
    new_output_path = f"{base_name}{GPU_num}{step_name}{size_name}{ext}"
    return new_output_path

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, 
                        default="generated_images/example.png")
    parser.add_argument("--num_inference_steps", type=int, default=100, help="Number of inference steps")
    parser.add_argument("--image_size", type=int, nargs="*", default=1024, help="Image size of generation") 
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--warmup_steps", type=int, default=4, help="Number of warmup steps")
    parser.add_argument(
        "--sync_mode",
        type=str,
        default="corrected_async_gn",
        choices=["separate_gn", "stale_gn", "corrected_async_gn", "sync_gn", "full_sync", "no_sync"],
        help="Different GroupNorm synchronization modes",
    )
    parser.add_argument(
        "--parallelism", 
        type=str, 
        choices=["patch", "tensor", "orche"],
        default="orche")
    parser.add_argument(
        "--no_split_batch", action="store_true", help="Disable the batch splitting for classifier-free guidance"
    )
    parser.add_argument("--no_cuda_graph", action="store_true", help="Disable CUDA graph")

    # Coordinator specific arguments
    parser.add_argument("--no_temporal_adaptation", action="store_true", help="No temporal_adaptation")
    parser.add_argument("--no_spatial_adaptation", action="store_true", help="No spatial_adaptation")
    parser.add_argument("--use_deeper_reduce", action="store_true", help="Use deeper reduction for slow GPUs (reduce steps by 4x instead of 2x)")
    parser.add_argument("--gpu_speed", type=str, default="0", help="set gpu_speed manually")
    parser.add_argument("--gpu_utils", type=str, default="0", help="set gpu_utils manually")
    parser.add_argument("--reduce_list", type=str, default="0", help="set reduce_list manually (Only for debug)")
    parser.add_argument("--patches", type=str, default="0", help="set patch_size manually (Only for debug)")
    # Comm_manager specific arguments 
    parser.add_argument("--comm_checkpoint", type=int, default=40, help="comm_checkpoint")

    
    args = parser.parse_args()

    def parse_list(value):
        if value == "0":
            return None
        return [int(x.strip()) for x in value.split(",")]
    
    args.gpu_speed = parse_list(args.gpu_speed)
    args.gpu_utils = parse_list(args.gpu_utils)

    args.reduce_list = parse_list(args.reduce_list)
    args.ori_patches = args.patches
    args.patches = parse_list(args.patches)
    return args


def main():
    args = get_args()
    if isinstance(args.image_size, int):
        args.image_size = [args.image_size, args.image_size]
    else:
        if len(args.image_size) == 1:
            args.image_size = [args.image_size[0], args.image_size[0]]
        else:
            assert len(args.image_size) == 2
    
    if args.parallelism == "orche" and (not args.no_split_batch or not args.no_cuda_graph):
        print("For our method, split_batch and cuda_graph is invalid. Fallback to no_split_batch and no_cuda_graph")
        args.no_split_batch = True
        args.no_cuda_graph = True

    orche_config = OrcheConfig(
        height=args.image_size[0],
        width=args.image_size[1],
        do_classifier_free_guidance=args.guidance_scale > 1, 
        split_batch=not args.no_split_batch,
        warmup_steps=args.warmup_steps,
        comm_checkpoint=args.comm_checkpoint,
        mode=args.sync_mode,
        parallelism=args.parallelism,
        num_inference_steps=args.num_inference_steps,
        use_temporal_adaptation=not args.no_temporal_adaptation,
        use_spatial_adaptation=not args.no_spatial_adaptation,
        use_cuda_graph=not args.no_cuda_graph,
        use_deeper_reduce=args.use_deeper_reduce
    )

    if args.gpu_utils:
        utils = args.gpu_utils
    else:
        utils = [0] * orche_config.world_size

    pretrained_model_name_or_path = args.model_path

    coordinator = None
    if orche_config.parallelism == "orche":
        scheduler = OrcheDDIMScheduler.from_pretrained(pretrained_model_name_or_path, 
                                                    orche_config=orche_config,
                                                    subfolder="scheduler")
        coordinator = OrcheCoordinator(orche_config)
        coordinator.set_preinference_gpu_utilization(utils)
        coordinator.temporal_adaptation(args.reduce_list)
        coordinator.spatial_adaptation(args.patches)
        coordinator.print_info()
        scheduler.set_coordinator(coordinator)
    else:
        scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, 
                                                  subfolder="scheduler")

    pipeline = OrcheSDXLPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        orche_config=orche_config,
        coordinator=coordinator,
        variant="fp16",
        use_safetensors=True,
        scheduler=scheduler,
    )

    assert args.output_path is not None
    pipeline.set_progress_bar_config(disable=orche_config.rank != 0)
    start = time.time()

    image = pipeline(
        prompt=args.prompt,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        num_inference_steps=orche_config.num_inference_steps,
        guidance_scale=args.guidance_scale
    ).images[0]
    latency = time.time()-start
    print(f"Rank {dist.get_rank()} Time taken: {latency:.2f} seconds.")
    args.output_path = modify_output_path(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    image.save(args.output_path)


if __name__ == "__main__":
    main()
    if dist.is_initialized():
        dist.destroy_process_group()