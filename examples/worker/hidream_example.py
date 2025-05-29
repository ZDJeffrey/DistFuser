import uuid
import time
import warnings
from pathlib import Path
from argparse import ArgumentParser

from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

from distfuser import HiDreamTextEncoderWorker, HiDreamDiTWorker, HiDreamVaeWorker

warnings.filterwarnings("ignore")


def pipeline_forward(request, encoder_group, dit_group, vae_group):
    """
    Execute the forward pass through the HiDream pipeline.
    """
    encoder_output = encoder_group.execute_all_sync("forward", request=request)[0]
    dit_output = dit_group.execute_all_sync("forward", request={**request, **encoder_output})[0]
    vae_output = vae_group.execute_all_sync("forward", request={**request, **dit_output})[0]
    return vae_output


def main(args):
    # Initialize the resource pool
    dit_pool = RayResourcePool([args.dit_num_gpus], use_gpu=True, name_prefix="dit", max_colocate_count=1)
    if args.merge_textencoder_vae:
        encoder_pool = vae_pool = RayResourcePool([1], use_gpu=True, name_prefix="textencoder_vae", max_colocate_count=2)
    else:
        encoder_pool = RayResourcePool([1], use_gpu=True, name_prefix="textencoder", max_colocate_count=1)
        vae_pool = RayResourcePool([1], use_gpu=True, name_prefix="vae", max_colocate_count=1)

    # Initialize the worker groups
    dit_class_with_args = RayClassWithInitArgs(
        cls=HiDreamDiTWorker,
        model_path=args.model,
        hidream_model_type=args.model_type,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_degree=2 if args.use_cfg_parallel else 1,
    )
    dit_group = RayWorkerGroup(dit_pool, dit_class_with_args)
    encoder_class_with_args = RayClassWithInitArgs(
        cls=HiDreamTextEncoderWorker,
        model_path=args.model,
        llama_model_name=args.llama_model,
        hidream_model_type=args.model_type,
    )
    encoder_group = RayWorkerGroup(encoder_pool, encoder_class_with_args)
    vae_class_with_args = RayClassWithInitArgs(
        cls=HiDreamVaeWorker,
        model_path=args.model,
    )
    vae_group = RayWorkerGroup(vae_pool, vae_class_with_args)

    # Warmup
    if args.warmup_steps > 0:
        print("Warming up the workers...")
        request = {
            "prompt": "",
            "num_inference_steps": args.warmup_steps,
            "height": args.height,
            "width": args.width,
            "guidance_scale": args.guidance_scale,
            "output_type": args.output_type,
        }
        output = pipeline_forward(request, encoder_group, dit_group, vae_group)
        print("Warmup completed.")

    # Run inference
    request = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "output_type": args.output_type,
    }
    start_time = time.perf_counter()
    output = pipeline_forward(request, encoder_group, dit_group, vae_group)
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds")

    if args.output_type == "pil":
        output_dir = Path("results/hidream")
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"{args.model_type}_{args.height}x{args.width}_{uuid.uuid4()}.png"
        output["image"].save(image_path)
        print(f"Image saved to {image_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="HiDream-ai/HiDream-I1-Fast", help="Model name or path")
    parser.add_argument(
        "--model_type", type=str, default="full", choices=["fast", "dev", "full"], help="Model type: fast/dev/full"
    )
    parser.add_argument(
        "--llama_model", type=str, default="Meta-Llama/Meta-Llama-3.1-8B-Instruct", help="Llama model name or path"
    )
    parser.add_argument("--prompt", type=str, default="", help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative text prompt for image generation")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_type", type=str, default="pil", choices=["pil", "latent"], help="Output type for image generation"
    )
    parser.add_argument("--dit_num_gpus", type=int, default=1, help="Number of GPUs for DiT worker")
    parser.add_argument("--ulysses_degree", type=int, default=1, help="Degree of the SP-Ulysses")
    parser.add_argument("--ring_degree", type=int, default=1, help="Degree of the SP-Ring")
    parser.add_argument("--use_cfg_parallel", action="store_true", help="Use CFG parallelism")
    parser.add_argument("--merge_textencoder_vae", action="store_true", help="Merge resource pools for encoder and vae")

    args = parser.parse_args()

    main(args)
