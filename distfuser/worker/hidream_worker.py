import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict
import torch
from .base_worker import BasexDiTWorker, BaseTorchDistWorker
from .hidream_patch import HiDreamTextEncoderPipeline, HiDreamDiTPipeline, HiDreamVaePipeline
from distfuser.utils import init_logger

import ray
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

logger = init_logger(__name__)

try:
    import hi_diffusers
except ImportError:
    logger.warning("hi_diffusers is not installed")
    HAS_HIDREAM_INSTALLED = False
else:
    HAS_HIDREAM_INSTALLED = True

HIDREAM_CONFIGS = {
    "dev": {
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
    },
    "full": {
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
    },
    "fast": {
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
    },
}


@ray.remote
class HiDreamTextEncoderWorker(BaseTorchDistWorker):
    def __init__(
        self,
        model_path: str = "HiDream-ai/HiDream-I1-Fast",
        hidream_model_type: str = "fast",
        llama_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        logger.info(
            f"Initializing HiDreamTextEncoderWorker with " f"model_path={model_path}, rank={self.rank}, gpu_id={self.gpu_id}"
        )

        if not HAS_HIDREAM_INSTALLED:
            raise ImportError("hi_diffusers is not installed. Please install it.")

        assert hidream_model_type in HIDREAM_CONFIGS, f"Invalid hidream model type: {hidream_model_type}"

        self.hidream_config = HIDREAM_CONFIGS[hidream_model_type]

        self.pipe = HiDreamTextEncoderPipeline(
            checkpoint_dir=model_path,
            llama_model_name=llama_model_name,
            dtype=dtype,
            device=self.device,
            rank=self.rank,
        )
        logger.info(f"Rank {self.rank}: HiDreamTextEncoderWorker initialization completed")

    @register(Dispatch.ONE_TO_ALL, blocking=True)
    def forward(self, request: Dict) -> Dict:
        start_time = time.time()
        logger.info("HiDreamTextEncoderWorker forward starting")
        if self.pipe is None:
            raise RuntimeError("HiDreamTextEncoderWorker has not been initialized")

        adapted_params = self.adapt_input(request)
        result = self.pipe.generate(**adapted_params)
        output = self.adapt_output(result)

        elapsed_time = time.time() - start_time
        logger.info(f"HiDreamTextEncoderWorker forward completed in {elapsed_time:.2f} seconds")
        return output

    def adapt_input(self, request: Dict):
        adapted_params = {
            "prompt": request.get("prompt", ""),
            "negative_prompt": request.get("negative_prompt", ""),
            "guidance_scale": request.get("guidance_scale", self.hidream_config["guidance_scale"]),
        }
        return adapted_params

    def adapt_output(self, result) -> Dict:
        prompt_embeds, pooled_prompt_embeds, batch_size = result
        output = {
            "batch_size": batch_size,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }
        return output

    @staticmethod
    def create_worker_group(
        config,
        resource_pool=None,
        max_colocate_count: int = 1,
    ) -> RayWorkerGroup:
        if resource_pool is None:
            resource_pool = RayResourcePool(
                process_on_nodes=[config.get("workers_per_node", 1)] * config.get("nnodes", 1),
                use_gpu=True,
                name_prefix=config.get("name_prefix", ""),
                max_colocate_count=max_colocate_count,
            )
        cls_with_init = RayClassWithInitArgs(
            cls=HiDreamTextEncoderWorker,
            model_path=config.get("model_path", "HiDream-ai/HiDream-I1-Fast"),
            hidream_model_type=config.get("hidream_model_type", "fast"),
            llama_model_name=config.get("llama_model_name", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        )
        if config.get("specific_device", None) is not None:
            cls_with_init.set_additional_resource({config["specific_device"]: 1 / resource_pool.max_collocate_count})
        return RayWorkerGroup(
            resource_pool,
            cls_with_init,
            name_prefix=config.get("name_prefix", f"HiDreamTextEncoderWorker_{uuid.uuid4()}"),
        )


@ray.remote
class HiDreamDiTWorker(BasexDiTWorker):
    def __init__(
        self,
        model_path: str = "HiDream-ai/HiDream-I1-Fast",
        hidream_model_type: str = "fast",
        ulysses_degree: int = 1,
        ring_degree: int = 1,
        cfg_degree: int = 1,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(ulysses_degree, ring_degree, cfg_degree)

        logger.info(f"Initializing HiDreamDiTWorker with " f"model_path={model_path}, rank={self.rank}, gpu_id={self.gpu_id}")

        if not HAS_HIDREAM_INSTALLED:
            raise ImportError("hi_diffusers is not installed. Please install it.")

        assert hidream_model_type in HIDREAM_CONFIGS, f"Invalid hidream model type: {hidream_model_type}"

        self.hidream_config = HIDREAM_CONFIGS[hidream_model_type]

        if self.parallel_config["cfg_parallel_degree"] != 1:
            assert hidream_model_type == "full", f"Only hidream[full] model supports cfg parallel"

        self.pipe = HiDreamDiTPipeline(
            checkpoint_dir=model_path,
            device=self.device,
            rank=self.rank,
            model_type=hidream_model_type,
            shift=self.hidream_config["shift"],
            dtype=dtype,
            use_usp=(ulysses_degree > 1 or ring_degree > 1),
            use_cfg_parallel=self.parallel_config["cfg_parallel_degree"] > 1,
        )
        logger.info(f"Rank {self.rank}: HiDreamDiTWorker initialization completed")

    @register(Dispatch.ONE_TO_ALL, blocking=False)
    def forward(self, request: Dict) -> Dict:
        logger.info("HiDreamDiTWorker forward starting")
        start_time = time.time()
        if self.pipe is None:
            raise RuntimeError("HiDreamDiTWorker has not been initialized")

        adapted_params = self.adapt_input(request)
        result = self.pipe.generate(**adapted_params)
        if self.rank == 0:
            output = self.adapt_output(result)
        else:
            output = None
        elapsed_time = time.time() - start_time
        logger.info(f"HiDreamDiTWorker forward completed in {elapsed_time:.2f} seconds")
        return output

    def adapt_input(self, request: Dict):
        seed = request.get("seed", -1)
        if seed == -1:
            seed = torch.randint(0, 1000000, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        generation_params = {
            "height": request.get("height", 1024),
            "width": request.get("width", 1024),
            "num_inference_steps": request.get("num_inference_steps", self.hidream_config["num_inference_steps"]),
            "guidance_scale": request.get("guidance_scale", self.hidream_config["guidance_scale"]),
            "generator": generator,
        }

        adapted_params = {
            "batch_size": request.get("batch_size", 1),
            "prompt_embeds": request.get("prompt_embeds", []),
            "pooled_prompt_embeds": request.get("pooled_prompt_embeds", []),
            **generation_params,
        }
        return adapted_params

    def adapt_output(self, result) -> Dict:
        return {"latents": result}

    @staticmethod
    def create_worker_group(
        config,
        resource_pool=None,
        max_colocate_count: int = 1,
    ) -> RayWorkerGroup:
        if resource_pool is None:
            resource_pool = RayResourcePool(
                process_on_nodes=[config.get("workers_per_node", 1)] * config.get("nnodes", 1),
                use_gpu=True,
                name_prefix=config.get("name_prefix", ""),
                max_colocate_count=max_colocate_count,
            )
        parallel_config = config.get("parallel_config", {})
        cls_with_init = RayClassWithInitArgs(
            cls=HiDreamDiTWorker,
            model_path=config.get("model_path", "HiDream-ai/HiDream-I1-Fast"),
            hidream_model_type=config.get("hidream_model_type", "fast"),
            ulysses_degree=parallel_config.get("ulysses_degree", 1),
            ring_degree=parallel_config.get("ring_degree", 1),
            cfg_degree=parallel_config.get("cfg_parallel_degree", 1),
        )
        if config.get("specific_device", None):
            cls_with_init.set_additional_resource({config["specific_device"]: 1 / resource_pool.max_collocate_count})
        return RayWorkerGroup(
            resource_pool,
            cls_with_init,
            name_prefix=config.get("name_prefix", f"HiDreamDiTWorker_{uuid.uuid4()}"),
        )


@ray.remote
class HiDreamVaeWorker(BaseTorchDistWorker):
    def __init__(
        self,
        model_path: str = "HiDream-ai/HiDream-I1-Fast",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        logger.info(f"Initializing HiDreamVaeWorker with " f"model_path={model_path}, rank={self.rank}, gpu_id={self.gpu_id}")

        if not HAS_HIDREAM_INSTALLED:
            raise ImportError("hi_diffusers is not installed. Please install it.")

        self.pipe = HiDreamVaePipeline(
            checkpoint_dir=model_path,
            device=self.device,
            rank=self.rank,
            dtype=dtype,
        )
        logger.info(f"Rank {self.rank}: HiDreamVaeWorker initialization completed")

    @register(Dispatch.ONE_TO_ALL, blocking=False)
    def forward(self, request: Dict) -> Dict:
        logger.info("HiDreamVaeWorker forward starting")
        start_time = time.time()
        if self.pipe is None:
            raise RuntimeError("HiDreamVaeWorker has not been initialized")

        adapted_params = self.adapt_input(request)
        image = self.pipe.generate(**adapted_params).images[0]

        if self.rank == 0:
            return_type = request.get("return_type", "file_path")
            if return_type == "bytes":
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
            else:
                image_path = Path(f"results/{request.get('request_id', str(uuid.uuid4()))}.png")
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(image_path)

            output = self.adapt_output(
                image_path=str(image_path) if return_type == "file_path" else None,
                image_bytes=image_bytes if return_type == "bytes" else None,
            )

            elapsed_time = time.time() - start_time
            logger.info(f"HiDreamVaeWorker forward completed in {elapsed_time:.2f} seconds")
            return output

        return None

    def adapt_input(self, request: Dict):
        adapted_params = {
            "latents": request.get("latents", []),
            "output_type": request.get("output_type", "pil"),
            "return_dict": request.get("return_dict", True),
        }
        return adapted_params

    def adapt_output(self, image_path: str, image_bytes: bytes) -> Dict:
        return {"image_path": image_path, "image_bytes": image_bytes}

    @staticmethod
    def create_worker_group(
        config,
        resource_pool=None,
        max_colocate_count: int = 1,
    ) -> RayWorkerGroup:
        if resource_pool is None:
            resource_pool = RayResourcePool(
                process_on_nodes=[config.get("workers_per_node", 1)] * config.get("nnodes", 1),
                use_gpu=True,
                name_prefix=config.get("name_prefix", ""),
                max_colocate_count=max_colocate_count,
            )
        cls_with_init = RayClassWithInitArgs(
            cls=HiDreamVaeWorker,
            model_path=config.get("model_path", "HiDream-ai/HiDream-I1-Fast"),
        )
        if config.get("specific_device", None) is not None:
            cls_with_init.set_additional_resource({config["specific_device"]: 1 / resource_pool.max_collocate_count})
        return RayWorkerGroup(
            resource_pool,
            cls_with_init,
            name_prefix=config.get("name_prefix", f"HiDreamVaeWorker_{uuid.uuid4()}"),
        )
