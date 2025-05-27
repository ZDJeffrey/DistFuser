import os
from collections import defaultdict
import torch
import torch.distributed as dist
from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
from verl.single_controller.base import Worker
from distfuser.utils import init_logger

logger = init_logger(__name__)


class BaseWorker(Worker):
    """Base Worker"""

    def __init__(
        self,
        ulysses_degree: int = 1,
        ring_degree: int = 1,
        cfg_degree: int = 1,
    ):
        """
        Initialize base worker
        Args:
            ulysses_degree: SP-Ulysses parallel degree
            ring_degree: SP-Ring parallel degree
            cfg_degree: CFG parallel degree
        """
        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")
        self.gpu_id = self.get_cuda_visible_devices()
        # Initialize parallel config
        assert (
            ulysses_degree * ring_degree * cfg_degree
        ) == self.world_size, "Sequence parallel degree and CFG degree must multiply to world size"
        self.parallel_config = defaultdict(
            lambda: 1,
            {
                "sequence_parallel_degree": ulysses_degree * ring_degree,
                "ulysses_degree": ulysses_degree,
                "ring_degree": ring_degree,
                "classifier_free_guidance_degree": cfg_degree,
            },
        )

        self.init_dist_env()

    def init_dist_env(self) -> None:
        """Initialize the distributed environment"""
        torch.cuda.set_device(self.device)
        logger.info(f"{self.__class__.__name__}: {self.rank=}, {self.world_size=}, {self.gpu_id=}, {os.environ['MASTER_PORT']=}")
        dist.init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)

        init_distributed_environment(backend="nccl", rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=self.parallel_config["sequence_parallel_degree"],
            ulysses_degree=self.parallel_config["ulysses_degree"],
            ring_degree=self.parallel_config["ring_degree"],
            classifier_free_guidance_degree=self.parallel_config["classifier_free_guidance_degree"],
            pipeline_parallel_degree=self.parallel_config["pipefusion_degree"],
            tensor_parallel_degree=self.parallel_config["tensor_parallel_degree"],
        )

    def forward(self, *args, **kwargs) -> str:
        """
        Generate output
        """
        raise NotImplementedError("Subclasses must implement forward method")
