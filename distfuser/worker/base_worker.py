import os
from collections import defaultdict
import torch
import torch.distributed as dist
from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
from verl.single_controller.base import Worker as VerlWorker
from distfuser.utils import init_logger

logger = init_logger(__name__)


class BaseTorchDistWorker(VerlWorker):
    """Base Worker for torch distributed environment initialization"""

    def __init__(
        self,
        backend: str = "nccl",
        init_method: str = None,
    ):
        """
        Initialize base torch distributed worker
        Args:
            backend: Distributed backend (nccl, gloo, mpi)
            init_method: URL specifying how to initialize the process group
        """
        super().__init__()
        
        self.backend = backend
        self.init_method = init_method
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")
        self.gpu_id = self.get_cuda_visible_devices()
        
        self.init_torch_dist_env()

    def init_torch_dist_env(self) -> None:
        """Initialize the torch distributed environment"""
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        
        logger.info(f"{self.__class__.__name__}: {self.rank=}, {self.world_size=}, {self.gpu_id=}, {os.environ.get('MASTER_PORT', 'N/A')=}")
        
        # Initialize process group if not already initialized
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                rank=self.rank,
                world_size=self.world_size,
                init_method=self.init_method
            )
            logger.info(f"Initialized torch distributed process group with backend={self.backend}")
        else:
            logger.info("Torch distributed process group already initialized")

    def cleanup_dist_env(self) -> None:
        """Cleanup the distributed environment"""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Destroyed torch distributed process group")

    def forward(self, *args, **kwargs):
        """
        Generate output - to be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.cleanup_dist_env()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


class BasexDiTWorker(VerlWorker):
    """Base Worker using xDiT to do DiT inference"""

    def __init__(
        self,
        ulysses_degree: int = 1,
        ring_degree: int = 1,
        cfg_degree: int = 1,
    ):
        """
        Initialize base worker running xdit inference engine
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

        self.init_xdit_dist_env()

    def init_xdit_dist_env(self) -> None:
        """Initialize the distributed environment using xDiT APIs"""
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

    def cleanup_dist_env(self) -> None:
        """Cleanup the distributed environment"""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Destroyed xdit distributed process group")
        
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.cleanup_dist_env()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")