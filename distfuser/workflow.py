from abc import ABC, abstractmethod
from typing import Dict
from .engine import Engine
import time
from distfuser.utils import init_logger

logger = init_logger(__name__)


class BaseWorkflow(ABC):
    def __init__(self, *args, **kwargs):
        pass

    async def startup(self):
        pass

    async def shutdown(self):
        pass

    def _format_success(self, data: Dict, latency: float = None) -> Dict:
        success_info = {"status": "success", "data": data, "metadata": {}}
        if latency is not None:
            success_info["metadata"]["latency"] = latency
        return success_info

    def _format_error(self, error: str) -> Dict:
        error_info = {"status": "error", "error": error}
        return error_info

    @abstractmethod
    async def generate(self, request: Dict) -> Dict:
        pass


class SimpleT2XWorkflow(BaseWorkflow):
    def __init__(self, encoder: Engine, dit: Engine, vae: Engine):
        super().__init__()
        self.encoder = encoder
        self.dit = dit
        self.vae = vae

    async def startup(self):
        await self.encoder.startup()
        await self.dit.startup()
        await self.vae.startup()

    async def shutdown(self):
        await self.encoder.shutdown()
        await self.dit.shutdown()
        await self.vae.shutdown()

    async def generate(self, request: Dict):
        try:
            start_time = time.time()
            encoder_output = (await self.encoder.add_request(request))[0]
            dit_output = (await self.dit.add_request({**request, **encoder_output}))[0]
            vae_output = (await self.vae.add_request({**request, **dit_output}))[0]
            latency = time.time() - start_time
            return self._format_success(vae_output, latency)
        except Exception as e:
            logger.error(f"{self.__class__.__name__} execution failed: {e}")
            return self._format_error(str(e))
