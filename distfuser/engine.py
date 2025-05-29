import asyncio
from typing import Dict
from verl.single_controller.ray import RayWorkerGroup

from distfuser.utils import init_logger

logger = init_logger(__name__)


class Engine:
    def __init__(self, worker_group: RayWorkerGroup):
        self.worker_group = worker_group
        self.input_queue = asyncio.Queue()
        self.futures = {}
        self.loop_task = None

    async def startup(self):
        self.loop_task = asyncio.create_task(self.event_loop())

    async def shutdown(self):
        self.loop_task.cancel()
        await self.loop_task

    async def add_request(self, request: Dict):
        """
        Add a request to the input queue and wait for the result.
        """
        request_id = request["request_id"]
        fut = asyncio.get_event_loop().create_future()
        self.futures[request_id] = fut
        await self.input_queue.put(request)

        # wait until event_loop send the request to worker group
        ray_objs = await fut
        return await asyncio.gather(*ray_objs)

    async def event_loop(self):
        """
        The event loop that processes requests from the input queue.
        It sends requests to the worker group and sets the result in the future.
        """
        try:
            while True:
                request = await self.input_queue.get()

                request_id = request["request_id"]
                ray_objs = self.worker_group.execute_all_async("forward", request=request)
                fut = self.futures.pop(request_id, None)
                if fut and not fut.done():
                    fut.set_result(ray_objs)
        except asyncio.CancelledError:
            logger.info(f"Event loop for {self.worker_group.name_prefix} cancelled")

    @classmethod
    def from_config(cls, worker_cls, worker_config, **kwargs):
        """
        Create an Engine instance from the worker class and configuration.
        """
        worker_group = worker_cls.create_worker_group(worker_config, **kwargs)
        return cls(worker_group)
