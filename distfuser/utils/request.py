import uuid
from pydantic import BaseModel
from contextvars import ContextVar
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="")


class GenerateRequest(BaseModel):
    """Request model for generating"""

    prompt: str = ""
    negative_prompt: str = ""
    num_inference_steps: int = 50
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 5.0
    seed: int = -1
    return_type: str = "file_path" # 'file_path' of 'bytes'


def get_request_id() -> str:
    return request_id_ctx_var.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_ctx_var.set(request_id)
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
