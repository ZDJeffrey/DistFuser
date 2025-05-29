import json
from io import BytesIO
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from distfuser.workflow import BaseWorkflow
from distfuser.utils import init_logger
from distfuser.utils.request import RequestIDMiddleware, GenerateRequest, get_request_id

logger = init_logger(__name__)


def run_api_server(workflow: BaseWorkflow, host: str = "localhost", port: int = 8000):
    """
    Run the FastAPI application with the given workflow.

    Args:
        workflow (BaseWorkflow): The workflow to be used by the FastAPI application.
        host (str): The host to run the FastAPI application on.
        port (int): The port to run the FastAPI application on.
    """
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting up the FastAPI application.")
        await workflow.startup()

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down the FastAPI application.")
        await workflow.shutdown()

    @app.post("/api/v1/generate")
    async def generate(request: GenerateRequest):
        logger.info("Received a request to generate.")
        try:
            request_data = request.model_dump()
            request_id = get_request_id()
            request_data["request_id"] = request_id

            result = await workflow.generate(request_data)

            # Check if the result is an error
            if result.get("status") == "error":
                logger.error(f"Request {request_id} failed with error: {result.get('error')}")
                return JSONResponse(
                    status_code=501,
                    content={
                        "exception": result.get("error"),
                        "request_id": request_id,
                    },
                )

            # Return StreamingResponse when requesting for bytes
            if request.return_type == "bytes":
                image_bytes = result.get("data", {}).get("image_bytes")
                result.pop("data", None)
                if image_bytes:
                    return StreamingResponse(
                        BytesIO(image_bytes),
                        media_type="image/png",
                        headers={"X-Result-Metadata": json.dumps(result)},
                    )

            result.pop("data", None)
            return JSONResponse(
                content=result,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
        except Exception as e:
            logger.error(f"Request {request_id} failed: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "exception": "internal_error",
                    "request_id": request_id,
                },
            )

    uvicorn.run(app, host=host, port=port)
