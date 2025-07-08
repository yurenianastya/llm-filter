import logging

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.router import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(router)

Instrumentator().instrument(app).expose(app)
