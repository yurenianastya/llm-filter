import asyncio
from logging import getLogger

from fastapi import APIRouter, Depends, HTTPException

from src.core.rabbitmq import RabbitMQService
from src.core.manager import MessageManager
from src.pydantic.response import UserInput, ModelResponse

router = APIRouter()
logger = getLogger(__name__)

def provide_message_manager() -> MessageManager:
    return MessageManager(RabbitMQService())

@router.post("/prompt")
async def process_prompt(
    user_input: UserInput,
    msg_service: MessageManager = Depends(provide_message_manager)
) -> dict:
    logger.info("POST /prompt - Received input: %s", user_input.message)
    try:
        result: ModelResponse = await asyncio.to_thread(
            msg_service.get_filters_results, user_input.message
        )
        logger.info("POST /prompt - Successfully processed input")
        return {"response": result.model_dump(exclude_unset=True)}
    except Exception as exc:
        logger.exception("POST /prompt - Processing failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {type(exc).__name__}: {exc}"
        ) from exc

@router.get("/")
def root() -> dict:
    logger.info("GET / - Health check OK")
    return {"message": "FastAPI is running"}
