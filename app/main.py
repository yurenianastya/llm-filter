import asyncio

from fastapi import FastAPI, Depends, HTTPException
from services.rabbitmq import RabbitMQService
from services.llm_handler import MessageManager, UserInput, ModelResponse

app = FastAPI()

def provide_message_manager() -> MessageManager:
    return MessageManager(RabbitMQService())

@app.post("/prompt")
async def process_prompt(
    user_input: UserInput,
    msg_service: MessageManager = Depends(provide_message_manager)
) -> dict:
    try:
        result: ModelResponse = await asyncio.to_thread(msg_service.get_filters_results, user_input.message)
        return {"response": result.model_dump(exclude_unset=True)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {type(exc).__name__}: {exc}")

@app.get("/")
def root() -> dict:
    return {"message": "FastAPI is running"}
