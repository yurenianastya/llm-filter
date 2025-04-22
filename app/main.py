from fastapi import FastAPI, Depends
from pydantic import BaseModel
from services.rabbitmq import RabbitMQService
from services.model_processing import LLMProcessingService
import os

app = FastAPI()

def get_llm_service() -> LLMProcessingService:
    rabbitmq_url = f'amqp://guest:guest@{os.getenv("RABBITMQ_HOST")}'
    rabbitmq_service = RabbitMQService(rabbitmq_url=rabbitmq_url)
    return LLMProcessingService(rabbitmq_service)

class PromptRequest(BaseModel):
    message: str

@app.post('/prompt/')
async def process_prompt(
    prompt: PromptRequest,
    llm_service: LLMProcessingService = Depends(get_llm_service)
):
    response = await llm_service.send_message_to_model(prompt.message)
    return response

@app.get("/")
async def root():
    return {"message": "Hello, FastAPI is running!"}