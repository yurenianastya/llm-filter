import asyncio
import os

from fastapi import FastAPI, Depends, Request

from services.rabbitmq import RabbitMQService
from services.model_processing import LLMProcessingService

app = FastAPI()

def get_llm_service() -> LLMProcessingService:
    rabbitmq_service = RabbitMQService()
    return LLMProcessingService(rabbitmq_service)

@app.post('/prompt')
async def process_prompt(
    request: Request,
    llm_service: LLMProcessingService = Depends(get_llm_service)
):
    data = await request.json()
    prompt = data.get("message")
    
    if not prompt:
        return {"error": "Missing 'prompt' in request body"}

    try:
        response = await asyncio.to_thread(llm_service.send_message_to_model, prompt)
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
    return {'status': 'After preprocessing', 'message': prompt, 'response': response}

@app.get("/")
def root():
    return {"message": "FastAPI is running"}