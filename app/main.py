import asyncio

from fastapi import FastAPI, Depends

from services.rabbitmq import RabbitMQService
from services.llm_handler import MessageManager, UserInput

app = FastAPI()

def get_message_manager() -> MessageManager:
    rabbitmq_service = RabbitMQService()
    return MessageManager(rabbitmq_service)

@app.post('/prompt')
async def process_prompt(
    user_input: UserInput,
    msg_service: MessageManager = Depends(get_message_manager)
):
    prompt = user_input.message
    try:
        filtering_response = await asyncio.to_thread(msg_service.get_filters_results, prompt)
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
    return {
        'response': filtering_response.model_dump(exclude_unset=True)
    }

@app.get("/")
def root():
    return {"message": "FastAPI is running"}
