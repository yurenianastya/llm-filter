from fastapi import FastAPI, Request
import requests
import json
import pika  # For RabbitMQ

app = FastAPI()

RABBITMQ_HOST = "rabbitmq"
QUEUE_NAME = "llm_messages"

def log_message_to_rabbitmq(message):
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME)
    channel.basic_publish(exchange='', routing_key=QUEUE_NAME, body=json.dumps(message))
    connection.close()

@app.post("/chat/")
async def chat_with_model(request: Request):
    data = await request.json()
    user_message = data.get("message", "No input provided")

    log_message_to_rabbitmq({"role": "user", "content": user_message})

    ollama_url = "http://ollama:11434/api/generate"
    request_payload = {
        "model": "llama3.1:8b",
        "prompt": user_message,
        "stream": False
    }

    try:
        response = requests.post(ollama_url, json=request_payload)
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text}")

        if response.status_code != 200:
            return {"error": "Error from Ollama service", "details": response.text}

        model_response = response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        return {"error": "Ollama service not reachable", "details": str(e)}

    log_message_to_rabbitmq({"role": "assistant", "content": model_response})

    return {"response": model_response}