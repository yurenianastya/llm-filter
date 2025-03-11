from fastapi import FastAPI, HTTPException, Request
from dotenv import load_dotenv, find_dotenv
import aio_pika
import httpx
import asyncio
import json
import uuid
import os

load_dotenv(find_dotenv())

RABBITMQ_HOST = os.getenv('RABBITMQ_HOST')
rabbitmq_url = f'amqp://guest:guest@{RABBITMQ_HOST}'

app = FastAPI()

async def send_message_to_model(message):
    filter_output = await communicate_with_filter({'message': message})
    
    if filter_output['message'] == 'The prompt did not pass the filtering.':
        return { 'response_fail': 'Prompt did not pass preprocessing filter', 'filter_output': filter_output }

    ollama_url = os.environ.get('OLLAMA_HOST') + '/api/generate'
    payload = {
        'model': os.environ.get('OLLAMA_MODEL'),
        'prompt': message,
        'stream': False
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(ollama_url, json=payload)

            if response.status_code == 200:
                print(response.json())
                response_data = response.json().get('response', '')
                return {'response_success': { 'filter_output': filter_output, 'llm_model_output': response_data}}
            else:
                return {'response_fail': f'There was an error while connecting to model. Status code:{response.status_code}'}

        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f'Request failed: {exc}')


async def communicate_with_filter(message):
    connection = await aio_pika.connect_robust(rabbitmq_url)
    async with connection:
        channel = await connection.channel()
        publish_queue = 'preprocessing_request'
        consume_queue = 'preprocessing_result'

        await channel.declare_queue(publish_queue, durable=True)
        await channel.declare_queue(consume_queue, durable=True)

        correlation_id = str(uuid.uuid4())
        future_response = asyncio.Future()

        async def on_filter_output(message: aio_pika.IncomingMessage):
            async with message.process():
                if message.correlation_id == correlation_id:
                    future_response.set_result(json.loads(message.body.decode()))

        queue = await channel.get_queue(consume_queue)
        await queue.consume(on_filter_output)
        message['correlation_id'] = correlation_id

        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(message).encode(),
                correlation_id=correlation_id,
                reply_to=consume_queue,
            ),
            routing_key=publish_queue,
        )
        response = await future_response

    return response

@app.post('/chat/')
async def send_message(request: Request):
    data = await request.json()
    user_message = data.get('message', '')
    
    if not user_message:
        return {'error': 'Message cannot be empty'}
    
    final_response = await send_message_to_model(user_message)

    return {'status': 'After preprocessing', 'message': user_message, 'response': final_response}
