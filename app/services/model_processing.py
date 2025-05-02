import os
from typing import Dict, Any

import httpx
from fastapi import HTTPException

from .rabbitmq import RabbitMQService


class LLMProcessingService:

    def __init__(self, rabbitmq_service: RabbitMQService):
        self.rabbitmq_service = rabbitmq_service
        self.ollama_url = os.environ.get('OLLAMA_HOST') + '/api/generate'
        self.ollama_model = os.environ.get('OLLAMA_MODEL')

    def send_message_to_model(self, message: str) -> Dict[str, Any]:
        filter_output = self.rabbitmq_service.process_request(message)

        if filter_output.get('status') == 'Filtered out':
            return {
                'response_fail': 'Prompt did not pass preprocessing filter',
                'filter_output': filter_output
            }

        payload = {
            'model': self.ollama_model,
            'prompt': message,
            'stream': False
        }

        return self._send_http_request(payload, filter_output)

    def _send_http_request(self, payload: Dict[str, Any], filter_output: Dict[str, Any]) -> Dict[str, Any]:
        try:
            with httpx.Client(timeout=500.0) as client:
                response = client.post(self.ollama_url, json=payload)
                if response.status_code == 200:
                    response_data = response.json().get('response', '')
                    return {
                        'response_success': {
                            'filter_output': filter_output,
                            'model_output': response_data
                        }
                    }
                return {
                    'response_fail': f'There was an error while connecting to the model. Status code: {response.status_code}'
                }
        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f'Request to LLM model failed: {exc}') from exc

