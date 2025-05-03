import os
from typing import Dict, Any

import httpx
from pydantic import BaseModel
from fastapi import HTTPException

from .rabbitmq import RabbitMQService


class UserInput(BaseModel):
    message: str

class FilterOutput(BaseModel):
    label: str = ""
    score: float = 0.0

class ProcessingResult(BaseModel):
    status: bool = False
    filter_output: FilterOutput = FilterOutput()

class ModelResponsePayload(BaseModel):
    preprocessing_result: ProcessingResult = ProcessingResult()
    postprocessing_result: ProcessingResult = ProcessingResult()
    llm_output: str = ""

class ModelResponse(BaseModel):
    user_message: str = ""
    results: ModelResponsePayload = ModelResponsePayload()

class MessageManager:

    def __init__(self, rabbitmq_service: RabbitMQService):
        self.rabbitmq_service = rabbitmq_service
        self.ollama_url = os.environ.get('OLLAMA_HOST') + '/api/generate'
        self.ollama_model = os.environ.get('OLLAMA_MODEL')
        self.output = None

    def get_filters_results(self, message: str) -> Dict[str, Any]:
        self.output = ModelResponse(
            user_message=message,
            results=ModelResponsePayload()
        )
        filter_output = self.rabbitmq_service.process_request(message)
        self.output.results.preprocessing_result = ProcessingResult.parse_obj(filter_output)

        if filter_output.get('status') is False:
            return self.output

        model_output = self._send_http_request(message)
        filter_output = self.rabbitmq_service.process_request(model_output)
        self.output.results.postprocessing_result = ProcessingResult.parse_obj(filter_output)
        
        if filter_output.get('status') is False:
            return self.output

        self.output.results.llm_output = model_output
        return self.output

    def _send_http_request(self, message: str) -> str:
        payload = {
            'model': self.ollama_model,
            'prompt': message,
            'stream': False
        }
        try:
            with httpx.Client(timeout=500.0) as client:
                response = client.post(self.ollama_url, json=payload)
                if response.status_code == 200:
                    response_data = response.json().get('response', '')
                    return response_data
                return f'There was an error while connecting to the model: Status code: {response.status_code}'
        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f'Request to LLM model failed: {exc}') from exc
