import os

import httpx
from pydantic import BaseModel, Field
from fastapi import HTTPException

from .rabbitmq import RabbitMQService

class UserInput(BaseModel):
    message: str

class FilterOutput(BaseModel):
    label: str = ""
    score: float = 0.0

class SemanticOutput(BaseModel):
    score: float = 0.0

class ProcessingResult(BaseModel):
    status: bool = False
    classification_result: FilterOutput = Field(default_factory=FilterOutput)
    semantic_result: SemanticOutput = Field(default_factory=SemanticOutput)

class ModelResponsePayload(BaseModel):
    preprocessing_result: ProcessingResult = Field(default_factory=ProcessingResult)
    postprocessing_result: ProcessingResult = Field(default_factory=ProcessingResult)
    llm_output: str = ""

class ModelResponse(BaseModel):
    user_message: str = ""
    results: ModelResponsePayload = Field(default_factory=ModelResponsePayload)

class MessageManager:

    def __init__(self, rabbitmq_service: RabbitMQService):
        self.rabbitmq_service = rabbitmq_service
        self.ollama_url = os.environ.get('OLLAMA_HOST') + '/api/generate'
        self.ollama_model = os.environ.get('OLLAMA_MODEL')
        if not self.ollama_model or not self.ollama_url:
            raise RuntimeError("OLLAMA_HOST and OLLAMA_MODEL environment variables must be set")

    def get_filters_results(self, message: str) -> ModelResponse:
        pre_filter = self.rabbitmq_service.process_request(message)
        pre_result = ProcessingResult.parse_obj(pre_filter)

        if not pre_filter.get('status'):
            return ModelResponse(user_message=message, results=ModelResponsePayload(preprocessing_result=pre_result))

        llm_output = self._send_http_request(message)
        post_filter = self.rabbitmq_service.process_request(llm_output)
        post_result = ProcessingResult.parse_obj(post_filter)

        return ModelResponse(
            user_message=message,
            results=ModelResponsePayload(
                preprocessing_result=pre_result,
                postprocessing_result=post_result,
                llm_output=llm_output if post_filter.get('status') else ""
            )
        )

    def _send_http_request(self, message: str) -> str:
        payload = { 'model': self.ollama_model, 'prompt': message, 'stream': False }
        try:
            with httpx.Client(timeout=500.0) as client:
                response = client.post(self.ollama_url, json=payload)
                if response.status_code == 200:
                    return response.json().get('response', '')
                return f"Error from model: Status code {response.status_code}"
        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f'Request to LLM model failed: {exc}') from exc
