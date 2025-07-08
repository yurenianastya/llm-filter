import os
import time
from logging import getLogger

import httpx
from fastapi import HTTPException

from src.core.rabbitmq import RabbitMQService
from src.pydantic.response import ModelResponse, ModelResponsePayload, ProcessingResult
from src.utils.metrics import FILTER_DURATION, LLM_RESPONSE_TIME, FILTER_RESULT_COUNTER

logger = getLogger(__name__)

class MessageManager:

    def __init__(self, rabbitmq_service: RabbitMQService):
        self.rabbitmq_service = rabbitmq_service
        self.ollama_url = os.environ.get('OLLAMA_HOST') + '/api/generate'
        self.ollama_model = os.environ.get('OLLAMA_MODEL')
        if not self.ollama_model or not self.ollama_url:
            logger.error("Missing OLLAMA_HOST or OLLAMA_MODEL environment variable")
            raise RuntimeError("OLLAMA_HOST and OLLAMA_MODEL environment variables must be set")
        logger.info("MessageManager initialized with model: %s", self.ollama_model)

    def get_filters_results(self, message: str) -> ModelResponse:
        logger.info("Received message: %s", message)
        try:
            start_filter = time.time()
            pre_filter = self.rabbitmq_service.process_request(message)
            filter_time = time.time() - start_filter
            FILTER_DURATION.observe(filter_time)
            pre_result = ProcessingResult.parse_obj(pre_filter)
            logger.info("Pre-filter result: %s", pre_result)
        except Exception as e:
            logger.exception("Pre-filter processing failed: %s", e)
            raise HTTPException(status_code=500, detail="Pre-filter processing failed") from e

        if not pre_filter.get('status'):
            FILTER_RESULT_COUNTER.labels(status="blocked", type="pre").inc()
            logger.warning("Message blocked by pre-filter")
            return ModelResponse(user_message=message, results=ModelResponsePayload(preprocessing_result=pre_result))

        try:
            start_llm = time.time()
            llm_output = self._send_http_request(message)
            LLM_RESPONSE_TIME.observe(time.time() - start_llm)
            logger.info("LLM output: %s", llm_output)
        except Exception as e:
            logger.exception("LLM request failed with exception: %s", e)
            raise HTTPException(status_code=500, detail="LLM request failed") from e

        try:
            post_filter = self.rabbitmq_service.process_request(llm_output)
            post_result = ProcessingResult.parse_obj(post_filter)
            logger.info("Post-filter result: %s", post_result)
        except Exception as e:
            logger.exception("Post-filter processing failed: %s", e)
            raise HTTPException(status_code=500, detail="Post-filter processing failed") from e

        if not post_filter.get('status'):
            logger.warning("LLM output blocked by post-filter")
            llm_output = ""
            FILTER_RESULT_COUNTER.labels(status="blocked", type="post").inc()
        else:
            FILTER_RESULT_COUNTER.labels(status="passed", type="post").inc()

        return ModelResponse(
            user_message=message,
            results=ModelResponsePayload(
                preprocessing_result=pre_result,
                postprocessing_result=post_result,
                llm_output=llm_output
            )
        )

    def _send_http_request(self, message: str) -> str:
        payload = {'model': self.ollama_model, 'prompt': message, 'stream': False}
        logger.info("Sending request to LLM: %s", payload)
        try:
            with httpx.Client(timeout=500.0) as client:
                response = client.post(self.ollama_url, json=payload)
                logger.info("Received response with status: %s", response.status_code)
                if response.status_code == 200:
                    return response.json().get('response', '')
                logger.error("Non-200 response from model: %s", response.text)
                raise HTTPException(status_code=response.status_code, detail="Error from LLM model")
        except httpx.RequestError as exc:
            logger.exception("Request to LLM model failed")
            raise HTTPException(status_code=500, detail=f'Request to LLM model failed: {exc}') from exc
