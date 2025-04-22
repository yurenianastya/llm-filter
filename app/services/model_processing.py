import os
import httpx
from fastapi import HTTPException
from services.rabbitmq import RabbitMQService
from typing import Dict, Any


class LLMProcessingService:
    def __init__(self, rabbitmq_service: RabbitMQService):
        self.rabbitmq_service = rabbitmq_service
        self.ollama_url = self._get_env_variable('OLLAMA_HOST') + '/api/generate'
        self.ollama_model = self._get_env_variable('OLLAMA_MODEL')

    async def communicate_with_filter(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Communicate with the RabbitMQ filter service.

        Args:
            message (Dict[str, Any]): The message to send to the filter.

        Returns:
            Dict[str, Any]: The filter's response.
        """
        return await self.rabbitmq_service.send_message(message)

    async def send_message_to_model(self, message: str) -> Dict[str, Any]:
        """
        Send a message to the LLM model after passing through the filter.

        Args:
            message (str): The input message to process.

        Returns:
            Dict[str, Any]: The response from the LLM model or an error message.
        """
        filter_output = await self.communicate_with_filter({'message': message})

        if filter_output.get('message') == 'The prompt did not pass the filtering.':
            return {
                'response_fail': 'Prompt did not pass preprocessing filter',
                'filter_output': filter_output
            }

        payload = self._build_payload(message)

        return await self._send_request_to_model(payload, filter_output)

    def _get_env_variable(self, key: str) -> str:
        """
        Retrieve an environment variable or raise an exception if not found.

        Args:
            key (str): The environment variable key.

        Returns:
            str: The value of the environment variable.

        Raises:
            HTTPException: If the environment variable is not set.
        """
        value = os.environ.get(key)
        if not value:
            raise HTTPException(status_code=500, detail=f"Environment variable '{key}' is not set.")
        return value

    def _build_payload(self, message: str) -> Dict[str, Any]:
        """
        Build the payload for the LLM model request.

        Args:
            message (str): The input message.

        Returns:
            Dict[str, Any]: The payload for the request.
        """
        return {
            'model': self.ollama_model,
            'prompt': message,
            'stream': False
        }

    async def _send_request_to_model(self, payload: Dict[str, Any], filter_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a request to the LLM model and handle the response.

        Args:
            payload (Dict[str, Any]): The request payload.
            filter_output (Dict[str, Any]): The filter's output.

        Returns:
            Dict[str, Any]: The response from the LLM model or an error message.
        """
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.post(self.ollama_url, json=payload)

                if response.status_code == 200:
                    response_data = response.json().get('response', '')
                    return {
                        'response_success': {
                            'filter_output': filter_output,
                            'llm_model_output': response_data
                        }
                    }
                else:
                    return {
                        'response_fail': f'There was an error while connecting to the model. Status code: {response.status_code}'
                    }

            except httpx.RequestError as exc:
                raise HTTPException(status_code=500, detail=f'Request to LLM model failed: {exc}')
