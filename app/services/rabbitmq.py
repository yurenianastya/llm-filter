import uuid
import json
import asyncio
from typing import Any, Dict
from aio_pika import connect_robust, IncomingMessage, Message, RobustConnection, RobustChannel
from fastapi import HTTPException

PUBLISH_QUEUE_NAME = "preproc_request"
CONSUME_QUEUE_NAME = "preproc_result"


class RabbitMQService:
    def __init__(self, rabbitmq_url: str):
        self.rabbitmq_url: str = rabbitmq_url
        self.connection: RobustConnection | None = None
        self.channel: RobustChannel | None = None

    async def connect(self) -> None:
        """Establish a connection and declare queues."""
        if self.connection is None or self.connection.is_closed:
            try:
                self.connection = await connect_robust(self.rabbitmq_url)
                self.channel = await self.connection.channel()
                await self._declare_queues()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to connect to RabbitMQ: {str(e)}")

    async def _declare_queues(self) -> None:
        """Declare the publish and consume queues."""
        if self.channel:
            await self.channel.declare_queue(PUBLISH_QUEUE_NAME, durable=True)
            await self.channel.declare_queue(CONSUME_QUEUE_NAME, durable=True)

    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message to the publish queue and wait for a response from the consume queue.

        Args:
            message (Dict[str, Any]): The message to send.

        Returns:
            Dict[str, Any]: The response message.

        Raises:
            HTTPException: If the response times out or an error occurs.
        """
        await self.connect()

        correlation_id = str(uuid.uuid4())
        future_response = asyncio.Future()

        async def on_filter_output(incoming_message: IncomingMessage) -> None:
            """Callback to process messages from the consume queue."""
            async with incoming_message.process():
                if incoming_message.correlation_id == correlation_id:
                    future_response.set_result(json.loads(incoming_message.body.decode()))

        await self._consume_queue(on_filter_output)

        message['correlation_id'] = correlation_id

        await self._publish_message(message, correlation_id)

        response = await asyncio.wait_for(future_response, timeout=300)
        return response

    async def _consume_queue(self, callback) -> None:
        """Set up a consumer for the consume queue."""
        if self.channel:
            queue = await self.channel.get_queue(CONSUME_QUEUE_NAME)
            await queue.consume(callback)

    async def _publish_message(self, message: Dict[str, Any], correlation_id: str) -> None:
        """Publish a message to the publish queue."""
        if self.channel:
            try:
                await self.channel.default_exchange.publish(
                    Message(
                        body=json.dumps(message).encode(),
                        correlation_id=correlation_id,
                        reply_to=CONSUME_QUEUE_NAME,
                    ),
                    routing_key=PUBLISH_QUEUE_NAME,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to publish message: {str(e)}")

    async def close(self) -> None:
        """Close the RabbitMQ connection."""
        if self.connection and not self.connection.is_closed:
            await self.connection.close()