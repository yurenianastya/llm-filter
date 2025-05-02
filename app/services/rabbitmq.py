import os
import uuid
import json
from threading import Event

from pika import BasicProperties, BlockingConnection, ConnectionParameters, PlainCredentials
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

EXCHANGE_NAME = os.environ.get("EXCHANGE_NAME")
QUEUE_TASK = os.environ.get("QUEUE_TASK")
QUEUE_RESULT = os.environ.get("QUEUE_RESULT")

class RabbitMQService:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.correlation_id = None
        self.response = None
        self.response_event = Event()
        self.initialize()

    def initialize(self):
        params = ConnectionParameters(
            host='rabbitmq',
            port=5672,
            virtual_host='/',
            credentials=PlainCredentials('guest', 'guest'),
            blocked_connection_timeout=300
        )
        self.connection = BlockingConnection(params)
        self.channel = self.connection.channel()

        self.channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type='direct')
        self.channel.queue_declare(queue=QUEUE_TASK, durable=True)
        self.channel.queue_declare(queue=QUEUE_RESULT, durable=True)

        self.channel.queue_bind(queue=QUEUE_TASK, exchange=EXCHANGE_NAME, routing_key=QUEUE_TASK)
        self.channel.queue_bind(queue=QUEUE_RESULT, exchange=EXCHANGE_NAME, routing_key=QUEUE_RESULT)
        self.channel.basic_qos(prefetch_count=1)

    def process_request(self, message: str):
        self.response = None
        self.response_event.clear()

        correlation_id = str(uuid.uuid4())
        self.correlation_id = correlation_id

        self.channel.basic_publish(
            exchange=EXCHANGE_NAME,
            routing_key=QUEUE_TASK,
            body=json.dumps({"message": message}),
            properties=BasicProperties(
                reply_to=QUEUE_RESULT,
                correlation_id=correlation_id
            )
        )       
        def on_response(ch, method, props, body):
            if props.correlation_id == self.correlation_id:
                self.response = json.loads(body)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                self.response_event.set()
                ch.stop_consuming()

        self.channel.basic_consume(
            queue=QUEUE_RESULT,
            on_message_callback=on_response,
            auto_ack=False
        )

        self.channel.start_consuming()
        return self.response
   
    def close(self):
        if self.connection and not self.connection.is_closed:
            self.connection.close()
