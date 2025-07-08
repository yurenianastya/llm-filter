import uuid
import json
import logging
from threading import Event

from pika import BasicProperties, BlockingConnection, ConnectionParameters
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)

class RabbitMQService:

    def __init__(self):
        self.connection = None
        self.channel = None
        self.correlation_id = None
        self.response = None
        self.response_event = Event()
        self.initialize()

    def initialize(self):
        try:
            logger.info("Initializing RabbitMQ in app")
            params = ConnectionParameters(
                host='rabbitmq',
                blocked_connection_timeout=300
            )
            self.connection = BlockingConnection(params)
            self.channel = self.connection.channel()

            self.channel.exchange_declare(exchange='default', exchange_type='direct')
            self.channel.queue_declare(queue='task', durable=True)
            self.channel.queue_declare(queue='output', durable=True)

            self.channel.queue_bind(queue='task', exchange='default', routing_key='task')
            self.channel.queue_bind(queue='output', exchange='default', routing_key='output')
            self.channel.basic_qos(prefetch_count=1)

            logger.info("RabbitMQ setup complete")
        except Exception:
            logger.exception("Failed to initialize RabbitMQ")
            raise

    def process_request(self, message: str):
        self.response = None
        self.response_event.clear()

        self.correlation_id = str(uuid.uuid4())
        logger.info("Publishing message with correlation_id: %s", self.correlation_id)

        try:
            self.channel.basic_publish(
                exchange='default',
                routing_key='task',
                body=json.dumps({"message": message}),
                properties=BasicProperties(
                    reply_to='output',
                    correlation_id=self.correlation_id
                )
            )
        except Exception as e:
            logger.exception("Failed to publish message to RabbitMQ: %s", e)
            raise

        def on_response(ch, method, props, body):
            if props.correlation_id == self.correlation_id:
                logger.info("Received matching response for correlation_id: %s", props.correlation_id)
                try:
                    self.response = json.loads(body)
                except Exception as e:
                    logger.error("Failed to decode JSON response: %s", e)
                    self.response = {"status": False}
                ch.basic_ack(delivery_tag=method.delivery_tag)
                self.response_event.set()
                ch.stop_consuming()
            else:
                logger.warning("Ignored unmatched response with correlation_id: %s", props.correlation_id)
                ch.basic_nack(delivery_tag=method.delivery_tag)

        try:
            logger.info("Starting consumption on output queue")
            self.channel.basic_consume(
                queue='output',
                on_message_callback=on_response,
                auto_ack=False
            )
            self.channel.start_consuming()
        except Exception as e:
            logger.exception("Error while consuming RabbitMQ message: %s", e)
            raise

        logger.info("Returning response from worker")
        return self.response

    def close(self):
        if self.connection and not self.connection.is_closed:
            logger.info("Closing RabbitMQ connection")
            self.connection.close()
