import json
import logging

from pika import BasicProperties, SelectConnection, ConnectionParameters, PlainCredentials

from src.core.filter import is_safe

logger = logging.getLogger(__name__)

class RabbitMQService:

    def __init__(self):
        self.connection_params = ConnectionParameters(
            host='rabbitmq',
            blocked_connection_timeout=300,
        )
        self.connection = None
        self.channel = None

    def initialize(self):
        try:
            logger.info("Initializing RabbitMQ in filter")
            self.connection = SelectConnection(
                parameters=self.connection_params,
                on_open_callback=self.on_connected,
                on_open_error_callback=self.on_open_error,
                on_close_callback=self.on_connection_closed
            )
            self.connection.ioloop.start()
        except Exception:
            logger.exception("Worker failed to start")
            raise

    def on_connected(self, connection):
        logger.info("Connected to RabbitMQ")
        connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, channel):
        logger.info("Channel opened")
        self.channel = channel
        channel.exchange_declare(
            exchange='default',
            exchange_type='direct',
            callback=lambda _: self.setup_queues(channel)
        )

    def setup_queues(self, channel):
        def on_output_declared(_):
            channel.queue_bind(queue='task', exchange='default', routing_key='task')
            channel.queue_bind(queue='output', exchange='default', routing_key='output')
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue='task', on_message_callback=self._process_message)
            logger.info("Worker ready and consuming")

        def on_task_declared(_):
            channel.queue_declare(queue='output', durable=True, callback=on_output_declared)

        channel.queue_declare(queue='task', durable=True, callback=on_task_declared)

    def _process_message(self, ch, method, properties, body):
        response = {}
        try:
            request = json.loads(body)
            input_message = request.get('message')
            correlation_id = properties.correlation_id

            logger.info("Received message with correlation_id: %s", correlation_id)

            if not input_message or not correlation_id:
                raise ValueError("Invalid message format")

            response = is_safe(input_message)
            logger.info("Filtering complete for correlation_id: %s", correlation_id)

        except Exception as e:
            logger.exception("Failed to process message: %s", e)
            response = {
                "error": f"ERROR: {e}"
            }

        try:
            ch.basic_publish(
                exchange='default',
                routing_key=properties.reply_to,
                properties=BasicProperties(correlation_id=properties.correlation_id),
                body=json.dumps(response)
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info("Published response for correlation_id: %s", properties.correlation_id)
        except Exception as e:
            logger.exception("Error sending response: %s", e)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def on_open_error(self, connection, exception):
        logger.error("Connection failed: %s", exception)
        connection.ioloop.stop()

    def on_connection_closed(self, connection, reason):
        logger.warning("Connection closed: %s", reason)
        connection.ioloop.stop()
