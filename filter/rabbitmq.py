import json
import logging

from pika import BasicProperties, SelectConnection, ConnectionParameters, PlainCredentials

from filtering import is_message_safe

logger = logging.getLogger(__name__)

def process_message(ch, method, properties, body):
    response = {}
    try:
        request = json.loads(body)
        input_message = request.get('message')
        correlation_id = properties.correlation_id

        logger.info("Received message with correlation_id: %s", correlation_id)

        if not input_message or not correlation_id:
            raise ValueError("Invalid message format")

        response = is_message_safe(input_message)
        logger.info("Filtering complete for correlation_id: %s", correlation_id)

    except Exception as e:
        logger.exception("Failed to process incoming message: %s", e)
        response = {
            "status": False,
            "classification_result": {
                "label": f"ERROR: {e}",
                "score": 0.0
            },
            "semantic_result": {
                "score": 0.0
            }
        }

    try:
        ch.basic_publish(
            exchange='message_exchange',
            routing_key=properties.reply_to,
            properties=BasicProperties(
                correlation_id=properties.correlation_id
            ),
            body=json.dumps(response)
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.info("Published response for correlation_id: %s", correlation_id)
    except Exception as e:
        logger.exception("Error sending response message: %s", e)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

def on_channel_open(channel):
    logger.info("Channel opened: %s", channel)
    channel.exchange_declare(exchange='message_exchange', exchange_type='direct')
    channel.queue_declare(queue='task', durable=True)
    channel.queue_declare(queue='output', durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='task', on_message_callback=process_message)
    logger.info("Worker registered and consuming from 'task' queue")

def on_connected(connection):
    logger.info("Connection established to RabbitMQ")
    connection.channel(on_open_callback=on_channel_open)

def on_open_error(connection, exception):
    logger.error("Connection failed during setup: %s", exception)
    connection.ioloop.stop()

def on_connection_closed(connection, reason):
    logger.warning("Connection closed: %s", reason)
    connection.ioloop.stop()

def initialize():
    try:
        logger.info("Initializing RabbitMQ worker...")
        params = ConnectionParameters(
            host='rabbitmq',
            blocked_connection_timeout=300
        )
        connection = SelectConnection(
            parameters=params,
            on_open_callback=on_connected,
            on_open_error_callback=on_open_error,
            on_close_callback=on_connection_closed
        )

        connection.ioloop.start()

    except Exception as e:
        logger.exception("Exception occurred during worker setup: %s", e)