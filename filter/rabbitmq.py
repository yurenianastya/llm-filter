import json
import os
from pika import BasicProperties, SelectConnection, ConnectionParameters, PlainCredentials

from model import classify_message

EXCHANGE_NAME = os.environ.get("EXCHANGE_NAME")
QUEUE_TASK = os.environ.get("QUEUE_TASK")
QUEUE_RESULT = os.environ.get("QUEUE_RESULT")
RABBITMQ_URL = os.environ.get("RABBITMQ_HOST")

def process_message(ch, method, properties, body):
    response = {}
    try:
        request = json.loads(body)
        input_message = request.get('message')
        correlation_id = properties.correlation_id

        if not input_message or not correlation_id:
            raise ValueError("Invalid message format")

        result = classify_message(input_message)
        label = result[0].get('label') if result else None
        score = result[0].get('score') if result else 0

        if label == 'NOT_TOXIC' or score < 0.4:
            response = {'status': 'Passed', 'filter_result': result}
        else:
            response = {'status': 'Filtered out', 'filter_result': result}

    except Exception as e:
        response = {'status': 'Error', 'filter_result': str(e)}

    try:
        ch.basic_publish(
            exchange=EXCHANGE_NAME,
            routing_key=properties.reply_to,
            properties=BasicProperties(
                correlation_id=properties.correlation_id
            ),
            body=json.dumps(response)
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print(f"Error processing message: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

def on_channel_open(channel):
    print(f"Channel opened: {channel}")
    channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type='direct')
    channel.queue_declare(queue=QUEUE_TASK, durable=True)
    channel.queue_declare(queue=QUEUE_RESULT, durable=True)
    channel.basic_consume(queue=QUEUE_TASK, on_message_callback=process_message)
    print("filter worker is registered")
    channel.basic_qos(prefetch_count=1)

def on_connected(connection):
    connection.channel(on_open_callback=on_channel_open)

def on_open_error(connection, exception):
    print("Failed to connect:", exception)
    connection.ioloop.stop()

def on_connection_closed(connection, reason):
    print(f"Connection closed: {reason}")
    connection.ioloop.stop()

def initialize():
    try:
        params = ConnectionParameters(
            host='rabbitmq',
            port=5672,
            virtual_host='/',
            credentials=PlainCredentials('guest', 'guest'),
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
        print("Exception occurred during setup:", e)