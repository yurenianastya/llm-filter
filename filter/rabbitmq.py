import json
import os
from pika import BasicProperties, SelectConnection, ConnectionParameters, PlainCredentials

from filtering import is_message_safe

def process_message(ch, method, properties, body):
    response = {}
    try:
        request = json.loads(body)
        input_message = request.get('message')
        correlation_id = properties.correlation_id

        if not input_message or not correlation_id:
            raise ValueError("Invalid message format")

        response = is_message_safe(input_message)

    except Exception as e:
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
    except Exception as e:
        print(f"Error processing message: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

def on_channel_open(channel):
    print(f"Channel opened: {channel}")
    channel.exchange_declare(exchange='message_exchange', exchange_type='direct')
    channel.queue_declare(queue='task', durable=True)
    channel.queue_declare(queue='output', durable=True)
    channel.basic_consume(queue='task', on_message_callback=process_message)
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