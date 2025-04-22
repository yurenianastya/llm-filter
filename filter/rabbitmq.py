import pika
import json

import torch
from config import RABBITMQ_HOST
from model import classify_message

connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
channel = connection.channel()
channel.queue_declare(queue='preproc_request', durable=True)
channel.queue_declare(queue='preproc_result', durable=True)
channel.basic_qos(prefetch_count=1)

def process_message(ch, method, properties, body):
    try:
        request = json.loads(body)
        message = request.get('message')
        correlation_id = request.get('correlation_id')

        if not message or not correlation_id:
            raise ValueError("Invalid message format")

        result = classify_message(message)
        label = result[0].get('label') if result else None
        score = result[0].get('score') if result else 0

        if label == 'NOT_TOXIC' or score < 0.2:
            response = {'message': 'Passed', 'filter_result': result}
        else:
            response = {'message': 'Filtered out', 'filter_result': result}
    except Exception as e:
        response = {'message': 'Error', 'filter_result': str(e)}
    finally:
        ch.basic_publish(
            exchange='',
            routing_key=properties.reply_to,
            properties=pika.BasicProperties(correlation_id=properties.correlation_id),
            body=json.dumps(response)
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)

def start_consumer():
    channel.basic_consume(queue='preproc_request', on_message_callback=process_message)
    classify_message("warm up the model")
    channel.start_consuming()

def stop_consumer(*_):
    print("Graceful shutdown")
    channel.stop_consuming()
    connection.close()