from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
import pika
import json
import os
import signal
import threading

load_dotenv(find_dotenv())

model_path = os.environ.get('HF_MODEL')
pipe = pipeline('text-classification', model=model_path, tokenizer=model_path)

RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST')
connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
channel = connection.channel()

channel.queue_declare(queue='preprocessing_request', durable=True)
channel.queue_declare(queue='preprocessing_result', durable=True)
channel.basic_qos(prefetch_count=1)


def classify_message(message):
    try:
        output = pipe(message)
        return output
    except Exception as e:
        print(f'Error in classification: {e}')
        return None


def process_message(ch, method, properties, body):
    try:
        request = json.loads(body)
        message = request.get('message')
        correlation_id = request.get('correlation_id')

        if not message or not correlation_id:
            raise ValueError('Invalid message format')

        filter_output = classify_message(message)
        if filter_output:
            label = filter_output[0].get('label')
            score =  filter_output[0].get('score')
            if label == 'NOT_TOXIC' or score < 0.2:
                response = {'message': 'The prompt passed the filtering.', 'filter_result': filter_output}
            else:
                response = {'message': 'The prompt did not pass the filtering.', 'filter_result': filter_output}
        else:
            response = {'message': 'The prompt was not processed correctly.', 'filter_result': 'Not available'}

        ch.basic_publish(
            exchange='',
            routing_key=properties.reply_to,
            properties=pika.BasicProperties(correlation_id=properties.correlation_id),
            body=json.dumps(response)
        )
    except Exception as e:
        response = {'message': 'Error processing the request', 'filter_result': f'Error: {e}'}
        ch.basic_publish(
            exchange='',
            routing_key=properties.reply_to,
            properties=pika.BasicProperties(correlation_id=properties.correlation_id),
            body=json.dumps(response)
        )
    finally:
        ch.basic_ack(delivery_tag=method.delivery_tag)


def graceful_shutdown():
    print('Shutting down gracefully...')
    channel.stop_consuming()
    connection.close()

signal.signal(signal.SIGTERM, graceful_shutdown)

def start_consuming():
    channel.basic_consume(queue='preprocessing_request', on_message_callback=process_message)
    channel.start_consuming()

thread = threading.Thread(target=start_consuming)
thread.start()