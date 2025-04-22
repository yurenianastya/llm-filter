from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

HF_MODEL = os.environ.get('HF_MODEL')
RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST')