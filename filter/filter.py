from rabbitmq import start_consumer, stop_consumer
import signal
import threading

signal.signal(signal.SIGTERM, stop_consumer)
signal.signal(signal.SIGINT, stop_consumer)

thread = threading.Thread(target=start_consumer)
thread.start()