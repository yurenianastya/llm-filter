import logging
import sys

from src.core.rabbitmq import RabbitMQService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    try:
        logger.info("Worker initializing...")
        service = RabbitMQService()
        service.initialize()
        logger.info("Worker shutdown cleanly")
    except KeyboardInterrupt:
        logger.warning("Worker interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Worker failed during initialization or runtime: %s", e)
        sys.exit(1)
