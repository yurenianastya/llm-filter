import logging
from rabbitmq import initialize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    try:
        logger.info("Worker initializing...")
        initialize()
        logger.info("Worker shutdown cleanly")
    except KeyboardInterrupt:
        logger.warning("Worker interrupted by user")
        exit(0)
    except Exception as e:
        logger.exception("Worker failed during initialization or runtime")
        exit(1)
