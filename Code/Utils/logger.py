import logging
import os
from datetime import datetime

from config import LOG_PATH



logger = logging.getLogger()

def initialize_logger(filename = os.path.join(LOG_PATH, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s -> %(message)s",
        filename=filename,
        filemode="w",
    )

    logger.info(f"Logger initialized in filename {datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

def get_logger():
    return logger
