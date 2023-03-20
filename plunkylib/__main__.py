# base class
from pathlib import Path
from dotenv import load_dotenv
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
from loguru import logger

from .plunkycore import *
from .plunkycli import *

if __name__ == "__main__":
    # logging config should exclude warnings
    config = {
        "handlers": [
            {"sink": "file.log", "format": "{time} - {message}"},
        ],
        "extra": {"user": "someone"},
    }
    logger.configure(**config)
    import log
    log.silence("datafiles")
    log.silence("openai")
    log.silence("chronological")
    app()