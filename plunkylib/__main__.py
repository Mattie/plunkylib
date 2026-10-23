"""Shared startup for the module and installed plunky command."""

import logging

from loguru import logger

from .plunkycli import app


def main():
    """Configure CLI logging and dispatch the requested command."""
    config = {
        "handlers": [
            {"sink": "file.log", "format": "{time} - {message}"},
        ],
        "extra": {"user": "someone"},
    }
    logger.configure(**config)
    for name in ("snapclass", "openai", "chronological"):
        logging.getLogger(name).setLevel(logging.WARNING)
    app()


if __name__ == "__main__":
    main()
