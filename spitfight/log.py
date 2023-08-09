from __future__ import annotations

import queue
import logging
from logging.handlers import QueueHandler, QueueListener

ROOT_LOGGER_NAMES: list[str | None] = []
ROOT_LOGGER_QUEUE_LISTENERS: list[QueueListener] = []


def init_queued_root_logger(
    name: str | None,
    filepath: str,
    level: int = logging.INFO,
) -> None:
    """Initialize a queue-based pseudo-root logger.

    The pseudo-root logger will aggregate log messages from children
    loggers under its namespace and send them to a queue. A QueueListener,
    running in a separate thread, will then process the messages in the
    queue and send them to the configured handlers.
    """
    global ROOT_LOGGER_NAMES, ROOT_LOGGER_QUEUE_LISTENERS

    # Make this function idempotent.
    if name in ROOT_LOGGER_NAMES:
        return

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    shared_queue = queue.SimpleQueue()
    queue_handler = QueueHandler(shared_queue)
    logger.addHandler(queue_handler)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s](%(filename)s:%(lineno)d) %(message)s"
    )

    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filepath, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    queue_listener = QueueListener(shared_queue, file_handler, stderr_handler)
    queue_listener.start()

    ROOT_LOGGER_NAMES.append(name)
    ROOT_LOGGER_QUEUE_LISTENERS.append(queue_listener)


def shutdown_queued_root_loggers() -> None:
    """Shutdown all queue-based pseudo-root loggers.

    This is necessary to make sure all log messages are flushed
    before the application exits.
    """
    for queue_listener in ROOT_LOGGER_QUEUE_LISTENERS:
        queue_listener.stop()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with the given name and level."""
    # Don't reconfigure existing loggers.
    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True

    return logger
