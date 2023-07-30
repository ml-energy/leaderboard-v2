import logging
import concurrent.futures
import time

class AsyncLogHandler(logging.Handler):
    def __init__(self, handler):
        super().__init__()
        self.handler = handler
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def emit(self, record):
        self.executor.submit(self.handler.emit, record)

def setup_logger():
    file_handler = logging.FileHandler('my_log_file.log')  # Send log messages to the file
    # console_handler = logging.StreamHandler()  # Send log messages to the console

    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    async_handler = AsyncLogHandler(file_handler)
    async_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger()
    logger.addHandler(async_handler)
    # logger.setLevel(logging.DEBUG)

    return logger
