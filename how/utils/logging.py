"""Logging-related functionality"""

import time
import logging

# Logging

def init_logger(log_path):
    """Return a logger instance which logs to stdout and, if log_path is not None, also to a file"""
    logger = logging.getLogger("HOW")
    logger.setLevel(logging.DEBUG)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(logging.Formatter('%(name)s %(levelname)s: %(message)s'))
    logger.addHandler(stdout_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Stopwatch

class LoggingStopwatch:
    """Stopwatch context that produces one message when entered and another one when exited,
        with the time spent in the context embedded in the exiting message.

    :param str message: Message to be logged at the start and finish. If the first word
            of the message ends with 'ing', convert to passive for finish message.
    :param callable log_start: Will be called with given message at the start
    :param callable log_finish: Will be called with built message at the finish. If None, use
            log_start
    """

    def __init__(self, message, log_start, log_finish=None):
        self.message = message
        self.log_start = log_start
        self.log_finish = log_finish if log_finish is not None else log_start
        self.time0 = None

    def __enter__(self):
        self.time0 = time.time()
        if self.log_start:
            self.log_start(self.message.capitalize())

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Build message
        words = self.message.split(" ")
        secs = "%.1fs" % (time.time() - self.time0)
        if words[0].endswith("ing"):
            words += [words.pop(0).replace("ing", "ed"), "in", secs]
        else:
            words += ["(%.1f)" % secs]

        # Log message
        if self.log_finish:
            self.log_finish(" ".join(words).capitalize())
