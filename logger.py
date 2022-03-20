from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler

LOGGER = None
handlers = []


def init_logger(log_file='./train.log'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    handlers.append(handler1)
    handlers.append(handler2)

    global LOGGER
    LOGGER = logger


def close_logger():
    global LOGGER
    for handler in handlers:
        LOGGER.removeHandler(handler)


def Logger():
    global LOGGER
    return LOGGER
