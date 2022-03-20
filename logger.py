from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler

LOGGER = None


def init_logger(log_file='./train.log'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    global LOGGER
    LOGGER = logger


def Logger():
    return LOGGER
