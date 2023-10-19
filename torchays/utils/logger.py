import logging


def get_logger(title: str = "log"):
    logger = logging.getLogger(title)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] - %(name)s : %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger
