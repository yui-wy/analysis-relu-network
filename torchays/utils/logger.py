import logging


def get_logger(
    title: str = "log",
    path: str = None,
    formatter: logging.Formatter = logging.Formatter("[%(asctime)s] - %(name)s : %(message)s"),
):
    logger = logging.getLogger(title)
    logger.setLevel(level=logging.INFO)
    logger.propagate = False
    if path is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(path, mode='w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
