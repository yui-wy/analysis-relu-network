import logging
from logging.handlers import QueueHandler, QueueListener


class Logger(logging.Logger):
    def __init__(self, name, level=0):
        super().__init__(name, level)

    def setting(self, *handlers: logging.Handler, multi: bool = False):
        self.multi = False
        self.propagate = False
        # On windows, spawn model can not use the queue.
        if multi:
            self.multi = True
            self._set_multi_processing(handlers)
        for handler in handlers:
            self.addHandler(handler)

    def _set_multi_processing(self, handlers: logging.Handler):
        import multiprocessing as mp

        self.multi = True
        self.queue = mp.Manager().Queue()
        handler = QueueHandler(self.queue)
        self.addHandler(handler)
        self.queue_listener = QueueListener(self.queue, *handlers, respect_handler_level=True)
        self.queue_listener.start()

    def close(self):
        if self.multi:
            self.queue_listener.stop()


def get_logger(
    title: str = "log",
    path: str = None,
    formatter: logging.Formatter = logging.Formatter("[%(asctime)s] - %(name)s : %(message)s"),
    multi: bool = False,
):
    logger = logging.getLogger(title)
    logger.setLevel(level=logging.INFO)
    if path is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(path, mode='w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    # logger.setting(handler, multi=multi)
    logger.propagate = False
    logger.addHandler(handler)
    return logger
