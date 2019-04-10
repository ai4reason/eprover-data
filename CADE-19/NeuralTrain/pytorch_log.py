import logging
import socket

SEPARATOR_GLOBAL = ' ; '
SEPARATOR = SEPARATOR_GLOBAL

KWARG = 'KWARG' + SEPARATOR + '%s' + SEPARATOR + '%s'

BATCH_LOSS = 'BATCH_LOSS' + SEPARATOR + '%s' + SEPARATOR + '%s'
EPOCH_LOSS = 'EPOCH_LOSS' + SEPARATOR + '%s' + SEPARATOR + '%s'
BATCH_NUM = 'BATCH_NUM' + SEPARATOR + '%s'
EPOCH_NUM = 'EPOCH_NUM' + SEPARATOR + '%s'
EPOCH_SUCCRATE = 'EPOCH_SUCCRATE' + SEPARATOR + '%s' + SEPARATOR + '%s'
EPOCH_POSLABELS = 'EPOCH_POSLABELS' + SEPARATOR + '%s' + SEPARATOR + '%s'


class PyTorchLog:
    def __init__(self, name, filename):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # new_format = '%(asctime)s' + SEPARATOR_GLOBAL + '%(name)s' + SEPARATOR_GLOBAL + '%(levelname)s' + SEPARATOR_GLOBAL + '%(message)s'
        new_format = '%(asctime)s' + SEPARATOR_GLOBAL + '%(levelname)s' + SEPARATOR_GLOBAL + '%(message)s'
        
        formatter = logging.Formatter(new_format)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.info(name)
        self.logger.info(socket.gethostname())

    def critical(self, *args, **kwargs):
        self.logger.critical(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)

    def notset(self, *args, **kwargs):
        self.logger.notset(*args, **kwargs)

    def kwarg(self, a, v):
        self.logger.info(KWARG, a, v)

    def batch_loss(self, loss, on='TRAIN'):
        self.logger.info(BATCH_LOSS, on, loss)

    def epoch_loss(self, loss, on='TRAIN'):
        self.logger.info(EPOCH_LOSS, on, loss)

    def batch_num(self, batch):
        self.logger.info(BATCH_NUM, batch)

    def epoch_num(self, epoch):
        self.logger.info(EPOCH_NUM, epoch)

    def epoch_succrate(self, epoch_succrate, on='TRAIN'):
        self.logger.info(EPOCH_SUCCRATE, on, epoch_succrate)

    def epoch_poslabels(self, poslabels, on='TRAIN'):
        self.logger.info(EPOCH_POSLABELS, on, poslabels)
