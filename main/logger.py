import logging
import time
import copy

import colorama

from functools import wraps
from colorama import Fore, init

init(autoreset=True)


def timer(func):
    @wraps(func)
    def wrapper(*args):
        logger = logging.getLogger(__name__)
        st = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        internal = int(end - st)*1000
        logger.debug(f'{func.__name__} cost {internal} ms')
        return result

    return wrapper


class ColorFormattor(logging.Formatter):
    def __init__(self,
                 fmt,
                 msg_prefix='[*]',
                 datefmt=None,
                 style=None,
                 color=True):
        super(ColorFormattor, self).__init__(fmt=fmt,
                                             datefmt=datefmt,
                                             style=style)
        self.msg_prefix = msg_prefix
        self.use_color = color
        self.color_table = {
            'levelname': Fore.RED,
            'module': Fore.BLUE,
            'name': Fore.BLUE,
            'funcName': Fore.BLUE,
            'asctime': Fore.BLUE,
            'process': Fore.BLUE,
            'message': Fore.GREEN
        }

    def format(self, record):
        record.message = record.getMessage()
        if self.msg_prefix:
            record.message = self.msg_prefix + ' ' + record.message
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        record.process = str(record.process)
        if self.use_color:
            for k, v in self.color_table.items():
                attr = getattr(record, k)
                setattr(record, k, v + attr + Fore.RESET)
        s = self.formatMessage(record)
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        if record.stack_info:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + self.formatStack(record.stack_info)
        return s


def get_logger():
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    t = time.strftime("%Y-%m-%d-%H-%M", time.gmtime())
    filename = f'logs/run/run_{t}.log'
    fh = logging.FileHandler(filename)
    ch.setLevel(logging.INFO)
    fh.setLevel(logging.DEBUG)

    msg_fmt = '[{levelname:^5s}] time {asctime} moudle {name} function {funcName:16s} +{lineno:<4d} ' \
              ' pid [{process}] \n {message}'
    formatter_ch = ColorFormattor(fmt=msg_fmt, style='{', datefmt=t)
    formatter_fh = ColorFormattor(fmt=msg_fmt, style='{', datefmt=t, color=False)

    ch.setFormatter(formatter_ch)
    fh.setFormatter(formatter_fh)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


global_logger = get_logger()
