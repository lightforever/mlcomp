import os
import logging
from logging.config import dictConfig

from mlcomp.db.providers import LogProvider
from mlcomp.db.models import Log
from mlcomp.utils.misc import now

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))


class Formatter(logging.Formatter):
    def format(self, record):
        if not record.pathname.startswith(ROOT):
            return super().format(record)

        msg = str(record.msg)
        if record.args:
            try:
                msg = msg % record.args
            except Exception:
                pass
        record.message = msg
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
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


class DbHandler(logging.Handler):
    """
    A handler class which writes logging records, appropriately formatted,
    to the database.
    """
    def __init__(self):
        """
        Initialize the handler.
        """
        logging.Handler.__init__(self)
        self.provider = LogProvider()

    def emit(self, record):
        """
        Emit a record.
        """
        try:
            if not record.pathname.startswith(ROOT):
                return

            assert 1 <= len(record.args), \
                'Args weer not been provided for logging'
            assert len(record.args) <= 4, 'Too many args for logging'

            step = None
            task = None
            computer = None

            if len(record.args) == 1:
                component = record.args[0]
            elif len(record.args) == 2:
                component, computer = record.args
            elif len(record.args) == 3:
                component, computer, task = record.args
            else:
                component, computer, task, step = record.args

            if not isinstance(component, int):
                component = component.value

            module = os.path.relpath(record.pathname, ROOT). \
                replace(os.sep, '.').replace('.py', '')
            if record.funcName and record.funcName != '<module>':
                module = f'{module}:{record.funcName}'
            log = Log(
                message=record.msg[:4000],
                time=now(),
                level=record.levelno,
                step=step,
                component=component,
                line=record.lineno,
                module=module,
                task=task,
                computer=computer
            )
            self.provider.add(log)
        except Exception:
            self.handleError(record)


CONSOLE_LOG_LEVEL = os.getenv('CONSOLE_LOG_LEVEL', 'DEBUG')
DB_LOG_LEVEL = os.getenv('DB_LOG_LEVEL', 'DEBUG')

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '[%(asctime)s][%(threadName)s] %(funcName)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': CONSOLE_LOG_LEVEL,
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        },
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False
        }
    }
}


def create_logger():
    dictConfig(LOGGING)
    logger = logging.getLogger()
    handler = DbHandler()
    handler.setLevel(DB_LOG_LEVEL)
    logger.handlers.append(handler)

    for h in logger.handlers:
        h.formatter = Formatter()
    return logger


logger = create_logger()

__all__ = ['create_logger', 'logger']
