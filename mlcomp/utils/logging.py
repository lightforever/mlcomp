import os
import logging
import sys
from logging.handlers import RotatingFileHandler

from mlcomp import LOG_FOLDER, LOG_NAME, FILE_LOG_LEVEL, DB_LOG_LEVEL, \
    CONSOLE_LOG_LEVEL
from mlcomp.db.core import Session
from mlcomp.db.providers import LogProvider
from mlcomp.db.models import Log
from mlcomp.utils.misc import now

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))


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
            if s[-1:] != '\n':
                s = s + '\n'
            s = s + record.exc_text
        if record.stack_info:
            if s[-1:] != '\n':
                s = s + '\n'
            s = s + self.formatStack(record.stack_info)
        return s


class DbHandler(logging.Handler):
    """
    A handler class which writes logging records, appropriately formatted,
    to the database.
    """

    def __init__(self, session: Session):
        """
        Initialize the handler.
        """
        logging.Handler.__init__(self)
        self.provider = LogProvider(session)

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
                message=record.msg[-16000:],
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


def create_logger(session: Session, name: str, db=True, file=True,
                  console=True):
    logger = logging.Logger(name)
    logger.handlers = []

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(CONSOLE_LOG_LEVEL)
        console_handler.stream = sys.stdout
        logger.handlers.append(console_handler)

    if file:
        file_path = os.path.join(LOG_FOLDER, f'{LOG_NAME}.txt')
        file_handler = RotatingFileHandler(file_path)
        file_handler.setLevel(FILE_LOG_LEVEL)
        file_handler.maxBytes = 10485760
        file_handler.backupCount = 1
        logger.handlers.append(file_handler)

    if db:
        handler = DbHandler(session)
        handler.setLevel(DB_LOG_LEVEL)
        logger.handlers.append(handler)

    for h in logger.handlers:
        fmt = '%(asctime)s.%(msecs)03d %(levelname)s' \
              ' %(module)s - %(funcName)s: %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        if isinstance(h, DbHandler):
            fmt, datefmt = None, None
        h.formatter = Formatter(fmt=fmt, datefmt=datefmt)

    # ignore messages from some libraries
    class NoRunningFilter(logging.Filter):
        def filter(self, record):
            return 'ran tasks' not in str(record.msg)

    for k in logging.root.manager.loggerDict:
        if 'apscheduler' in k:
            logging.getLogger(k).setLevel(logging.ERROR)
        if 'mlcomp' in k:
            logging.getLogger(k).addFilter(NoRunningFilter())
        if 'serializer' in k:
            logging.getLogger(k).setLevel(logging.ERROR)

    return logger


__all__ = ['create_logger']
