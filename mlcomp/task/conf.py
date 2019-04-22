import logging
from logging.config import dictConfig
import sys

logger = logging.getLogger(__name__)


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
            'level': 'INFO',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        },
        # 'file': {
        #     'class': 'logging.handlers.RotatingFileHandler',
        #     'level': 'INFO',
        #     'formatter': 'simple',
        #     'filename': os.path.join(BASE_DIR, f"{sys.modules['__main__'].__file__.split('.')[0]}.log"),
        #     'mode': 'w',
        #     'maxBytes': 10485760,
        #     'backupCount': 5,
        #     'encoding': 'utf-8'
        # }
    },

    'loggers': {
        '': {
            'handlers': [
                'console',
                #'file'
            ],
            'level': 'INFO',
            'propagate': False
        }
    }
}

dictConfig(LOGGING)