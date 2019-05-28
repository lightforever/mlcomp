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
        }
    },

    'loggers': {
        '': {
            'handlers': [
                'console'
            ],
            'level': 'INFO',
            'propagate': False
        }
    }
}

dictConfig(LOGGING)