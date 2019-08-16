from __future__ import absolute_import, unicode_literals
from celery import Celery
import os
import sys

from mlcomp import REDIS_PASSWORD, REDIS_HOST, REDIS_PORT

sys.path.insert(0, os.path.dirname(__file__))

broker = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0'

app = Celery(
    'mlcomp',
    broker=broker,
    backend=broker,
    include=['mlcomp.worker.tasks']
)
__all__ = ['app']
