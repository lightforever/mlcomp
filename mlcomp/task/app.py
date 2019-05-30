from __future__ import absolute_import, unicode_literals
from celery import Celery
import os

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '12345')
REDIS_PORT = os.getenv('REDIS_PORT', '6379')

app = Celery('mlcomp',
             broker=f'redis://{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0',
             backend='redis://',
             include=['mlcomp.task.tasks']
             )

__all__ = ['app']