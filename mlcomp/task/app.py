from __future__ import absolute_import, unicode_literals
from celery import Celery

app = Celery('mlcomp',
             broker='amqp://',
             backend='amqp://',
             include=['mlcomp.task.tasks']
             )