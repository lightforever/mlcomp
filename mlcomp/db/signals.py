from mlcomp.db.models import *
from sqlalchemy import event

# @event.listens_for(Task.status, 'modified')
# def receive_modified(target, initiator):
#     print(target, initiator)

# standard decorator style
@event.listens_for(Task, 'after_update')
def task_after_update(mapper, connection, target):
    print('task updated')