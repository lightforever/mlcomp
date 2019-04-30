from mlcomp.db.providers import TaskProvider
from mlcomp.task.executors import Executor
import json
from mlcomp.utils.config import Config
from mlcomp.utils.logging import logger
from mlcomp.task.app import app
from mlcomp.db.enums import *
from mlcomp.task.storage import Storage
import sys
import traceback

@app.task
def execute(id:int):
    provider = TaskProvider()
    storage = Storage()
    thismodule = sys.modules[__name__]
    task = provider.by_id(id)

    try:
        provider.change_status(task, TaskStatus.InProgress)

        folder = storage.download(id)
        config = Config(json.loads(task.config))

        imported = storage.import_folder(thismodule, folder, config['executors'][task.executor]['type'])
        if not imported:
            logger.warn(f'Task {task.id}, the executor {task.executor} has not been imported')

        executor = Executor.from_config(task.executor, config)
        executor()

        provider.change_status(task, TaskStatus.Success)
    except Exception:
        provider.change_status(task, TaskStatus.Failed)
        logger.error(traceback.format_exc())

if __name__=='__main__':
    from task.tasks import execute
    execute.delay(42)