from db.providers import TaskProvider
from task.executors import Executor
import json
from utils.config import Config
from task.app import app
from db.enums import *

@app.task
def execute(id:int):
    provider = TaskProvider()
    task = provider.by_id(id)
    provider.change_status(task, TaskStatus.InProgress)

    try:
        config = Config(json.loads(task.config))
        executor = Executor.from_config(task.executor, config)
        executor()

        provider.change_status(task, TaskStatus.Success)
    except Exception:
        provider.change_status(task, TaskStatus.Failed)

if __name__=='__main__':
    from task.tasks import execute
    execute.delay(42)