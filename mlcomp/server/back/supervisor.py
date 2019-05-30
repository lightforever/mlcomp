from mlcomp.utils.logging import logger
import traceback
from mlcomp.task.tasks import execute, queue_list
from mlcomp.db.providers import *
from mlcomp.utils.schedule import start_schedule

def supervisor():
    provider = TaskProvider()
    computer_provider = ComputerProvider()

    try:
        queues = queue_list()
        if len(queues)==0:
            return
        not_ran_tasks = provider.by_status(TaskStatus.NotRan)
        not_ran_tasks = [task for task in not_ran_tasks if not task.debug]
        logger.debug(f'Found {len(not_ran_tasks)} not ran tasks', ComponentType.Supervisor)

        dep_status = provider.dependency_status(not_ran_tasks)
        computers = computer_provider.computers()
        for task in provider.by_status(TaskStatus.InProgress):
            assigned = task.computer_assigned
            computers[assigned]['cpu'] -= task.cpu
            computers[assigned]['gpu'] -= task.gpu
            computers[assigned]['memory'] -= task.memory

        for task in not_ran_tasks:
            if TaskStatus.Stopped.value in dep_status[task.id] or TaskStatus.Failed.value in dep_status[task.id]:
                provider.change_status(task, TaskStatus.Skipped)
                continue

            status_set = set(dep_status[task.id])
            if len(status_set) != 0 and status_set != {TaskStatus.Success.value}:
                continue

            for name, computer in computers.items():
                if task.gpu > computer['gpu'] or task.cpu > computer['cpu'] or task.memory > computer['memory']:
                    continue
                if task.computer is not None and task.computer != computer.name:
                    continue

                queue = f'{computer["name"]}_{task.dag_rel.docker_img or "default"}'
                if queue not in queues:
                    continue

                r = execute.apply_async((task.id,), queue=queue)
                task.status = TaskStatus.Queued.value
                task.computer_assigned = computer['name']
                task.celery_id = r.id

                provider.update()

                computer['gpu'] -= task.gpu
                computer['cpu'] -= task.cpu
                computer['memory'] -= task.memory
                break

    except Exception:
        logger.error(traceback.format_exc(), ComponentType.Supervisor)


def register_supervisor():
    start_schedule([(supervisor, 1)])
