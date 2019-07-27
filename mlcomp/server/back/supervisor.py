import datetime
import traceback
from collections import defaultdict

import numpy as np
from sqlalchemy.exc import ProgrammingError

from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType, TaskStatus, TaskType
from mlcomp.db.models import Task, Auxiliary
from mlcomp.utils.io import yaml_dump, yaml_load
from mlcomp.utils.logging import create_logger
from mlcomp.utils.misc import now
from mlcomp.worker.tasks import execute
from mlcomp.db.providers import *
from mlcomp.utils.schedule import start_schedule


class SupervisorBuilder:
    def __init__(self):
        self.logger = create_logger()
        self.provider = None
        self.computer_provider = None
        self.docker_provider = None
        self.auxiliary_provider = None
        self.queues = None
        self.not_ran_tasks = None
        self.dep_status = None
        self.computers = None
        self.auxiliary = {}

    def create_base(self):
        self.provider = TaskProvider()
        self.computer_provider = ComputerProvider()
        self.docker_provider = DockerProvider()
        self.auxiliary_provider = AuxiliaryProvider()

        self.queues = [f'{d.computer}_{d.name}'
                       for d in self.docker_provider.all()
                       if d.last_activity >=
                       now() - datetime.timedelta(seconds=10)]

        self.auxiliary['queues'] = self.queues

    def load_tasks(self):
        not_ran_tasks = self.provider.by_status(TaskStatus.NotRan)
        self.not_ran_tasks = [task for task in not_ran_tasks if not task.debug]
        self.logger.debug(f'Found {len(not_ran_tasks)} not ran tasks',
                          ComponentType.Supervisor)

        self.dep_status = self.provider.dependency_status(self.not_ran_tasks)

        self.auxiliary['not_ran_tasks'] = [
            {
                'id': t.id,
                'name': t.name,
                'dep_status': self.dep_status.get(t.id, [])
            }
            for t in not_ran_tasks
        ]

    def load_computers(self):
        computers = self.computer_provider.computers()
        for computer in computers.values():
            computer['gpu'] = [0] * computer['gpu']
            computer['ports'] = set()
            computer['cpu_total'] = computer['cpu']
            computer['memory_total'] = computer['memory']
            computer['gpu_total'] = len(computer['gpu'])

        for task in self.provider.by_status(TaskStatus.Queued,
                                            TaskStatus.InProgress):
            if task.computer_assigned is None:
                continue
            assigned = task.computer_assigned
            comp_assigned = computers[assigned]
            comp_assigned['cpu'] -= task.cpu

            if task.gpu_assigned is not None:
                comp_assigned['gpu'][task.gpu_assigned] = task.id
            comp_assigned['memory'] -= task.memory

            info = yaml_load(task.additional_info)
            if 'distr_info' in info:
                dist_info = info['distr_info']
                if dist_info['rank'] == 0:
                    comp_assigned['ports'].add(dist_info['master_port'])

        self.computers = [
            {**value, 'name': name}
            for name, value in computers.items()
        ]

        self.auxiliary['computers'] = self.computers

    def process_to_celery(self, task: Task, queue: str, computer: dict):
        r = execute.apply_async((task.id,), queue=queue)
        task.status = TaskStatus.Queued.value
        task.computer_assigned = computer['name']
        task.celery_id = r.id

        if task.gpu_assigned is not None:
            computer['gpu'][task.gpu_assigned] = task.id
            computer['cpu'] -= task.cpu
            computer['memory'] -= task.memory

        self.provider.update()

    def create_service_task(self,
                            task: Task,
                            gpu_assigned=None,
                            distr_info: dict = None):
        task_dict = {k: v for k, v in task.__dict__.items() if
                     not k.startswith('_') and v is not None}
        new_task = Task(**task_dict)
        new_task.additional_info = task.additional_info

        new_task.id = None
        new_task.type = TaskType.Service.value
        new_task.gpu_assigned = gpu_assigned
        new_task.parent = task.id
        if distr_info:
            additional_info = yaml_load(new_task.additional_info)
            additional_info['distr_info'] = distr_info
            new_task.additional_info = yaml_dump(additional_info)
        return self.provider.add(new_task)

    def find_port(self, c: dict, docker_name: str):
        docker = self.docker_provider.get(c['name'], docker_name)
        ports = list(map(int, docker.ports.split('-')))
        for p in range(ports[0], ports[1] + 1):
            if p not in c['ports']:
                return p
        raise Exception(f'All ports in {c["name"]} are taken')

    def process_task(self, task: Task):
        auxiliary = self.auxiliary['process_tasks'][task.id]
        auxiliary['valid_computer'] = {}

        def valid_computer(c: dict):
            if task.computer is not None and task.computer != c['name']:
                auxiliary['valid_computer'][c['name']] = \
                    'name set in the config!= name of this computer'
                return False

            if task.cpu > c['cpu']:
                auxiliary['valid_computer'][c['name']] = \
                    f'task cpu = {task.cpu} > computer free cpu = {c["cpu"]}'
                return False

            if task.memory > c['memory']:
                auxiliary['valid_computer'][c['name']] = \
                    f'task cpu = {task.cpu} > computer ' \
                    f'free memory = {c["memory"]}'
                return False

            queue = f'{c["name"]}_' \
                    f'{task.dag_rel.docker_img or "default"}'
            if queue not in self.queues:
                auxiliary['valid_computer'][c['name']] = \
                    f'required queue = {queue} not in queues'

                return False
            if task.gpu > 0 and sum(c['gpu']) == 0:
                auxiliary['valid_computer'][c['name']] = \
                    f'task requires gpu, but there is not any free'

                return False

            auxiliary['valid_computer'][c['name']] = 'valid'
            return True

        computers = [c for c in self.computers if valid_computer(c)]
        free_gpu = sum(sum(c['gpu'] == 0) for c in computers)
        if task.gpu > free_gpu:
            auxiliary['valid'] = f'gpu required by the task = {task.gpu},' \
                                 f' but there are only {free_gpu} free gpus'
            return

        to_send = []
        computer_gpu_taken = defaultdict(list)
        for computer in self.computers:
            queue = f'{computer["name"]}_' \
                    f'{task.dag_rel.docker_img or "default"}'

            if task.gpu > 0:
                for i in list(np.where(computer['gpu'] == 0)[0]):
                    to_send.append([computer, queue, i])
                    computer_gpu_taken[computer['name']].append(int(i))

            else:
                self.process_to_celery(task, queue, computer)
                break

        auxiliary['to_send'] = to_send

        rank = 0
        master_port = None
        if len(to_send) > 0:
            master_port = self.find_port(to_send[0][0], to_send[0][1])

        for computer, queue, gpu_assigned in to_send:
            gpu_visible = computer_gpu_taken[computer['name']]
            main_cmp = to_send[0][0]
            # noinspection PyTypeChecker
            ip = 'localhost' if computer['name'] == main_cmp['name'] \
                else main_cmp['ip']

            distr_info = {
                'master_addr': ip,
                'rank': rank,
                'local_rank': gpu_visible.index(gpu_assigned),
                'gpu_visible': gpu_visible,
                'master_port': master_port
            }
            service_task = self.create_service_task(task,
                                                    distr_info=distr_info,
                                                    gpu_assigned=gpu_assigned)
            self.process_to_celery(service_task,
                                   queue,
                                   computer)
            rank += 1

        if len(to_send) > 0:
            task.status = TaskStatus.Queued.value
            self.provider.commit()

    def process_tasks(self):
        self.auxiliary['process_tasks'] = {}

        for task in self.not_ran_tasks:
            auxiliary = {}
            self.auxiliary['process_tasks'][task.id] = auxiliary

            if task.dag_rel is None:
                auxiliary['not_valid'] = 'no dag_rel'
                continue

            if TaskStatus.Stopped.value in self.dep_status[task.id] \
                    or TaskStatus.Failed.value in self.dep_status[task.id]:
                auxiliary['not_valid'] = 'stopped or failed in dep_status'
                self.provider.change_status(task, TaskStatus.Skipped)
                continue

            status_set = set(self.dep_status[task.id])
            if len(status_set) != 0 \
                    and status_set != {TaskStatus.Success.value}:
                auxiliary['not_valid'] = 'not all dep tasks are finished'
                continue
            self.process_task(task)

    def process_parent_tasks(self):
        tasks = self.provider.parent_tasks_stats()

        was_change = False
        for task, started, finished, statuses in tasks:
            status = task.status
            if statuses[TaskStatus.Failed] > 0:
                status = TaskStatus.Failed.value
            elif statuses[TaskStatus.Skipped] > 0:
                status = TaskStatus.Skipped.value
            elif statuses[TaskStatus.Queued] > 0:
                status = TaskStatus.Queued.value
            elif statuses[TaskStatus.InProgress] > 0:
                status = TaskStatus.InProgress.value
            elif statuses[TaskStatus.Success] > 0:
                status = TaskStatus.Success.value

            if status != task.status:
                was_change = True
                task.status = status
                if status == TaskStatus.InProgress.value:
                    task.started = started
                elif status >= TaskStatus.Failed.value:
                    task.started = started
                    task.finished = finished

        if was_change:
            self.provider.commit()

        self.auxiliary['parent_tasks_stats'] = [
            {
                'name': task.name,
                'id': task.id,
                'started': task.started,
                'finished': finished,
                'statuses': statuses,
            }
            for task, started, finished, statuses in tasks
        ]

    def write_auxiliary(self):
        self.auxiliary['duration'] = (now() - self.auxiliary['time']). \
            total_seconds()

        auxiliary = Auxiliary(name='supervisor',
                              data=yaml_dump(self.auxiliary)
                              )
        self.auxiliary_provider.create_or_update(auxiliary, 'name')

    def build(self):
        try:
            self.auxiliary = {'time': now()}

            self.create_base()

            self.process_parent_tasks()

            self.load_tasks()

            self.load_computers()

            self.process_tasks()

            self.write_auxiliary()

        except Exception as error:
            if type(error) == ProgrammingError:
                Session.cleanup()

            self.logger.error(traceback.format_exc(),
                              ComponentType.Supervisor)


def register_supervisor():
    builder = SupervisorBuilder()
    start_schedule([(builder.build, 1)])
