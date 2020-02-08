import datetime
import traceback
from typing import List

from sqlalchemy.orm.exc import ObjectDeletedError

from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType, TaskStatus, TaskType
from mlcomp.db.models import Task, Auxiliary
from mlcomp.db.providers import \
    ComputerProvider, \
    TaskProvider, \
    DockerProvider, \
    AuxiliaryProvider, DagProvider
from mlcomp.utils.io import yaml_dump, yaml_load
from mlcomp.utils.logging import create_logger
from mlcomp.utils.misc import now
from mlcomp.worker.tasks import execute
from mlcomp.utils.schedule import start_schedule
import mlcomp.worker.tasks as celery_tasks


class SupervisorBuilder:
    def __init__(self):
        self.session = Session.create_session(key='SupervisorBuilder')
        self.logger = create_logger(self.session, 'SupervisorBuilder')
        self.provider = None
        self.computer_provider = None
        self.docker_provider = None
        self.auxiliary_provider = None
        self.dag_provider = None
        self.queues = None
        self.not_ran_tasks = None
        self.dep_status = None
        self.computers = None
        self.auxiliary = {}

        self.tasks = []
        self.tasks_stop = []
        self.dags_start = []
        self.sent_tasks = 0

    def create_base(self):
        self.session.commit()

        self.provider = TaskProvider(self.session)
        self.computer_provider = ComputerProvider(self.session)
        self.docker_provider = DockerProvider(self.session)
        self.auxiliary_provider = AuxiliaryProvider(self.session)
        self.dag_provider = DagProvider(self.session)

        self.queues = [
            f'{d.computer}_{d.name}' for d in self.docker_provider.all()
            if d.last_activity >= now() - datetime.timedelta(seconds=15)
        ]

        self.auxiliary['queues'] = self.queues

    def load_tasks(self):
        self.tasks = self.provider.by_status(TaskStatus.NotRan,
                                             TaskStatus.InProgress,
                                             TaskStatus.Queued)

        not_ran_tasks = [t for t in self.tasks if
                         t.status == TaskStatus.NotRan.value]

        self.not_ran_tasks = [task for task in not_ran_tasks if not task.debug]
        self.not_ran_tasks = sorted(
            self.not_ran_tasks, key=lambda x: x.gpu or 0,
            reverse=True)

        self.logger.debug(
            f'Found {len(not_ran_tasks)} not ran tasks',
            ComponentType.Supervisor
        )

        self.dep_status = self.provider.dependency_status(self.not_ran_tasks)

        self.auxiliary['not_ran_tasks'] = [
            {
                'id': t.id,
                'name': t.name,
                'dep_status': [
                    TaskStatus(s).name
                    for s in self.dep_status.get(t.id, set())
                ]
            } for t in not_ran_tasks[:5]
        ]

    def load_computers(self):
        computers = self.computer_provider.computers()
        for computer in computers.values():
            computer['gpu'] = [0] * computer['gpu']
            computer['ports'] = set()
            computer['cpu_total'] = computer['cpu']
            computer['memory_total'] = computer['memory']
            computer['gpu_total'] = len(computer['gpu'])
            computer['can_process_tasks'] = computer['can_process_tasks']

        tasks = [
            t for t in self.tasks if
            t.status in [TaskStatus.InProgress.value,
                         TaskStatus.Queued.value]
        ]

        for task in tasks:
            if task.computer_assigned is None:
                continue
            assigned = task.computer_assigned
            comp_assigned = computers[assigned]
            comp_assigned['cpu'] -= task.cpu

            if task.gpu_assigned is not None:
                for g in task.gpu_assigned.split(','):
                    comp_assigned['gpu'][int(g)] = task.id
            comp_assigned['memory'] -= task.memory * 1024

            info = yaml_load(task.additional_info)
            if 'distr_info' in info:
                dist_info = info['distr_info']
                if dist_info['rank'] == 0:
                    comp_assigned['ports'].add(dist_info['master_port'])

        self.computers = [
            {
                **value, 'name': name
            } for name, value in computers.items()
        ]

        self.auxiliary['computers'] = self.computers

    def process_to_celery(self, task: Task, queue: str, computer: dict):
        r = execute.apply_async((task.id,), queue=queue, retry=False)
        task.status = TaskStatus.Queued.value
        task.computer_assigned = computer['name']
        task.docker_assigned = queue
        task.celery_id = r.id

        if task.computer_assigned is not None:
            if task.gpu_assigned:
                for g in map(int, task.gpu_assigned.split(',')):
                    computer['gpu'][g] = task.id
            computer['cpu'] -= task.cpu
            computer['memory'] -= task.memory * 1024

        self.logger.info(
            f'Sent task={task.id} to celery. Queue = {queue} '
            f'Task status = {task.status} Celery_id = {r.id}',
            ComponentType.Supervisor)
        self.provider.update()

    def create_service_task(
            self,
            task: Task,
            gpu_assigned=None,
            distr_info: dict = None,
            resume: dict = None
    ):
        new_task = Task(
            name=task.name,
            computer=task.computer,
            executor=task.executor,
            status=TaskStatus.NotRan.value,
            type=TaskType.Service.value,
            gpu_assigned=gpu_assigned,
            parent=task.id,
            report=task.report,
            dag=task.dag
        )
        new_task.additional_info = task.additional_info

        if distr_info:
            additional_info = yaml_load(new_task.additional_info)
            additional_info['distr_info'] = distr_info
            new_task.additional_info = yaml_dump(additional_info)

        if resume:
            additional_info = yaml_load(new_task.additional_info)
            additional_info['resume'] = resume
            new_task.additional_info = yaml_dump(additional_info)

        return self.provider.add(new_task)

    def find_port(self, c: dict, docker_name: str):
        docker = self.docker_provider.get(c['name'], docker_name)
        ports = list(map(int, docker.ports.split('-')))
        for p in range(ports[0], ports[1] + 1):
            if p not in c['ports']:
                return p
        raise Exception(f'All ports in {c["name"]} are taken')

    def _process_task_valid_computer(self, task: Task, c: dict,
                                     single_node: bool):
        if not c['can_process_tasks']:
            return 'this computer can not process tasks'

        if task.computer is not None and task.computer != c['name']:
            return 'name set in the config!= name of this computer'

        if task.cpu > c['cpu']:
            return f'task cpu = {task.cpu} > computer' \
                   f' free cpu = {c["cpu"]}'

        if task.memory > c['memory']:
            return f'task cpu = {task.cpu} > computer ' \
                   f'free memory = {c["memory"]}'

        queue = f'{c["name"]}_' \
                f'{task.dag_rel.docker_img or "default"}'
        if queue not in self.queues:
            return f'required queue = {queue} not in queues'

        if task.gpu > 0 and not any(g == 0 for g in c['gpu']):
            return f'task requires gpu, but there is not any free'

        free_gpu = sum(g == 0 for g in c['gpu'])
        if single_node and task.gpu > free_gpu:
            return f'task requires {task.gpu} ' \
                   f'but there are only {free_gpu} free'

    def _process_task_get_computers(
            self, executor: dict, task: Task, auxiliary: dict
    ):
        single_node = executor.get('single_node', True)

        computers = []
        for c in self.computers:
            error = self._process_task_valid_computer(task, c, single_node)
            auxiliary['computers'].append({'name': c['name'], 'error': error})
            if not error:
                computers.append(c)

        if task.gpu > 0 and single_node and len(computers) > 0:
            computers = sorted(
                computers,
                key=lambda x: sum(g == 0 for g in c['gpu']),
                reverse=True
            )[:1]

        free_gpu = sum(sum(g == 0 for g in c['gpu']) for c in computers)
        if task.gpu > free_gpu:
            auxiliary['not_valid'] = f'gpu required by the ' \
                                     f'task = {task.gpu},' \
                                     f' but there are only {free_gpu} ' \
                                     f'free gpus'
            return []
        return computers

    def _process_task_to_send(
            self, executor: dict, task: Task, computers: List[dict]
    ):
        distr = executor.get('distr', True)
        to_send = []
        for computer in computers:
            queue = f'{computer["name"]}_' \
                    f'{task.dag_rel.docker_img or "default"}'

            if task.gpu_max > 1 and distr:
                for index, task_taken_gpu in enumerate(computer['gpu']):
                    if task_taken_gpu:
                        continue
                    to_send.append([computer, queue, index])

                    if len(to_send) >= task.gpu_max:
                        break

                if len(to_send) >= task.gpu_max:
                    break
            elif task.gpu_max > 0:
                cuda_devices = []
                for index, task_taken_gpu in enumerate(computer['gpu']):
                    if task_taken_gpu:
                        continue

                    cuda_devices.append(index)
                    if len(cuda_devices) >= task.gpu_max:
                        break

                task.gpu_assigned = ','.join(map(str, cuda_devices))
                self.process_to_celery(task, queue, computer)
            else:
                self.process_to_celery(task, queue, computer)
                break
        return to_send

    def process_task(self, task: Task):
        auxiliary = self.auxiliary['process_tasks'][-1]
        auxiliary['computers'] = []

        config = yaml_load(task.dag_rel.config)
        executor = config['executors'][task.executor]

        computers = self._process_task_get_computers(executor, task, auxiliary)
        if len(computers) == 0:
            return

        to_send = self._process_task_to_send(executor, task, computers)
        auxiliary['to_send'] = to_send[:5]
        additional_info = yaml_load(task.additional_info)

        rank = 0
        master_port = None
        if len(to_send) > 0:

            master_port = self.find_port(
                to_send[0][0], to_send[0][1].split('_')[1]
            )
            computer_names = {c['name'] for c, _, __ in to_send}
            if len(computer_names) == 1:
                task.computer_assigned = list(computer_names)[0]

        for computer, queue, gpu_assigned in to_send:
            main_cmp = to_send[0][0]
            # noinspection PyTypeChecker
            ip = 'localhost' if computer['name'] == main_cmp['name'] \
                else main_cmp['ip']

            distr_info = {
                'master_addr': ip,
                'rank': rank,
                'local_rank': gpu_assigned,
                'master_port': master_port,
                'world_size': len(to_send),
                'master_computer': main_cmp['name']
            }
            service_task = self.create_service_task(
                task,
                distr_info=distr_info,
                gpu_assigned=gpu_assigned,
                resume=additional_info.get('resume')
            )
            self.process_to_celery(service_task, queue, computer)
            rank += 1
            main_cmp['ports'].add(master_port)

        if len(to_send) > 0:
            task.status = TaskStatus.Queued.value
            self.sent_tasks += len(to_send)

    def process_tasks(self):
        self.auxiliary['process_tasks'] = []

        for task in self.not_ran_tasks:
            auxiliary = {'id': task.id, 'name': task.name}
            self.auxiliary['process_tasks'].append(auxiliary)

            if task.dag_rel is None:
                task.dag_rel = self.dag_provider.by_id(task.dag)

            if TaskStatus.Stopped.value in self.dep_status[task.id] \
                    or TaskStatus.Failed.value in self.dep_status[task.id] or \
                    TaskStatus.Skipped.value in self.dep_status[task.id]:
                auxiliary['not_valid'] = 'stopped or failed in dep_status'
                self.provider.change_status(task, TaskStatus.Skipped)
                continue

            if len(self.dep_status[task.id]) != 0 \
                    and self.dep_status[task.id] != {TaskStatus.Success.value}:
                auxiliary['not_valid'] = 'not all dep tasks are finished'
                continue
            self.process_task(task)

        self.auxiliary['process_tasks'] = self.auxiliary['process_tasks'][:5]

    def _stop_child_tasks(self, task: Task):
        self.provider.commit()

        children = self.provider.children(task.id, [Task.dag_rel])
        dags = [c.dag_rel for c in children]
        for c, d in zip(children, dags):
            celery_tasks.stop(self.logger, self.session, c, d)

    def _correct_catalyst_hangs(self, task, statuses):
        if task.type != TaskType.Train.value:
            return
        success = sum(s == TaskStatus.Success.value for s in statuses)
        in_progress = sum(s == TaskStatus.InProgress.value for s in statuses)

        if success + in_progress == len(statuses) and in_progress > 0:
            child_tasks = self.provider.children(task.id)
            for t in child_tasks:
                if t.status == TaskStatus.InProgress.value:
                    celery_tasks.kill.apply_async(
                        (t.pid,),
                        queue=t.docker_assigned,
                        retry=False)
                    t.status = TaskStatus.Success.value
            self.provider.commit()

    def process_parent_tasks(self):
        tasks = self.provider.parent_tasks_stats()

        was_change = False
        for task, started, finished, statuses in tasks:
            self._correct_catalyst_hangs(task, statuses)

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
                if status == TaskStatus.InProgress.value:
                    task.started = started
                elif status >= TaskStatus.Failed.value:
                    task.started = started
                    task.finished = finished
                    self._stop_child_tasks(task)

                was_change = True
                task.status = status

        if was_change:
            self.provider.commit()

        self.auxiliary['parent_tasks_stats'] = [
            {
                'name': task.name,
                'id': task.id,
                'started': task.started,
                'finished': finished,
                'statuses': [
                    {
                        'name': k.name,
                        'count': v
                    } for k, v in statuses.items()
                ],
            } for task, started, finished, statuses in tasks[:5]
        ]

    def write_auxiliary(self):
        self.auxiliary['duration'] = (now() - self.auxiliary['time']). \
            total_seconds()

        auxiliary = Auxiliary(
            name='supervisor', data=yaml_dump(self.auxiliary)
        )
        if len(auxiliary.data) > 16000:
            return

        self.auxiliary_provider.create_or_update(auxiliary, 'name')

    def stop_tasks(self, tasks: List[Task]):
        self.tasks_stop.extend([t.id for t in tasks])

    def process_stop_tasks(self):
        # Stop not running tasks
        if len(self.tasks_stop) == 0:
            return

        tasks = self.provider.by_ids(self.tasks_stop)
        tasks_not_ran = [t.id for t in tasks if
                         t.status in [TaskStatus.NotRan.value,
                                      TaskStatus.Queued.value]]
        tasks_started = [t for t in tasks if
                         t.status in [TaskStatus.InProgress.value]]
        tasks_started_ids = [t.id for t in tasks_started]

        self.provider.change_status_all(tasks=tasks_not_ran,
                                        status=TaskStatus.Skipped)

        pids = []
        for task in tasks_started:
            if task.pid:
                pids.append((task.computer_assigned, task.pid))

            additional_info = yaml_load(task.additional_info)
            for p in additional_info.get('child_processes', []):
                pids.append((task.computer_assigned, p))

        for computer, queue in self.docker_provider.queues_online():
            pids_computer = [p for c, p in pids if c == computer]
            if len(pids_computer) > 0:
                celery_tasks.kill_all.apply_async((pids_computer,),
                                                  queue=queue,
                                                  retry=False)

        self.provider.change_status_all(tasks=tasks_started_ids,
                                        status=TaskStatus.Stopped)

        self.tasks_stop = []

    def fast_check(self):
        if self.provider is None or self.computer_provider is None:
            return False

        if self.not_ran_tasks is None or self.queues is None:
            return False

        if len(self.tasks_stop) > 0:
            return False

        if len(self.dags_start) > 0:
            return False

        if len(self.auxiliary.get('to_send', [])) > 0:
            return False

        queues = set([
            f'{d.computer}_{d.name}' for d in self.docker_provider.all()
            if d.last_activity >= now() - datetime.timedelta(seconds=15)
        ])

        queues_set = set(queues)
        queues_set2 = set(self.queues)

        if queues_set != queues_set2:
            return False

        tasks = self.provider.by_status(TaskStatus.NotRan,
                                        TaskStatus.Queued,
                                        TaskStatus.InProgress)
        tasks_set = {t.id for t in tasks if
                     t.status == TaskStatus.NotRan.value and not t.debug}
        tasks_set2 = {t.id for t in self.tasks if
                      t.status == TaskStatus.NotRan.value}

        if tasks_set != tasks_set2:
            return False

        tasks_set = {t.id for t in tasks if
                     t.status == TaskStatus.InProgress.value}
        tasks_set2 = {t.id for t in self.tasks if
                      t.status == TaskStatus.InProgress.value}

        if tasks_set != tasks_set2:
            return False

        tasks_set = {t.id for t in tasks if
                     t.status == TaskStatus.Queued.value}
        tasks_set2 = {t.id for t in self.tasks if
                      t.status == TaskStatus.Queued.value}

        if tasks_set != tasks_set2:
            return False

        return True

    def start_dag(self, id: int):
        self.dags_start.append(id)

    def process_start_dags(self):
        if len(self.dags_start) == 0:
            return

        for id in self.dags_start:
            can_start_statuses = [
                TaskStatus.Failed.value, TaskStatus.Skipped.value,
                TaskStatus.Stopped.value
            ]

            tasks = self.provider.by_dag(id)
            children_all = self.provider.children([t.id for t in tasks])

            def find_resume(task):
                children = [c for c in children_all if c.parent == task.id]
                children = sorted(children, key=lambda x: x.id, reverse=True)

                if len(children) > 0:
                    for c in children:
                        if c.parent != task.id:
                            continue

                        info = yaml_load(c.additional_info)
                        if 'distr_info' not in info:
                            continue

                        if info['distr_info']['rank'] == 0:
                            return {
                                'master_computer': c.computer_assigned,
                                'master_task_id': c.id,
                                'load_last': True
                            }
                    raise Exception('Master task not found')
                else:
                    return {
                        'master_computer': task.computer_assigned,
                        'master_task_id': task.id,
                        'load_last': True
                    }

            for t in tasks:
                if t.status not in can_start_statuses:
                    continue

                if t.parent:
                    continue

                if t.type == TaskType.Train.value:
                    info = yaml_load(t.additional_info)
                    info['resume'] = find_resume(t)
                    t.additional_info = yaml_dump(info)

                t.status = TaskStatus.NotRan.value
                t.pid = None
                t.started = None
                t.finished = None
                t.computer_assigned = None
                t.celery_id = None
                t.worker_index = None
                t.docker_assigned = None

        self.provider.commit()
        self.dags_start = []

    def build(self):
        try:
            # if self.fast_check():
            #     return

            self.auxiliary = {'time': now()}

            self.create_base()

            self.process_stop_tasks()

            self.process_start_dags()

            self.process_parent_tasks()

            self.load_tasks()

            self.load_computers()

            self.process_tasks()

            self.write_auxiliary()

        except ObjectDeletedError:
            pass
        except Exception as e:
            if Session.sqlalchemy_error(e):
                Session.cleanup(key='SupervisorBuilder')
                self.session = Session.create_session(key='SupervisorBuilder')
                self.logger = create_logger(self.session, 'SupervisorBuilder')

            self.logger.error(traceback.format_exc(), ComponentType.Supervisor)


def register_supervisor():
    builder = SupervisorBuilder()
    start_schedule([(builder.build, 1)])
    return builder


__all__ = ['SupervisorBuilder', 'register_supervisor']
