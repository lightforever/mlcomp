from collections import defaultdict

from sqlalchemy import and_

from mlcomp.db.enums import TaskStatus, TaskType
from mlcomp.db.models import TaskSynced, Task, Dag, Project, Computer
from mlcomp.db.providers.base import BaseDataProvider


class TaskSyncedProvider(BaseDataProvider):
    model = TaskSynced

    def for_computer(self, name: str):
        query = self.query(Task).filter(
            Task.status == TaskStatus.Success.value). \
            filter(Task.type <= TaskType.Train.value). \
            filter(Task.computer_assigned.__ne__(None)). \
            join(TaskSynced, and_(TaskSynced.task == Task.id,
                                  TaskSynced.computer == name), isouter=True).\
            filter(TaskSynced.task.__eq__(None))

        res = []
        by_computer_project = defaultdict(list)
        for task in query.all():
            if task.computer_assigned == name:
                continue
            dag = self.query(Dag).filter(Dag.id == task.dag).one()
            by_computer_project[(task.computer_assigned, dag.project)].append(
                task)

        for (c, p), tasks in by_computer_project.items():
            c = self.query(Computer).filter(Computer.name == c).one()
            p = self.query(Project).filter(Project.id == p).one()

            res.append((c, p, tasks))
        return res


__all__ = ['TaskSyncedProvider']
