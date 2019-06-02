from mlcomp.db.enums import StepStatus, LogStatus
from mlcomp.db.providers.base import *
from mlcomp.utils.misc import to_snake


class StepProvider(BaseDataProvider):
    model = Step

    def _hierarchy(self, parent: dict, steps: list, start: int, end: int):
        current = start
        while current <= end:
            s = steps[current][0]
            if s.level == parent['level'] + 1:
                child = {**self.step_info(steps[current]), 'children': []}
                parent['children'].append(child)
                for i in range(current + 1, end + 1):
                    if steps[i][0].level <= s.level:
                        self._hierarchy(child, steps, current + 1, i - 1)
                        current = i
                        break
                    if i == len(steps) - 1:
                        self._hierarchy(child, steps, current + 1, i)
                else:
                    break

    def step_info(self, step):
        step, *log_status = step
        res = {'id': step.id, 'name': step.name,
               'status': to_snake(StepStatus(step.status).name),
               'level': step.level,
               'duration': ((step.finished if step.finished else now()) - step.started).total_seconds(),
               'log_statuses': [{'name': to_snake(e.name), 'count': s} for e, s in zip(LogStatus, log_status)]
               }
        return res

    def get(self, task_id: int):
        log_status = []
        for s in LogStatus:
            log_status.append(func.count(Log.level).filter(Log.level == s.value).label(s.name))

        query = self.query(Step, *log_status).filter(Step.task == task_id).order_by(Step.started)
        query = query.join(Log).group_by(Step.id)
        steps = query.all()
        hierarchy = {**self.step_info(steps[0]), 'children': []}
        self._hierarchy(hierarchy, steps, 1, len(steps) - 1)
        return [hierarchy]

    def last_for_task(self, id: int):
        return self.query(Step).filter(Step.task == id).order_by(Step.started.desc()).first()
