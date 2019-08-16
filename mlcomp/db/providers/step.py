from sqlalchemy import func, case

from mlcomp.db.enums import LogStatus
from mlcomp.db.models import Step, Log
from mlcomp.db.providers.base import BaseDataProvider
from mlcomp.utils.misc import to_snake, now


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
        duration = ((step.finished if step.finished else now()) - step.started)
        res = {
            'id': step.id,
            'name': step.name,
            'level': step.level,
            'duration': duration.total_seconds(),
            'log_statuses': [
                {
                    'name': to_snake(e.name),
                    'count': s
                } for e, s in zip(LogStatus, log_status)
            ]
        }
        return res

    def get(self, task_id: int):
        log_status = []
        for s in LogStatus:
            log_status.append(
                func.sum(
                    case(
                        whens=[(Log.level == s.value, 1)],
                        else_=0
                    ).label(s.name)
                )
            )

        query = self.query(Step, *log_status).filter(Step.task == task_id
                                                     ).order_by(Step.started)
        query = query.join(Log, isouter=True).group_by(Step.id)
        steps = query.all()
        if len(steps) == 0:
            return []
        d = self.step_info(steps[0]) if len(steps) > 0 else dict()

        hierarchy = {**d, 'children': []}
        self._hierarchy(hierarchy, steps, 1, len(steps) - 1)
        return {'data': [hierarchy]}

    def last_for_task(self, id: int):
        return self.query(Step).filter(Step.task == id
                                       ).order_by(Step.started.desc()).first()

    def unfinished(self, task_id: int):
        return self.query(Step).filter(Step.task == task_id
                                       ).filter(Step.finished.__eq__(None)
                                                ).order_by(Step.started).all()


__all__ = ['StepProvider']
