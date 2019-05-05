import json

from mlcomp.db.providers.base import *
import datetime


class ComputerProvider(BaseDataProvider):
    model = Computer

    def computers(self):
        return {c.name: {k: v for k, v in c.__dict__.items()} for c in self.query(Computer).all()}

    def computers_list(self):
        return self.query(Computer).all()

    def computer_usage(self, computer: str):
        min_time = now() - datetime.timedelta(days=3)
        query = self.query(ComputerUsage.usage).filter(ComputerUsage.time >= min_time).filter(
            ComputerUsage.computer == computer).order_by(ComputerUsage.time)
        return query.all()

    def current_usage(self, name: str, usage: dict):
        computer = self.query(Computer).filter(Computer.name==name).first()
        computer.usage = json.dumps(usage)
        self.update()


