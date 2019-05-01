from mlcomp.db.providers.base import *

class ComputerProvider(BaseDataProvider):
    model = Computer

    def computers(self):
        return {c.name: {k: v for k, v in c.__dict__.items()} for c in self.query(Computer).all()}