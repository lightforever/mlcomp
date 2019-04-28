from mlcomp.db.providers.base import *

class ComputerProvider(BaseDataProvider):
    def computers(self):
        return self.query(Computer).all()