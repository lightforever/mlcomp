from mlcomp.db.providers.base import *

class TaskStorageProvider(BaseDataProvider):
    def by_task(self, task:int):
        return self.query(TaskStorage, File).join(File, isouter=True).filter(TaskStorage.task==task).all()