from mlcomp.db.providers.base import *

class FileProvider(BaseDataProvider):
    model = File

    def hashs(self, project: int):
        return {obj[0]:obj[1] for obj in self.query(File.md5, File.id).filter(File.project==project).all()}