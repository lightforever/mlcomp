from mlcomp.db.providers.base import *

class ProjectProvider(BaseDataProvider):
    def add(self, name: str):
        project = Project(name=name)
        self.session.add(project)

    def get(self, options: PaginatorOptions):
        query = self.query(Project)
        return self.paginator(query, options).all()
