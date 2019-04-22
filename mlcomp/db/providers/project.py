from mlcomp.db.providers.base import *

class ProjectProvider(BaseDataProvider):
    def add(self, name: str):
        project = Project(name=name)
        self.handler.insert(project)

    def get(self, options: PaginatorOptions):
        try:
            query = self.session.query(Project)
            return self.paginator(query, options).all()
        except Exception:
            self.session.rollback()