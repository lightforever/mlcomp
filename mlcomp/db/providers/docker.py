from mlcomp.db.models import Docker
from mlcomp.db.providers.base import BaseDataProvider


class DockerProvider(BaseDataProvider):
    model = Docker

    def get(self, computer: str, name: str):
        return self.query(Docker).\
            filter(Docker.computer == computer).\
            filter(Docker.name == name).one()


__all__ = ['DockerProvider']