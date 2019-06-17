from mlcomp.db.providers.base import *


class DockerProvider(BaseDataProvider):
    model = Docker

    def get(self, computer: str, name: str):
        return self.query(Docker).filter(Docker.computer == computer).filter(Docker.name == name).one()
