import datetime

from mlcomp.db.models import Docker
from mlcomp.db.providers.base import BaseDataProvider
from mlcomp.utils.misc import now


class DockerProvider(BaseDataProvider):
    model = Docker

    def get(self, computer: str, name: str):
        return self.query(Docker). \
            filter(Docker.computer == computer). \
            filter(Docker.name == name).one()

    def get_online(self):
        min_activity = now() - datetime.timedelta(seconds=30)
        return self.query(Docker).\
            filter(Docker.last_activity >= min_activity).\
            all()


__all__ = ['DockerProvider']
