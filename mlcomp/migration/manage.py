import os

import migrate.versioning.api as api
from migrate.exceptions import DatabaseAlreadyControlledError

from mlcomp.db.conf import SA_CONNECTION_STRING


def migrate():
    folder = os.path.dirname(__file__)

    try:
        api.version_control(url=SA_CONNECTION_STRING, repository=folder)
    except DatabaseAlreadyControlledError:
        pass

    api.upgrade(url=SA_CONNECTION_STRING, repository=folder)


__all__ = ['migrate']
