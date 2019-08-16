import os

import migrate.versioning.api as api
from migrate.exceptions import DatabaseAlreadyControlledError

from mlcomp import SA_CONNECTION_STRING


def migrate(connection_string: str = None):
    folder = os.path.dirname(__file__)
    connection_string = connection_string or SA_CONNECTION_STRING
    try:
        api.version_control(url=connection_string, repository=folder)
    except DatabaseAlreadyControlledError:
        pass

    api.upgrade(url=connection_string, repository=folder)


__all__ = ['migrate']
