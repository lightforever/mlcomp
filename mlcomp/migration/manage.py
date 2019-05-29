from migrate.versioning.shell import main
from migrate.exceptions import DatabaseAlreadyControlledError
import os
from mlcomp.db.conf import SA_CONNECTION_STRING

if __name__ == '__main__':
    try:
        folder = os.path.dirname(__file__)
        main(repository='.', url=SA_CONNECTION_STRING, debug='False')
    except DatabaseAlreadyControlledError:
        pass
