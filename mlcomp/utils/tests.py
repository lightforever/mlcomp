import pytest
import os
from uuid import uuid4
from importlib import reload

import mlcomp

os.environ['ROOT_FOLDER'] = os.path.join(f'/tmp/mlcomp/{uuid4()}')
reload(mlcomp)

# flake8: noqa
from mlcomp import DB_FOLDER
from mlcomp.db.core import Session
from mlcomp.migration.manage import migrate


@pytest.fixture()
def session(tmpdir: str):
    path = f'{DB_FOLDER}/sqlite3.sqlite'
    if os.path.exists(path):
        os.remove(path)

    migrate()
    res = Session.create_session()
    yield res
