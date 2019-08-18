import pytest
import os
from uuid import uuid4
from importlib import reload
import shutil

import mlcomp

os.environ['ROOT_FOLDER'] = os.path.join(f'/tmp/mlcomp/{uuid4()}')
if os.path.exists(os.environ['ROOT_FOLDER']):
    shutil.rmtree(os.environ['ROOT_FOLDER'])

reload(mlcomp)

# flake8: noqa
from mlcomp.db.core import Session
from mlcomp.migration.manage import migrate


@pytest.fixture()
def session(tmpdir: str):
    migrate()
    res = Session.create_session()
    yield res
