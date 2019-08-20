import pytest
from importlib import reload
import shutil

import mlcomp
from mlcomp import ROOT_FOLDER
from mlcomp.db.core import Session
from mlcomp.migration.manage import migrate


@pytest.fixture()
def session():
    if ROOT_FOLDER:
        shutil.rmtree(ROOT_FOLDER)
        reload(mlcomp)

    migrate()
    res = Session.create_session()
    yield res
