# flake8: noqa
# noinspection PyUnresolvedReferences
from mlcomp.utils.tests import session
from mlcomp.db.core import Session
from mlcomp.db.providers import ProjectProvider


class TestProject(object):

    def _configure(self, session):
        provider = ProjectProvider(session)
        provider.add_project(name='test')
        return provider

    def test_add(self, session: Session):
        provider = ProjectProvider(session)
        project = provider.add_project(name='test')
        assert provider.by_id(project.id)

    def test_by_name(self, session: Session):
        provider = self._configure(session)
        project = provider.by_name('test')
        assert project is not None

    def test_get(self, session: Session):
        provider = self._configure(session)
        res = provider.get()
        assert len(res['data']) == 1
        assert res['total'] == 1
