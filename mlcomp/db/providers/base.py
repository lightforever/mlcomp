from mlcomp.db.core import *
from mlcomp.db.models import *
from sqlalchemy.orm.query import Query
from sqlalchemy import desc

class BaseDataProvider:
    def __init__(self, session=None):
        if session is None:
            session = Session.create_session()
        self._session = session

    @property
    def query(self):
        return self.session.query

    def add(self, obj: Base):
        self._session.add(obj)

    @property
    def session(self):
        return self._session

    def __del__(self):
        self._session.close()
        self._session = None

    def paginator(self, query: Query, options: PaginatorOptions):
        if options.page_size:
            query = query.offset(options.page_size*options.page_number)

        if options.sort_descending:
            criterion = options.sort_column if not options.sort_column else desc(options.sort_column)
            query = query.order_by(criterion)

        return query