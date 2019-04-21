from db.core import *
from db.models import *
from sqlalchemy.orm.query import Query
from sqlalchemy import desc

class BaseDataProvider:
    def __init__(self, session=None):
        if session is None:
            session = Session()
        self._session = session
        self.handler = Handler(session)

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
            criterion = options.sort_column if not options.sort_descending else desc(options.sort_descending)
            query = query.order_by(criterion)

        return query