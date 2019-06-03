from sqlalchemy.orm import joinedload

from mlcomp.db.core import *
from mlcomp.db.models import *
from sqlalchemy.orm.query import Query
from sqlalchemy import desc
from mlcomp.utils.misc import now
from sqlalchemy.orm.attributes import flag_modified, set_attribute
from sqlalchemy import func
from sqlalchemy_serializer import Serializer

class BaseDataProvider:
    model = None

    date_format = '%Y-%m-%d'
    datetime_format = '%Y-%m-%d %H:%MZ'
    time_format = '%H:%M'

    def __init__(self, session=None):
        if session is None:
            session = Session.create_session()
        self._session = session
        self.serializer = Serializer(date_format=self.date_format, datetime_format=self.datetime_format, time_format=self.time_format)

    def serialize_datetime(self, value):
        return self.serializer.serialize_datetime(value)

    def remove(self, id: int):
        self.query(self.model).filter(getattr(self.model, 'id') == id).delete(synchronize_session=False)
        self.session.commit()

    def detach(self, obj):
        self.session.expunge(obj)

    @property
    def query(self):
        return self.session.query

    def add(self, obj: Base, commit=True):
        self._session.add(obj, commit=commit)
        return obj

    def by_id(self, id:int, joined_load=None):
        res = self.query(self.model).filter(getattr(self.model, 'id') == id)
        if joined_load is not None:
            for n in joined_load:
                res=res.options(joinedload(n))
        return res.first()

    def to_dict(self, item, rules=(), datetime_format=None):
        return item.to_dict(date_format=self.date_format, datetime_format=datetime_format or self.datetime_format,
                            time_format=self.time_format, rules=rules)

    def create_or_update(self, obj: Base, field: str):
        db = self.session.query(obj.__class__).filter(getattr(obj.__class__, field)==getattr(obj, field)).first()
        if db is not None:
            for field, value in obj.__dict__.items():
                if not field.startswith('_'):
                    setattr(db, field, value)
            self.session.update()
        else:
            self.add(obj)

    def update(self):
        self.session.update()
        self.session.commit()

    def commit(self):
        self.session.commit()

    @property
    def session(self):
        return self._session

    def paginator(self, query: Query, options: PaginatorOptions):
        if options.sort_column:
            column = getattr(self.model, options.sort_column) if \
                options.sort_column in self.model.__dict__ else options.sort_column
            criterion = column if not options.sort_descending else desc(column)
            query = query.order_by(criterion)

        if options.page_size:
            query = query.offset(options.page_size*options.page_number).limit(options.page_size)

        return query





def set_attribute_modified(instance, key, value):
    set_attribute(instance, key, value)
    flag_modified(instance, key)