from typing import List

from sqlalchemy.orm.query import Query
from sqlalchemy import desc
from sqlalchemy_serializer import Serializer
from sqlalchemy.orm import joinedload

from mlcomp.db.core import Session, PaginatorOptions
from mlcomp.db.models.base import Base


class BaseDataProvider:
    model = None

    date_format = '%Y-%m-%d'
    datetime_format = '%Y-%m-%d %H:%M:%SZ'
    time_format = '%H:%M'

    def __init__(self, session: Session = None):
        if session is None:
            session = Session.create_session()
        self._session = session
        self.serializer = Serializer(
            date_format=self.date_format,
            datetime_format=self.datetime_format,
            time_format=self.time_format
        )

    def serialize_datetime(self, value):
        return self.serializer.serialize_datetime(value)

    def remove(self, key_value, key_column: str = 'id'):
        self.query(self.model). \
            filter(getattr(self.model, key_column) == key_value). \
            delete(synchronize_session=False)
        self.session.commit()

    def detach(self, obj):
        self.session.expunge(obj)

    @property
    def query(self):
        return self.session.query

    def add_all(self, obs: List[Base], commit=True):
        self._session.add_all(obs, commit=commit)

    def add(self, obj: Base, commit=True):
        self._session.add(obj, commit=commit)
        return obj

    def by_id(self, id: int, joined_load=None):
        res = self.query(self.model).filter(getattr(self.model, 'id') == id)
        if joined_load is not None:
            for n in joined_load:
                res = res.options(joinedload(n, innerjoin=True))
        return res.first()

    def to_dict(self, item, rules=(), datetime_format=None):
        datetime_format = datetime_format or self.datetime_format
        return item.to_dict(
            date_format=self.date_format,
            datetime_format=datetime_format,
            time_format=self.time_format,
            rules=rules
        )

    def create_or_update(self, obj: Base, *fields):
        query = self.session.query(obj.__class__)
        for k in fields:
            query = query.filter(getattr(obj.__class__, k) == getattr(obj, k))

        db = query.first()
        if db is not None:
            for field, value in obj.__dict__.items():
                if not field.startswith('_'):
                    setattr(db, field, value)
            self.session.update()
        else:
            self.add(obj)

    def all(self):
        return self.query(self.model).all()

    def update(self):
        self.session.update()
        self.session.commit()

    def commit(self):
        self.session.commit()

    def rollback(self):
        self.session.rollback()

    @property
    def session(self):
        return self._session

    def paginator(self, query: Query, options: PaginatorOptions):
        if options is None:
            return query

        if options.sort_column:
            column = getattr(self.model, options.sort_column) if \
                options.sort_column in self.model.__dict__ \
                else options.sort_column
            criterion = column if not options.sort_descending else desc(column)
            query = query.order_by(criterion)

        if options.page_size:
            query = query. \
                offset(options.page_size * options.page_number). \
                limit(options.page_size)

        return query


__all__ = ['BaseDataProvider']
