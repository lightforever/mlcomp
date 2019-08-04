import sqlalchemy as sa
import sqlalchemy.orm.session as session
from sqlalchemy.orm import scoped_session, sessionmaker

from mlcomp.db.conf import SA_CONNECTION_STRING
from mlcomp.utils.misc import adapt_db_types


class Session(session.Session):
    __session = dict()

    def __init__(self, *args, **kwargs):
        key = kwargs.pop('key')
        if key in self.__session:
            raise Exception('Use static create_session for session creating')
        super().__init__(*args, **kwargs)

    @staticmethod
    def create_session(connection_string: str = None, key='default'):
        if key in Session.__session:
            return Session.__session[key][0]

        session_factory = scoped_session(sessionmaker(class_=Session, key=key))
        engine = sa.create_engine(
            connection_string or SA_CONNECTION_STRING, echo=False
        )
        session_factory.configure(bind=engine)
        s = session_factory()

        Session.__session[key] = [s, engine]
        return s

    @classmethod
    def cleanup(cls):
        for k, (s, engine) in cls.__session.items():
            try:
                s.close()
            except Exception:
                pass
            try:
                engine.dispose()
            except Exception:
                pass

        cls.__session = dict()

    def query(self, *entities, **kwargs):
        try:
            return super().query(*entities, **kwargs)
        except Exception as e:
            self.rollback()
            raise e

    def add_all(self, objs, commit=True):
        try:
            for obj in objs:
                adapt_db_types(obj)
            super().add_all(objs)
        except Exception as e:
            raise e

        if commit:
            try:
                self.commit()
            except Exception as e:
                self.rollback()
                raise e

    def add(self, obj, commit=True, _warn=False):
        try:
            adapt_db_types(obj)
            super().add(obj, _warn=_warn)
        except Exception as e:
            raise e

        if commit:
            try:
                self.commit()
            except Exception as e:
                self.rollback()
                raise e

    def commit(self):
        try:
            super().commit()
        except Exception as e:
            self.rollback()
            raise e

    def update(self):
        try:
            self.commit()
        except Exception as e:
            self.rollback()
            raise e


__all__ = ['Session']
