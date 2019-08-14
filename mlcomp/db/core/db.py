import sqlalchemy as sa
import sqlalchemy.orm.session as session
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import event

from mlcomp.db.conf import SA_CONNECTION_STRING, DB_TYPE
from mlcomp.utils.misc import adapt_db_types


class Session(session.Session):
    __session = dict()

    def __init__(self, *args, **kwargs):
        key = kwargs.pop('key')
        if key in self.__session:
            raise Exception('Use static create_session for session creating')
        super().__init__(*args, **kwargs)

    @staticmethod
    def create_session(*, connection_string: str = None, key='default'):
        if key in Session.__session:
            return Session.__session[key][0]

        session_factory = scoped_session(sessionmaker(class_=Session, key=key))
        connect_args = {}
        if DB_TYPE == 'SQLITE':
            connect_args = {
                'check_same_thread': False,
                'timeout': 30
            }

        engine = sa.create_engine(
            connection_string or SA_CONNECTION_STRING,
            echo=False,
            connect_args=connect_args
        )
        if DB_TYPE == 'SQLITE':
            def _fk_pragma_on_connect(dbapi_con, con_record):
                dbapi_con.execute('pragma foreign_keys=ON')

            event.listen(engine, 'connect', _fk_pragma_on_connect)

        session_factory.configure(bind=engine)
        s = session_factory()

        Session.__session[key] = [s, engine]
        return s

    @classmethod
    def cleanup(cls, key: str):
        if key in cls.__session:
            s, engine = cls.__session[key]
            try:
                s.close()
            except Exception:
                pass

            try:
                engine.dispose()
            except Exception:
                pass

            del cls.__session[key]

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
        return obj

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

    @staticmethod
    def sqlalchemy_error(e):
        s = str(type(e))
        return 'sqlalchemy.' in s


__all__ = ['Session']
