import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from db.conf import *

__all__ = ['Session']


class Session:
    __session = None
    __engine = None

    @staticmethod
    def __new__(cls, *args, **kwargs):
        if cls.__session is None:
            cls.__engine, cls.__session = cls.create_session()
        return cls.__session

    @staticmethod
    def create_session(connection_string:str=None):
        session_factory = sessionmaker()
        engine = sa.create_engine(connection_string or SA_CONNECTION_STRING, echo=False)
        session_factory.configure(bind=engine)
        session = session_factory()
        return engine, session

    @classmethod
    def cleanup(cls):
        if cls.__session is not None:
            cls.__session.close()
            cls.__session = None
            cls.__engine.dispose()
            cls.__engine = None
