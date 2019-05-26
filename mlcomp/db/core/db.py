import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, scoped_session

from mlcomp.db.conf import *
import logging
import sqlalchemy.orm.session as session

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

__all__ = ['Session']

logger = logging.getLogger(__name__)

class Session(session.Session):
    __session = dict()

    def __init__(self, *args, **kwargs):
        key = kwargs.pop('key')
        if key in self.__session:
            raise Exception('Use static create_session for session creating')
        super(Session, self).__init__(*args, **kwargs)


    @staticmethod
    def create_session(connection_string:str=None, key='default'):
        if key in Session.__session:
            return Session.__session[key][0]

        session_factory = scoped_session(sessionmaker(class_= Session, key=key))
        engine = sa.create_engine(connection_string or SA_CONNECTION_STRING, echo=False)
        session_factory.configure(bind=engine)
        session = session_factory()

        Session.__session[key] = [session, engine]
        return session

    @classmethod
    def cleanup(cls):
        for k, (session, engine) in cls.__session.items():
            try:
                session.close()
            except Exception:
                pass
            try:
                engine.dispose()
            except Exception:
                pass

        cls.__session = dict()

    def query(self, *entities, **kwargs):
        try:
            return super(Session, self).query(*entities, **kwargs)
        except Exception as e:
            logger.error('====Query ERROR====')
            logger.error(e)
            self.rollback()
            raise e

    def add(self, obj):
        try:
            super(Session, self).add(obj)
        except Exception as e:
            logger.error('====ADD ERROR====')
            logger.error(e)
            raise e
        try:
            self.commit()
        except Exception as e:
            logger.error('====COMMIT ERROR====')
            logger.error(e)
            self.rollback()
            raise e

    def update(self):
        try:
            self.commit()
        except Exception as e:
            logger.error('====COMMIT ERROR====')
            logger.error(e)
            self.rollback()
            raise e