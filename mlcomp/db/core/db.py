import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from mlcomp.db.conf import *
import logging
import sqlalchemy.orm.session as session

__all__ = ['Session']

logger = logging.getLogger(__name__)

class Session(session.Session):
    __session = None
    __engine = None

    def __init__(self, *args, **kwargs):
        if self.__session is not None:
            raise Exception('Use static create_session for session creating')
        super(Session, self).__init__(*args, **kwargs)

    @staticmethod
    def create_session(connection_string:str=None):
        if Session.__session is not None:
            return Session.__session

        session_factory = sessionmaker(class_= Session)
        engine = sa.create_engine(connection_string or SA_CONNECTION_STRING, echo=False)
        session_factory.configure(bind=engine)
        session = session_factory()

        Session.__engine = engine
        Session.__session = session
        return session

    @classmethod
    def cleanup(cls):
        if cls.__session is not None:
            cls.__session.close()
            cls.__session = None
            cls.__engine.dispose()
            cls.__engine = None

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