import math
import logging

logger = logging.getLogger(__name__)

class Handler(object):
    def __init__(self, session):
        self.session = session

    def pre_insert_item(self, data):
        for k, v in data.items():
            if isinstance(v, float):
                if math.isnan(v):
                    data[k] = None
        return data

    def insert(self, obj):
        try:
            self.session.add(obj)
        except Exception as e:
            logger.error('====ADD ERROR====')
            logger.error(e)
            return
        try:
            self.session.commit()
        except Exception as e:
            logger.error('====COMMIT ERROR====')
            logger.error(e)
            self.session.rollback()

    def update(self, session):
        try:
            self.session.commit()
        except Exception as e:
            logger.error('====COMMIT ERROR====')
            logger.error(e)
            self.session.rollback()
