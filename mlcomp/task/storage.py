from glob import glob
import os
import logging
from mlcomp.db.core import Session

logger = logging.getLogger(__name__)

class Storage:
    @staticmethod
    def upload(folder: str):
        session = Session()