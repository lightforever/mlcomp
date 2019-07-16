import os
import uuid

TOKEN = str(os.getenv('TOKEN', uuid.uuid4()))
