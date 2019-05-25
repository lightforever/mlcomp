import os
import uuid

TOKEN = os.getenv('TOKEN', str(uuid.uuid4()))