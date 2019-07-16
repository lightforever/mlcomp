from .base import *
import os
if not os.getenv('SERVER'):
    from .kaggle import *
    from .catalyst import *
    from .model import *
    from .split import *