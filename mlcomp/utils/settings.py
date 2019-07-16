import os

ROOT_FOLDER = os.getenv('ROOT', '/opt/mlcomp')
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')
MODEL_FOLDER = os.path.join(ROOT_FOLDER, 'models')
TASK_FOLDER = os.path.join(ROOT_FOLDER, 'tasks')