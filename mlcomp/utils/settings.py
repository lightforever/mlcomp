import os

ROOT_FOLDER = os.getenv('ROOT', '/opt/mlcomp')
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')
MODEL_FOLDER = os.path.join(ROOT_FOLDER, 'models')
TASK_FOLDER = os.path.join(ROOT_FOLDER, 'tasks')

MASTER_PORT_RANGE = list(
    map(int,
        os.getenv('MASTER_PORT_RANGE', '29500-29600').split('-'))
)
