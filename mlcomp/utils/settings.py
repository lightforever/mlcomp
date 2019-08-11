import os

ROOT_FOLDER = os.getenv('ROOT', '/opt/mlcomp')
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')
MODEL_FOLDER = os.path.join(ROOT_FOLDER, 'models')
TASK_FOLDER = os.path.join(ROOT_FOLDER, 'tasks')
LOG_FOLDER = os.path.join(ROOT_FOLDER, 'logs')
CONFIG_FOLDER = os.path.join(ROOT_FOLDER, 'configs')

os.makedirs(CONFIG_FOLDER, exist_ok=True)
os.makedirs(TASK_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

MASTER_PORT_RANGE = list(
    map(int,
        os.getenv('MASTER_PORT_RANGE').split('-'))
)
