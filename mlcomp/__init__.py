import os
from os.path import join
import shutil

from .__version__ import __version__  # noqa: F401

ROOT_FOLDER = os.path.expanduser('~/mlcomp')
DATA_FOLDER = join(ROOT_FOLDER, 'data')
MODEL_FOLDER = join(ROOT_FOLDER, 'models')
TASK_FOLDER = join(ROOT_FOLDER, 'tasks')
LOG_FOLDER = join(ROOT_FOLDER, 'logs')
CONFIG_FOLDER = join(ROOT_FOLDER, 'configs')
DB_FOLDER = join(ROOT_FOLDER, 'db')
TEST_FOLDER = join(ROOT_FOLDER, 'tests')

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(TASK_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(CONFIG_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

# copy conf files if they do not exist

folder = os.path.dirname(__file__)
docker_folder = join(folder, 'docker')

for name in os.listdir(docker_folder):
    file = join(docker_folder, name)
    target_file = join(CONFIG_FOLDER, name)
    if not os.path.exists(target_file):
        shutil.copy(file, target_file)

# exporting environment variables
with open(join(CONFIG_FOLDER, '.env')) as f:
    for l in f.readlines():
        k, v = l.strip().split('=')
        os.environ[k] = v

# for debugging
os.environ['PYTHONPATH'] = '.'

MASTER_PORT_RANGE = list(map(int, os.getenv('MASTER_PORT_RANGE').split('-')))
MODE_ECONOMIC = os.getenv('MODE_ECONOMIC') == 'True'

__all__ = [
    'ROOT_FOLDER', 'DATA_FOLDER', 'MODEL_FOLDER', 'TASK_FOLDER', 'LOG_FOLDER',
    'CONFIG_FOLDER', 'DB_FOLDER', 'TEST_FOLDER', 'MASTER_PORT_RANGE'
]
