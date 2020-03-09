import shutil
import traceback
from glob import glob
from os import makedirs
from os.path import dirname, join, basename, exists

import pandas as pd
import migrate.versioning.api as api
from mlcomp.utils.io import zip_folder

from mlcomp.db.enums import LogStatus, ComponentType

from mlcomp import SA_CONNECTION_STRING, REPORT_FOLDER, LOG_FOLDER, DB_TYPE, \
    CONFIG_FOLDER, ROOT_FOLDER, DATA_FOLDER, MODEL_FOLDER, TASK_FOLDER, \
    DB_FOLDER, TMP_FOLDER
from mlcomp.utils.misc import now, to_snake

from mlcomp.db.providers import DagProvider, LogProvider


def statuses(folder: str = None):
    rows = []

    folder_status = 'OK'
    folder_comment = ''

    folders = [
        ROOT_FOLDER,
        DATA_FOLDER,
        MODEL_FOLDER,
        TASK_FOLDER,
        LOG_FOLDER,
        CONFIG_FOLDER,
        DB_FOLDER,
        REPORT_FOLDER,
        TMP_FOLDER
    ]
    for f in folders:
        if not exists(f):
            folder_status = 'ERROR'
            folder_comment = f'folder {f} does not exist'

    files = [
        join(CONFIG_FOLDER, '.env')
    ]
    for f in files:
        if not exists(f):
            folder_status = 'ERROR'
            folder_comment = f'file {f} does not exist'

    rows.append({
        'name': 'Folders',
        'status': folder_status,
        'comment': folder_comment
    })

    database_status = 'OK'
    database_comment = f'DB_TYPE = {DB_TYPE}'
    try:
        provider = DagProvider()
        provider.count()
    except Exception:
        database_status = 'ERROR'
        database_comment += ' ' + traceback.format_exc()

    rows.append({
        'name': 'Database',
        'status': database_status,
        'comment': database_comment
    })

    if database_status == 'OK':
        migrate_status = 'OK'

        repository_folder = join(dirname(__file__), 'migration')
        repository_version = api.version(repository_folder)

        db_version = api.db_version(SA_CONNECTION_STRING, repository_folder)

        if db_version != repository_version:
            migrate_status = 'ERROR'
            migrate_comment = f'Repository version = {repository_version} ' \
                              f'Db version = {db_version}'
        else:
            migrate_comment = f'version: {db_version}'

        rows.append({
            'name': 'Migrate',
            'status': migrate_status,
            'comment': migrate_comment
        })

    df = pd.DataFrame(rows)

    if folder is not None:
        print('Statuses:')
        print(df)

        df.to_csv(join(folder, 'statuses.csv'), index=False)

    return df


def check_statuses():
    stats = statuses()
    failed = stats[stats['status'] != 'OK']
    if failed.shape[0] > 0:
        print('There are errors in statuses')
        for row in stats.itertuples():
            print(f'name: {row.name} status'
                  f' {row.status} comment {row.comment}')

        import time
        time.sleep(0.01)

        raise Exception('There are errors in statuses. '
                        'Please check them above')


def logs(statuses, folder: str = None):
    if folder is not None:
        for file in glob(join(LOG_FOLDER, '*')):
            shutil.copy(file, join(folder, basename(file)))
        print('logs formed')

    if statuses.query('status == "ERROR"').shape[0] > 0:
        return

    log_provider = LogProvider()
    errors = log_provider.last(count=1000, levels=[LogStatus.Error.value])
    service_components = [ComponentType.Supervisor.value,
                          ComponentType.API.value,
                          ComponentType.WorkerSupervisor.value]
    services = log_provider.last(count=1000, components=service_components)
    logs = errors + services

    rows = []
    for l, _ in logs:
        rows.append({
            'status': to_snake(LogStatus(l.level).name),
            'component': to_snake(ComponentType(l.component).name),
            'time': l.time,
            'message': l.message,
        })
    df = pd.DataFrame(rows)
    df.to_csv(join(folder, 'logs_db.csv'), index=False)
    return df


def create_report():
    print('*** Report Start ***')
    print()

    folder = join(REPORT_FOLDER, f'{now()}'.split('.')[0])
    makedirs(folder, exist_ok=True)

    statuses_res = statuses(folder)

    print()

    logs(statuses_res, folder)

    print()

    zip_path = folder + '.zip'
    zip_folder(folder, dst=zip_path)

    print('Report path', zip_path)
    print()

    print('*** Report End ***')


__all__ = ['check_statuses', 'create_report']
