import shutil
import traceback
import requests
import os
import json
from collections import OrderedDict
from functools import wraps
from uuid import uuid4

from flask import Flask, request, Response, send_from_directory, send_file
from flask_cors import CORS
from sqlalchemy.orm import joinedload

import mlcomp.worker.tasks as celery_tasks
from mlcomp import TOKEN, WEB_PORT, WEB_HOST, FLASK_ENV, TMP_FOLDER
from mlcomp.db.enums import TaskStatus, ComponentType
from mlcomp.db.core import PaginatorOptions, Session
from mlcomp.db.providers import ComputerProvider, ProjectProvider, \
    ReportLayoutProvider, ReportProvider, ModelProvider, ReportImgProvider, \
    DagProvider, DagStorageProvider, TaskProvider, LogProvider, StepProvider, \
    FileProvider, AuxiliaryProvider
from mlcomp.db.report_info import ReportLayoutInfo
from mlcomp.server.back.supervisor import register_supervisor
from mlcomp.utils.logging import create_logger
from mlcomp.utils.io import from_module_path, zip_folder
from mlcomp.server.back.create_dags import dag_model_add, dag_model_start
from mlcomp.utils.misc import to_snake, now
from mlcomp.db.models import Model, Report, ReportLayout, Task
from mlcomp.utils.io import yaml_load, yaml_dump
from mlcomp.worker.storage import Storage

app = Flask(__name__)
CORS(app)

_read_session = Session.create_session(key='server.read')
_write_session = Session.create_session(key='server.write')

logger = create_logger(_write_session, __name__)


@app.route('/', defaults={'path': ''}, methods=['GET'])
@app.route('/<path:path>', methods=['GET'])
def send_static(path):
    file = 'index.html'
    if '.' in path:
        file = path

    module_path = from_module_path(__file__, f'../front/dist/mlcomp/')
    return send_from_directory(module_path, file)


def request_data():
    return json.loads(request.data.decode('utf-8'))


def parse_int(args: dict, key: str):
    return int(args[key]) if args.get(key) and args[key].isnumeric() else None


def construct_paginator_options(args: dict, default_sort_column: str):
    return PaginatorOptions(
        sort_column=args.get('sort_column') or default_sort_column,
        sort_descending=args.get('sort_descending', 'true') == 'true',
        page_number=parse_int(args, 'page_number'),
        page_size=parse_int(args, 'page_size'),
    )


def check_auth(token):
    return str(token).strip() == TOKEN


def authenticate():
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials', 401,
        {'WWW-Authenticate': 'xBasic'}
    )


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not check_auth(token):
            return authenticate()
        return f(*args, **kwargs)

    return decorated


def error_handler(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        global _read_session, _write_session, logger

        success = True
        status = 200
        error = ''

        try:
            res = f(*args, **kwargs)
        except Exception as e:
            if Session.sqlalchemy_error(e):
                Session.cleanup('server.read')
                Session.cleanup('server.write')

                _read_session = Session.create_session(key='server.read')
                _write_session = Session.create_session(key='server.write')

                logger = create_logger(_write_session, __name__)

            logger.error(
                f'Requested Url: {request.path}\n\n{traceback.format_exc()}',
                ComponentType.API
            )

            error = traceback.format_exc()
            success = False
            status = 500
            res = None

        res = res or {}
        if isinstance(res, Response):
            return res

        res['success'] = success
        res['error'] = error

        return Response(json.dumps(res), status=status)

    return decorated


@app.route('/api/computers', methods=['POST'])
@requires_auth
@error_handler
def computers():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    options.sort_column = 'name'

    provider = ComputerProvider(_read_session)
    return provider.get(data, options)


@app.route('/api/projects', methods=['POST'])
@requires_auth
@error_handler
def projects():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])

    provider = ProjectProvider(_read_session)
    res = provider.get(data, options)
    return res


@app.route('/api/project/add', methods=['POST'])
@requires_auth
@error_handler
def project_add():
    data = request_data()

    provider = ProjectProvider(_write_session)
    res = provider.add_project(
        data['name'], yaml_load(data['class_names']),
        yaml_load(data['ignore_folders'])
    )
    return res


@app.route('/api/project/edit', methods=['POST'])
@requires_auth
@error_handler
def project_edit():
    data = request_data()

    provider = ProjectProvider(_write_session)
    res = provider.edit_project(
        data['name'], yaml_load(data['class_names']),
        yaml_load(data['ignore_folders'])
    )
    return res


@app.route('/api/report/add_start', methods=['POST'])
@requires_auth
@error_handler
def report_add_start():
    return {
        'projects': ProjectProvider(_read_session).get()['data'],
        'layouts': ReportLayoutProvider(_read_session).get()['data']
    }


@app.route('/api/report/add_end', methods=['POST'])
@requires_auth
@error_handler
def report_add_end():
    data = request_data()

    provider = ReportProvider(_write_session)
    layouts = ReportLayoutProvider(_write_session).all()
    layout = layouts[data['layout']]
    report = Report(
        name=data['name'], project=data['project'], config=yaml_dump(layout)
    )
    provider.add(report)


@app.route('/api/layouts', methods=['POST'])
@requires_auth
@error_handler
def report_layouts():
    data = request_data()

    provider = ReportLayoutProvider(_read_session)
    options = PaginatorOptions(**data['paginator'])
    res = provider.get(data, options)
    return res


@app.route('/api/layout/add', methods=['POST'])
@requires_auth
@error_handler
def report_layout_add():
    data = request_data()

    provider = ReportLayoutProvider(_write_session)
    layout = ReportLayout(name=data['name'], content='', last_modified=now())
    provider.add(layout)


@app.route('/api/layout/edit', methods=['POST'])
@requires_auth
@error_handler
def report_layout_edit():
    data = request_data()

    provider = ReportLayoutProvider(_write_session)
    layout = provider.by_name(data['name'])
    layout.last_modified = now()
    if 'content' in data and data['content'] is not None:
        data_loaded = yaml_load(data['content'])
        ReportLayoutInfo(data_loaded)
        layout.content = data['content']
    if 'new_name' in data and data['new_name'] is not None:
        layout.name = data['new_name']

    provider.commit()


@app.route('/api/layout/remove', methods=['POST'])
@requires_auth
@error_handler
def report_layout_remove():
    data = request_data()

    provider = ReportLayoutProvider(_write_session)
    provider.remove(data['name'], key_column='name')


@app.route('/api/model/add', methods=['POST'])
@requires_auth
@error_handler
def model_add():
    data = request_data()
    dag_model_add(_write_session, data)


@app.route('/api/model/remove', methods=['POST'])
@requires_auth
@error_handler
def model_remove():
    data = request_data()
    provider = ModelProvider(_write_session)
    model = provider.by_id(data['id'], joined_load=[Model.project_rel])
    celery_tasks.remove_model(
        _write_session, model.project_rel.name, model.name
    )
    provider.remove(model.id)


@app.route('/api/model/start_begin', methods=['POST'])
@requires_auth
@error_handler
def model_start_begin():
    data = request_data()
    return ModelProvider(_read_session).model_start_begin(data['model_id'])


@app.route('/api/model/start_end', methods=['POST'])
@requires_auth
@error_handler
def model_start_end():
    data = request_data()
    dag_model_start(_write_session, data)


@app.route('/api/img_classify', methods=['POST'])
@requires_auth
@error_handler
def img_classify():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    res = ReportImgProvider(_read_session).detail_img_classify(data, options)
    return res


@app.route('/api/img_segment', methods=['POST'])
@requires_auth
@error_handler
def img_segment():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    res = ReportImgProvider(_read_session).detail_img_segment(data, options)
    return res


@app.route('/api/config', methods=['POST'])
@requires_auth
@error_handler
def config():
    id = request_data()
    res = DagProvider(_read_session).config(id)
    return {'data': res}


@app.route('/api/graph', methods=['POST'])
@requires_auth
@error_handler
def graph():
    id = request_data()
    res = DagProvider(_read_session).graph(id)
    return res


@app.route('/api/dags', methods=['POST'])
@requires_auth
@error_handler
def dags():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    provider = DagProvider(_read_session)
    res = provider.get(data, options)
    return res


@app.route('/api/code', methods=['POST'])
@requires_auth
@error_handler
def code():
    id = request_data()
    res = OrderedDict()
    parents = dict()
    for s, f in DagStorageProvider(_read_session).by_dag(id):
        s.path = s.path.strip()
        parent = os.path.dirname(s.path)
        name = os.path.basename(s.path)
        if name == '':
            continue

        if s.is_dir:
            node = {'name': name, 'children': []}
            if not parent:
                res[name] = node
                parents[s.path] = res[name]
            else:
                # noinspection PyUnresolvedReferences
                parents[parent]['children'].append(node)
        else:
            node = {'name': name}
            try:
                node['content'] = f.content.decode('utf-8')
            except UnicodeDecodeError:
                node['content'] = ''

            if not parent:
                res[name] = node
            else:
                # noinspection PyUnresolvedReferences
                parents[parent]['children'].append(node)

    def sort_key(x):
        if 'children' in x and len(x['children']) > 0:
            return '_' * 5 + x['name']
        return x['name']

    def sort(node: dict):
        if 'children' in node and len(node['children']) > 0:
            node['children'] = sorted(node['children'], key=sort_key)

            for c in node['children']:
                sort(c)

    res = sorted(list(res.values()), key=sort_key)
    for r in res:
        sort(r)

    return {'items': res}


@app.route('/api/code_download', methods=['GET'])
@requires_auth
@error_handler
def code_download():
    id = int(request.args['id'])
    storage = Storage(_read_session)
    dag = DagProvider().by_id(id)
    folder = os.path.join(TMP_FOLDER, f'{dag.id}({dag.name})')

    try:
        storage.download_dag(id, folder)

        file_name = f'{dag.id}({dag.name}).zip'
        dst = os.path.join(TMP_FOLDER, file_name)
        zip_folder(folder, dst)
        res = send_from_directory(TMP_FOLDER, file_name)
        os.remove(dst)
        return res
    finally:
        shutil.rmtree(folder, ignore_errors=True)


@app.route('/api/tasks', methods=['POST'])
@requires_auth
@error_handler
def tasks():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    provider = TaskProvider(_read_session)
    res = provider.get(data, options)
    return res


@app.route('/api/task/stop', methods=['POST'])
@requires_auth
@error_handler
def task_stop():
    data = request_data()
    provider = TaskProvider(_write_session)
    task = provider.by_id(data['id'], joinedload(Task.dag_rel, innerjoin=True))

    dag = task.dag_rel
    status = celery_tasks.stop(logger, _write_session, task, dag)

    child_tasks = provider.children(task.id)
    for t in child_tasks:
        celery_tasks.stop(logger, _write_session, t, dag)

    return {'status': to_snake(TaskStatus(status).name)}


@app.route('/api/task/info', methods=['POST'])
@requires_auth
@error_handler
def task_info():
    data = request_data()
    task = TaskProvider(_read_session).by_id(
        data['id'], joinedload(Task.dag_rel, innerjoin=True)
    )
    return {
        'pid': task.pid,
        'worker_index': task.worker_index,
        'gpu_assigned': task.gpu_assigned,
        'celery_id': task.celery_id,
        'additional_info': task.additional_info or '',
        'result': task.result or '',
        'id': task.id
    }


@app.route('/api/dag/stop', methods=['POST'])
@requires_auth
@error_handler
def dag_stop():
    data = request_data()
    provider = DagProvider(_write_session)
    id = int(data['id'])
    dag = provider.by_id(id, joined_load=['tasks'])
    for t in dag.tasks:
        celery_tasks.stop(logger, _write_session, t, dag)
    return {'dag': provider.get({'id': id})['data'][0]}


@app.route('/api/dag/start', methods=['POST'])
@requires_auth
@error_handler
def dag_start():
    data = request_data()
    provider = DagProvider(_write_session)
    task_provider = TaskProvider(_write_session)

    id = int(data['id'])
    dag = provider.by_id(id, joined_load=['tasks'])
    can_start_statuses = [
        TaskStatus.Failed.value, TaskStatus.Skipped.value,
        TaskStatus.Stopped.value
    ]

    tasks = list(dag.tasks)

    def find_resume(task):
        children = task_provider.children(task.id)
        children = sorted(children, key=lambda x: x.id, reverse=True)

        if len(children) > 0:
            for c in children:
                if c.parent != task.id:
                    continue

                info = yaml_load(c.additional_info)
                if 'distr_info' not in info:
                    continue

                if info['distr_info']['rank'] == 0:
                    return {
                        'master_computer': c.computer_assigned,
                        'master_task_id': c.id,
                        'load_last': True
                    }
            raise Exception('Master task not found')
        else:
            return {
                'master_computer': task.computer_assigned,
                'master_task_id': task.id,
                'load_last': True
            }

    for t in tasks:
        if t.status not in can_start_statuses:
            continue

        if t.parent:
            continue

        info = yaml_load(t.additional_info)
        info['resume'] = find_resume(t)
        t.additional_info = yaml_dump(info)

        t.status = TaskStatus.NotRan.value
        t.pid = None
        t.started = None
        t.finished = None
        t.computer_assigned = None
        t.celery_id = None
        t.worker_index = None
        t.docker_assigned = None

    provider.commit()


@app.route('/api/auxiliary', methods=['POST'])
def auxiliary():
    provider = AuxiliaryProvider(_read_session)
    return provider.get()


@app.route('/api/dag/toogle_report', methods=['POST'])
@requires_auth
@error_handler
def dag_toogle_report():
    data = request_data()
    provider = ReportProvider(_write_session)
    if data.get('remove'):
        provider.remove_dag(int(data['id']), int(data['report']))
    else:
        provider.add_dag(int(data['id']), int(data['report']))
    return {'report_full': not data.get('remove')}


@app.route('/api/task/toogle_report', methods=['POST'])
@requires_auth
@error_handler
def task_toogle_report():
    data = request_data()
    provider = ReportProvider(_write_session)
    if data.get('remove'):
        provider.remove_task(int(data['id']), int(data['report']))
    else:
        provider.add_task(int(data['id']), int(data['report']))
    return {'report_full': not data.get('remove')}


@app.route('/api/logs', methods=['POST'])
@requires_auth
@error_handler
def logs():
    provider = LogProvider(_read_session)
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    res = provider.get(data, options)
    return res


@app.route('/api/reports', methods=['POST'])
@requires_auth
@error_handler
def reports():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    provider = ReportProvider(_read_session)
    res = provider.get(data, options)
    return res


@app.route('/api/report', methods=['POST'])
@requires_auth
@error_handler
def report():
    id = request_data()
    provider = ReportProvider(_read_session)
    res = provider.detail(id)
    return res


@app.route('/api/report/update_layout_start', methods=['POST'])
@requires_auth
@error_handler
def report_update_layout_start():
    id = request_data()['id']
    provider = ReportProvider(_write_session)
    res = provider.update_layout_start(id)
    return res


@app.route('/api/report/update_layout_end', methods=['POST'])
@requires_auth
@error_handler
def report_update_layout_end():
    data = request_data()
    provider = ReportProvider(_write_session)
    layout_provider = ReportLayoutProvider(_write_session)
    provider.update_layout_end(
        data['id'], data['layout'], layout_provider.all()
    )
    return provider.detail(data['id'])


@app.route('/api/task/steps', methods=['POST'])
@requires_auth
@error_handler
def steps():
    id = request_data()
    provider = StepProvider(_read_session)
    res = provider.get(id)
    return res


@app.route('/api/token', methods=['POST'])
def token():
    data = request_data()
    if str(data['token']).strip() != TOKEN:
        return Response(
            json.dumps({
                'success': False,
                'reason': 'invalid token'
            }),
            status=401
        )
    return json.dumps({'success': True})


@app.route('/api/project/remove', methods=['POST'])
@requires_auth
@error_handler
def project_remove():
    id = request_data()['id']
    ProjectProvider(_write_session).remove(id)


@app.route('/api/remove_imgs', methods=['POST'])
@requires_auth
@error_handler
def remove_imgs():
    data = request_data()
    provider = ReportImgProvider(_write_session)
    res = provider.remove(data)
    return res


@app.route('/api/remove_files', methods=['POST'])
@requires_auth
@error_handler
def remove_files():
    data = request_data()
    provider = FileProvider(_write_session)
    res = provider.remove(data)
    return res


@app.route('/api/dag/remove', methods=['POST'])
@requires_auth
@error_handler
def dag_remove():
    id = request_data()['id']
    celery_tasks.remove_dag(_write_session, id)
    DagProvider(_write_session).remove(id)


@app.route('/api/models', methods=['POST'])
@requires_auth
@error_handler
def models():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    provider = ModelProvider(_read_session)
    res = provider.get(data, options)
    return res


@app.route('/api/stop')
@requires_auth
@error_handler
def stop():
    pass


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/api/shutdown', methods=['POST'])
@requires_auth
@error_handler
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


def start_server():
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        logger.info(f'Server TOKEN = {TOKEN}', ComponentType.API)
        register_supervisor()

    app.run(debug=FLASK_ENV == 'development', port=WEB_PORT, host=WEB_HOST)


def stop_server():
    requests.post(
        f'http://localhost:{WEB_PORT}/api/shutdown',
        headers={'Authorization': TOKEN}
    )
