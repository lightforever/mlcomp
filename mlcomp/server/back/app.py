import traceback
import requests
import os
import json
from collections import OrderedDict
from functools import wraps

from flask import Flask, request, Response, send_from_directory
from flask_cors import CORS
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import joinedload

import mlcomp.worker.tasks as celery_tasks
from mlcomp.db.enums import TaskStatus, ComponentType
from mlcomp.db.providers import *
from mlcomp.db.core import PaginatorOptions, Session
from mlcomp.server.back.supervisor import register_supervisor
from mlcomp.server.back import conf
from mlcomp.utils.logging import logger
from mlcomp.utils.io import from_module_path
from mlcomp.server.back.create_dags import dag_model_add, dag_model_start
from mlcomp.utils.misc import to_snake

HOST = os.getenv('WEB_HOST', '0.0.0.0')
PORT = int(os.getenv('WEB_PORT', '4201'))

app = Flask(__name__)
CORS(app)


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
    return str(token).strip() == conf.TOKEN


def authenticate():
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials', 401,
        {'WWW-Authenticate': 'xBasic'})


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not check_auth(token):
            return authenticate()
        return f(*args, **kwargs)

    return decorated


@app.route('/api/computers', methods=['POST'])
@requires_auth
def computers():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    options.sort_column = 'name'

    provider = ComputerProvider()
    return json.dumps(provider.get(data, options))


@app.route('/api/projects', methods=['POST'])
@requires_auth
def projects():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])

    provider = ProjectProvider()
    res = provider.get(data, options)
    return json.dumps(res)


@app.route('/api/project/add', methods=['POST'])
@requires_auth
def project_add():
    data = request_data()

    provider = ProjectProvider()
    res = provider.add(data['name'], dict())
    return json.dumps(res)


@app.route('/api/model/add', methods=['POST'])
@requires_auth
def model_add():
    data = request_data()
    success = True
    error = ''
    try:
        dag_model_add(data)
    except Exception:
        error = traceback.format_exc()
        success = False
    return json.dumps({'success': success, 'error': error})


@app.route('/api/model/remove', methods=['POST'])
@requires_auth
def model_remove():
    data = request_data()
    success = True
    error = ''
    try:
        ModelProvider().remove(data['id'])
    except Exception:
        error = traceback.format_exc()
        success = False
    return json.dumps({'success': success, 'error': error})


@app.route('/api/model/start', methods=['POST'])
@requires_auth
def model_start():
    data = request_data()
    dag_model_start(data)
    return json.dumps({'success': True})


@app.route('/api/img_classify', methods=['POST'])
@requires_auth
def img_classify():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    res = ReportImgProvider().detail_img_classify(data, options)
    return json.dumps(res)


@app.route('/api/config', methods=['POST'])
@requires_auth
def config():
    id = request_data()
    res = DagProvider().config(id)
    return json.dumps({'data': res})


@app.route('/api/graph', methods=['POST'])
@requires_auth
def graph():
    id = request_data()
    res = DagProvider().graph(id)
    return json.dumps(res)


@app.route('/api/dags', methods=['POST'])
@requires_auth
def dags():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    provider = DagProvider()
    res = provider.get(data, options)
    return json.dumps(res)


@app.route('/api/code', methods=['POST'])
@requires_auth
def code():
    id = request_data()
    res = OrderedDict()
    parents = dict()
    for s, f in DagStorageProvider().by_dag(id):
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
                parents[parent]['children'].append(node)

    return json.dumps(list(res.values()))


@app.route('/api/tasks', methods=['POST'])
@requires_auth
def tasks():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    provider = TaskProvider()
    res = provider.get(data, options)
    return json.dumps(res)


@app.route('/api/task/stop', methods=['POST'])
@requires_auth
def task_stop():
    data = request_data()
    task = TaskProvider().by_id(data['id'], joinedload(Task.dag_rel))
    status = celery_tasks.stop(task)
    return json.dumps({
        'success': True,
        'status': to_snake(TaskStatus(status).name)
    })


@app.route('/api/dag/stop', methods=['POST'])
@requires_auth
def dag_stop():
    data = request_data()
    provider = DagProvider()
    id = int(data['id'])
    dag = provider.by_id(id, joined_load=['tasks'])
    for t in dag.tasks:
        t.dag_rel = dag
        celery_tasks.stop(t)
    return json.dumps(
        {'success': True,
         'dag': provider.get({'id': id})['data'][0]
         })


@app.route('/api/dag/toogle_report', methods=['POST'])
@requires_auth
def dag_toogle_report():
    data = request_data()
    provider = ReportProvider()
    if data.get('remove'):
        provider.remove_dag(int(data['id']), int(data['report']))
    else:
        provider.add_dag(int(data['id']), int(data['report']))
    return json.dumps({
        'success': True,
        'report_full': not data.get('remove')
    })


@app.route('/api/task/toogle_report', methods=['POST'])
@requires_auth
def task_toogle_report():
    data = request_data()
    provider = ReportProvider()
    if data.get('remove'):
        provider.remove_task(int(data['id']), int(data['report']))
    else:
        provider.add_task(int(data['id']), int(data['report']))
    return json.dumps({
        'success': True,
        'report_full': not data.get('remove')
    })


@app.route('/api/logs', methods=['POST'])
@requires_auth
def logs():
    provider = LogProvider()
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    res = provider.get(data, options)
    return json.dumps(res)


@app.route('/api/reports', methods=['POST'])
@requires_auth
def reports():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    provider = ReportProvider()
    res = provider.get(data, options)
    return json.dumps(res)


@app.route('/api/report', methods=['POST'])
@requires_auth
def report():
    id = request_data()
    provider = ReportProvider()
    res = provider.detail(id)
    return json.dumps(res)


@app.route('/api/task/steps', methods=['POST'])
@requires_auth
def steps():
    id = request_data()
    provider = StepProvider()
    res = provider.get(id)
    return json.dumps(res)


@app.route('/api/token', methods=['POST'])
def token():
    data = request_data()
    if str(data['token']).strip() != conf.TOKEN:
        return Response(
            json.dumps({'success': False, 'reason': 'invalid token'}),
            status=401)
    return json.dumps({'success': True})


@app.route('/api/project/remove', methods=['POST'])
@requires_auth
def project_remove():
    id = request_data()['id']
    ProjectProvider().remove(id)
    return json.dumps({'success': True})


@app.route('/api/remove_imgs', methods=['POST'])
@requires_auth
def remove_imgs():
    data = request_data()
    provider = ReportImgProvider()
    res = provider.remove(data)
    return json.dumps(res)


@app.route('/api/remove_files', methods=['POST'])
@requires_auth
def remove_files():
    data = request_data()
    provider = FileProvider()
    res = provider.remove(data)
    return json.dumps(res)


@app.route('/api/dag/remove', methods=['POST'])
@requires_auth
def dag_remove():
    id = request_data()['id']
    DagProvider().remove(id)
    return json.dumps({'success': True})


@app.route('/api/models', methods=['POST'])
@requires_auth
def models():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    provider = ModelProvider()
    res = provider.get(data, options)
    return json.dumps(res)


@app.route('/api/stop')
@requires_auth
def stop():
    pass


@app.errorhandler(Exception)
def all_exception_handler(error):
    if type(error) == ProgrammingError:
        Session.cleanup()

    logger.error(f'Requested Url: {request.path}\n\n{traceback.format_exc()}',
                 ComponentType.API)
    return str(error), 500


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/api/shutdown', methods=['POST'])
@requires_auth
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


def start_server():
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        logger.info(f'Server TOKEN = {conf.TOKEN}', ComponentType.API)
        register_supervisor()
    app.run(debug=os.getenv('FLASK_ENV') == 'development', port=PORT,
            host=HOST)


def stop_server():
    requests.post(f'http://localhost:{PORT}/api/shutdown',
                  headers={'Authorization': conf.TOKEN})
