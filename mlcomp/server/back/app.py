import hashlib
import shutil
import traceback
import requests
import os
import simplejson as json
from collections import OrderedDict
from functools import wraps

from flask import Flask, request, Response, send_from_directory
from flask_cors import CORS
from sqlalchemy.orm import joinedload

import mlcomp.worker.tasks as celery_tasks
from mlcomp import TOKEN, WEB_PORT, WEB_HOST, FLASK_ENV, TMP_FOLDER
from mlcomp.db.enums import TaskStatus, ComponentType
from mlcomp.db.core import PaginatorOptions, Session
from mlcomp.db.models.dag import DagTag
from mlcomp.db.providers import ComputerProvider, ProjectProvider, \
    ReportLayoutProvider, ReportProvider, ModelProvider, ReportImgProvider, \
    DagProvider, DagStorageProvider, TaskProvider, LogProvider, StepProvider, \
    FileProvider, AuxiliaryProvider, MemoryProvider, SpaceProvider
from mlcomp.db.report_info import ReportLayoutInfo
from mlcomp.server.back.create_dags.copy import dag_copy
from mlcomp.server.back.supervisor import register_supervisor
from mlcomp.utils.logging import create_logger
from mlcomp.utils.io import from_module_path, zip_folder
from mlcomp.server.back.create_dags import dag_model_add, dag_model_start
from mlcomp.utils.misc import now
from mlcomp.db.models import Model, Report, ReportLayout, Task, File, Memory, \
    Space, SpaceTag
from mlcomp.utils.io import yaml_load, yaml_dump
from mlcomp.worker.storage import Storage

app = Flask(__name__)
CORS(app)

_read_session = Session.create_session(key='server.read')
_write_session = Session.create_session(key='server.write')

logger = create_logger(_write_session, __name__)
supervisor = None


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

        return Response(json.dumps(res, ignore_nan=True), status=status)

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


@app.route('/api/computer_sync_start', methods=['POST'])
@requires_auth
@error_handler
def computer_sync_start():
    provider = ComputerProvider(_read_session)
    return provider.sync_start()


@app.route('/api/computer_sync_end', methods=['POST'])
@requires_auth
@error_handler
def computer_sync_end():
    data = request_data()
    provider = ComputerProvider(_write_session)
    for computer in provider.all():
        if data.get('computer') and data['computer'] != computer.name:
            continue
        meta = yaml_load(computer.meta)
        meta['manual_sync'] = {
            'project': data['id'],
            'sync_folders': yaml_load(data['sync_folders']),
            'ignore_folders': yaml_load(data['ignore_folders']),
        }
        computer.meta = yaml_dump(meta)
    provider.update()


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
    provider.add_project(
        data['name'],
        yaml_load(data['class_names']),
        yaml_load(data['sync_folders']),
        yaml_load(data['ignore_folders']),
    )


@app.route('/api/project/stop_all_dags', methods=['POST'])
@requires_auth
@error_handler
def stop_all_dags():
    data = request_data()
    provider = TaskProvider(_write_session)
    tasks = provider.by_status(TaskStatus.InProgress,
                               TaskStatus.Queued,
                               TaskStatus.NotRan,
                               project=data['project']
                               )

    for t in tasks:
        info = yaml_load(t.additional_info)
        info['stopped'] = True
        t.additional_info = yaml_dump(info)

    provider.update()
    supervisor.stop_tasks(tasks)


@app.route('/api/project/remove_all_dags', methods=['POST'])
@requires_auth
@error_handler
def remove_all_dags():
    data = request_data()
    provider = DagProvider(_write_session)
    dags = provider.by_project(data['project'])
    provider.remove_all([d.id for d in dags])


@app.route('/api/project/edit', methods=['POST'])
@requires_auth
@error_handler
def project_edit():
    data = request_data()

    provider = ProjectProvider(_write_session)
    provider.edit_project(
        data['name'],
        yaml_load(data['class_names']),
        yaml_load(data['sync_folders']),
        yaml_load(data['ignore_folders']),
    )


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


@app.route('/api/space/tag_add', methods=['POST'])
@requires_auth
@error_handler
def space_tag_add():
    data = request_data()
    provider = SpaceProvider(_write_session)
    tag = SpaceTag(space=data['space'], tag=data['tag'])
    provider.add(tag)


@app.route('/api/space/tag_remove', methods=['POST'])
@requires_auth
@error_handler
def space_tag_remove():
    data = request_data()
    provider = SpaceProvider(_write_session)
    provider.remove_tag(space=data['space'], tag=data['tag'])


@app.route('/api/dag/tag_add', methods=['POST'])
@requires_auth
@error_handler
def dag_tag_add():
    data = request_data()
    provider = DagProvider(_write_session)
    tag = DagTag(dag=data['dag'], tag=data['tag'])
    provider.add(tag)


@app.route('/api/dag/tag_remove', methods=['POST'])
@requires_auth
@error_handler
def dag_tag_remove():
    data = request_data()
    provider = DagProvider(_write_session)
    provider.remove_tag(dag=data['dag'], tag=data['tag'])


@app.route('/api/dag/restart', methods=['POST'])
@requires_auth
@error_handler
def dag_restart():
    data = request_data()
    dag_copy(_write_session, data['dag'], data['file_changes'])


@app.route('/api/spaces', methods=['POST'])
@requires_auth
@error_handler
def spaces():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    if options.sort_column == 'id':
        options.sort_column = 'name'
    provider = SpaceProvider(_read_session)
    res = provider.get(data, options)
    return res


def set_space_fields(space: Space, data: dict):
    data['content'] = data.get('content', '')
    yaml_load(data['content'])

    space.name = data['name']
    space.content = data['content']
    if not space.created:
        space.created = now()
    space.changed = now()
    return space


@app.route('/api/space/relation_append', methods=['POST'])
@requires_auth
@error_handler
def space_relation_append():
    data = request_data()
    provider = SpaceProvider(_write_session)
    provider.add_relation(data['parent'], data['child'])


@app.route('/api/space/relation_remove', methods=['POST'])
@requires_auth
@error_handler
def space_relation_remove():
    data = request_data()
    provider = SpaceProvider(_write_session)
    provider.remove_relation(data['parent'], data['child'])


@app.route('/api/space/run', methods=['POST'])
@requires_auth
@error_handler
def space_run():
    data = request_data()
    provider = SpaceProvider(_write_session)
    file_changes = data.get('file_changes', '\n')
    file_changes = yaml_load(file_changes)

    def merge(d: dict, d2: dict):
        res = {}
        for k in set(d) | set(d2):
            if k in d and k in d2:
                v = d[k]
                v2 = d2[k]
                if isinstance(v, list) and isinstance(v2, list):
                    res[k] = v[:]
                    res[k].extend(v2)
                elif isinstance(v, dict) and isinstance(v2, dict):
                    res[k] = v.copy()
                    res[k].update(v2)
                else:
                    raise Exception(f'Types are different: {type(v)}, {type(v2)}')
            elif k in d:
                res[k] = d[k]
            elif k in d2:
                res[k] = d2[k]
        return res

    for space in data['spaces']:
        if space['logic'] == 'and':
            space = provider.by_id(space['value'], key_column='name')
            if space.content:
                d = yaml_load(space.content)
                file_changes = merge(file_changes, d)

    has_or = any(s['logic'] == 'or' for s in data['spaces'])
    for space in data['spaces']:
        if space['logic'] != 'or' and has_or:
            continue
        space = provider.by_id(space['value'], key_column='name')
        space_related = provider.related(space.name)
        if space.content:
            space_related += [space]

        for rel in space_related:
            content = rel.content
            d = yaml_load(content)
            d = merge(file_changes, d)

            dag_copy(_write_session, data['dag'], file_changes=yaml_dump(d),
                     dag_suffix=rel.name)


@app.route('/api/space/add', methods=['POST'])
@requires_auth
@error_handler
def space_add():
    data = request_data()
    provider = SpaceProvider(_write_session)
    space = Space()
    set_space_fields(space, data)
    provider.add(space)


@app.route('/api/space/copy', methods=['POST'])
@requires_auth
@error_handler
def space_copy():
    data = request_data()
    provider = SpaceProvider(_write_session)
    space = Space()
    set_space_fields(space, data['space'])
    provider.add(space)

    old_children = provider.related(data['old_space'])
    for c in old_children:
        provider.add_relation(space.name, c.name)


@app.route('/api/space/edit', methods=['POST'])
@requires_auth
@error_handler
def space_edit():
    data = request_data()
    provider = SpaceProvider(_write_session)
    space = provider.by_id(data['name'], key_column='name')
    set_space_fields(space, data)
    provider.update()


@app.route('/api/space/remove', methods=['POST'])
@requires_auth
@error_handler
def space_remove():
    data = request_data()
    provider = SpaceProvider(_write_session)
    provider.remove(data['name'], key_column='name')


@app.route('/api/space/tags', methods=['POST'])
@requires_auth
@error_handler
def space_tags():
    data = request_data()
    provider = SpaceProvider(_write_session)
    return provider.tags(data['name'])


@app.route('/api/space/names', methods=['POST'])
@requires_auth
@error_handler
def space_names():
    data = request_data()
    provider = SpaceProvider(_write_session)
    return provider.names(data['name'])


@app.route('/api/memories', methods=['POST'])
@requires_auth
@error_handler
def memories():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    provider = MemoryProvider(_read_session)
    res = provider.get(data, options)
    return res


def set_memory_fields(memory: Memory, data: dict):
    memory.model = data['model']
    memory.memory = float(data['memory'])
    memory.batch_size = int(data['batch_size'])
    memory.variant = data.get('variant')
    if data.get('num_classes'):
        memory.num_classes = int(data['num_classes'])


@app.route('/api/memory/add', methods=['POST'])
@requires_auth
@error_handler
def memory_add():
    data = request_data()
    provider = MemoryProvider(_write_session)
    memory = Memory()
    set_memory_fields(memory, data)
    provider.add(memory)


@app.route('/api/memory/edit', methods=['POST'])
@requires_auth
@error_handler
def memory_edit():
    data = request_data()
    provider = MemoryProvider(_write_session)
    memory = provider.by_id(data['id'])
    set_memory_fields(memory, data)
    provider.update()


@app.route('/api/memory/remove', methods=['POST'])
@requires_auth
@error_handler
def memory_remove():
    data = request_data()
    provider = MemoryProvider(_write_session)
    provider.remove(data['id'])


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
            node = {'name': name, 'children': [], 'id': s.id, 'dag': id,
                    'storage': s.id}
            if not parent:
                res[name] = node
                parents[s.path] = node
            else:
                # noinspection PyUnresolvedReferences
                parents[parent]['children'].append(node)
                parents[os.path.join(parent, name)] = node
        else:
            node = {'name': name, 'id': f.id, 'dag': id, 'storage': s.id}

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


@app.route('/api/update_code', methods=['POST'])
@requires_auth
@error_handler
def update_code():
    data = request_data()
    provider = FileProvider(_write_session)
    file = provider.by_id(data['file_id'])
    content = data['content'].encode('utf-8')
    md5 = hashlib.md5(content).hexdigest()

    if md5 == file.md5:
        return

    if file.dag != data['dag']:
        new_file = File(md5=md5, content=content, project=file.project,
                        dag=data['dag'])
        provider.add(new_file)

        storage = DagStorageProvider(_write_session).by_id(data['storage'])
        storage.file = new_file.id
        provider.commit()
        return {'file': new_file.id}
    else:
        file.content = content
        file.md5 = md5
        provider.commit()
        return {'file': file.id}


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

    tasks = [task] + provider.children(task.id)
    supervisor.stop_tasks(tasks)


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


@app.route('/api/dag/tags', methods=['POST'])
@requires_auth
@error_handler
def dag_tags():
    data = request_data()
    provider = DagProvider(_write_session)
    return provider.tags(data['name'])


@app.route('/api/dag/stop', methods=['POST'])
@requires_auth
@error_handler
def dag_stop():
    data = request_data()
    provider = TaskProvider(_write_session)
    id = int(data['id'])
    tasks = provider.by_dag(id)

    supervisor.stop_tasks(tasks)

    dag_provider = DagProvider(_write_session)
    return {'dag': dag_provider.get({'id': id})['data'][0]}


@app.route('/api/dag/start', methods=['POST'])
@requires_auth
@error_handler
def dag_start():
    data = request_data()
    id = int(data['id'])
    supervisor.start_dag(id)


@app.route('/api/auxiliary', methods=['POST'])
@requires_auth
@error_handler
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

    dag_provider = DagProvider(_write_session)
    dag_provider.remove(id)


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
    global supervisor
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        logger.info(f'Server TOKEN = {TOKEN}', ComponentType.API)
        supervisor = register_supervisor()

    app.run(debug=FLASK_ENV == 'development', port=WEB_PORT, host=WEB_HOST)


def stop_server():
    requests.post(
        f'http://localhost:{WEB_PORT}/api/shutdown',
        headers={'Authorization': TOKEN}
    )
