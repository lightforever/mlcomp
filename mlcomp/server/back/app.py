from flask import Flask, request
import json
import click
import requests
from mlcomp.db.providers import *
from mlcomp.db.core import PaginatorOptions
from flask_cors import CORS
from mlcomp.server.back.supervisor import register_supervisor
import os
import mlcomp.task.tasks as celery_tasks

PORT = 4201

app = Flask(__name__)
CORS(app)


def request_data():
    return json.loads(request.data.decode('utf-8'))


def parse_int(args: dict, key: str):
    return int(args[key]) if args.get(key) and args[key].isnumeric() else None


def construct_paginator_options(args: dict, default_sort_column: str):
    return PaginatorOptions(sort_column=args.get('sort_column') or default_sort_column,
                            sort_descending=args['sort_descending'] == 'true' if 'sort_descending' in args else True,
                            page_number=parse_int(args, 'page_number'),
                            page_size=parse_int(args, 'page_size'),
                            )


@app.route('/computers', methods=['POST'])
def computers():
    data = request_data()
    options = PaginatorOptions(**data)
    options.sort_column = 'name'

    provider = ComputerProvider()
    return json.dumps(provider.get(data, options))


@app.route('/projects', methods=['POST'])
def projects():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])

    provider = ProjectProvider()
    res = provider.get(data, options)
    return json.dumps(res)


def get_dag_id():
    assert 'dag' in request.args, 'dag is needed'
    assert request.args['dag'].isnumeric(), 'dag must be integer'
    return int(request.args['dag'])


@app.route('/config', methods=['GET'])
def config():
    id = get_dag_id()
    res = DagProvider().config(id)
    return json.dumps({'data': res})


@app.route('/graph', methods=['GET'])
def graph():
    id = get_dag_id()
    res = DagProvider().graph(id)
    return json.dumps(res)


@app.route('/dags', methods=['POST'])
def dags():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    provider = DagProvider()
    res = provider.get(data, options)
    return json.dumps(res)


@app.route('/code', methods=['GET'])
def code():
    id = get_dag_id()
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


@app.route('/tasks', methods=['POST'])
def tasks():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    provider = TaskProvider()
    res = provider.get(data, options)
    return json.dumps(res)


@app.route('/task/stop', methods=['POST'])
def task_stop():
    data = request_data()
    status = celery_tasks.stop(data['id'])
    return json.dumps({'success': True, 'status': to_snake(TaskStatus(status).name)})


@app.route('/dag/stop', methods=['POST'])
def dag_stop():
    data = request_data()
    provider = DagProvider()
    id = int(data['id'])
    dag = provider.by_id(id, joined_load=['tasks'])
    for t in dag.tasks:
        celery_tasks.stop(t.id)
    return json.dumps({'success': True, 'dag': provider.get({'id': id})['data'][0]})


@app.route('/dag/toogle_report', methods=['POST'])
def dag_toogle_report():
    data = request_data()
    provider = ReportProvider()
    if data.get('remove'):
        provider.remove_dag(int(data['id']), int(data['report']))
    else:
        provider.add_dag(int(data['id']), int(data['report']))
    return json.dumps({'success': True, 'report_full': not data.get('remove')})


@app.route('/task/toogle_report', methods=['POST'])
def task_toogle_report():
    data = request_data()
    provider = ReportProvider()
    if data.get('remove'):
        provider.remove_task(int(data['id']), int(data['report']))
    else:
        provider.add_task(int(data['id']), int(data['report']))
    return json.dumps({'success': True, 'report_full': not data.get('remove')})


@app.route('/logs', methods=['POST'])
def logs():
    provider = LogProvider()
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    res = provider.get(data, options)
    return json.dumps(res)


@app.route('/reports', methods=['POST'])
def reports():
    data = request_data()
    options = PaginatorOptions(**data['paginator'])
    provider = ReportProvider()
    res = provider.get(data, options)
    return json.dumps(res)


@app.route('/report', methods=['POST'])
def report():
    id = request_data()
    provider = ReportProvider()
    res = provider.detail(id)
    return json.dumps(res)


@app.route('/task/steps', methods=['POST'])
def steps():
    id = request_data()
    provider = StepProvider()
    res = provider.get(id)
    return json.dumps(res)

@app.route('/stop')
def stop():
    pass


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


@click.group()
def base():
    pass


@base.command()
def start():
    register_supervisor()
    app.run(debug=True, port=PORT)


@base.command()
def stop():
    requests.post(f'http://localhost:{PORT}/shutdown')


if __name__ == '__main__':
    # files(30)
    base()
