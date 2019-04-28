from flask import Flask, request
import json
import click
import requests
from mlcomp.db.providers import *
from mlcomp.db.core import PaginatorOptions
from flask_cors import CORS
from threading import Thread
import time
from utils.logging import logger
import traceback
from task.tasks import execute

PORT = 4201

app = Flask(__name__)
CORS(app)

def supervisor():
    provider = TaskProvider()
    while True:
        try:
            time.sleep(1)
            not_ran_tasks = provider.by_status(TaskStatus.NotRan)
            logger.info(f'Found {len(not_ran_tasks)} not ran tasks')

            dep_status = provider.dependency_status(not_ran_tasks)
            for task in not_ran_tasks:
                if TaskStatus.Stopped.value in dep_status[task.id]:
                    provider.change_status(task, TaskStatus.Stopped)
                    continue

                if TaskStatus.Failed.value in dep_status[task.id]:
                    provider.change_status(task, TaskStatus.Failed)
                    continue

                status_set = set(dep_status[task.id])
                if len(status_set)!=0 and status_set!= {TaskStatus.Success.value}:
                    continue

                execute.delay(task.id)
                provider.change_status(task, TaskStatus.Queued)
        except Exception:
            logger.error(traceback.format_exc())



@app.route('/projects')
def projects():
    args = request.args
    options = PaginatorOptions(sort_column=args.get('sort_column'),
                               sort_descending=args.get('sort_descending')=='true',
                               page_number=int(args['page_number']) if args.get('page_number') else None,
                               page_size=int(args['page_size']) if args.get('page_size') else None,
                               )

    provider = ProjectProvider()
    res = provider.get(options)
    return json.dumps([r.to_dict() for r in res])


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
    supervisor()
    #Thread(target=supervisor, daemon=True).start()
    #app.run(debug=True, port=PORT)


@base.command()
def stop():
    requests.post(f'http://localhost:{PORT}/shutdown')




if __name__ == '__main__':
    base()
