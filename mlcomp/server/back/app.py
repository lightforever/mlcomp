from flask import Flask, request
import json
import click
import requests
from mlcomp.db.providers import *
from mlcomp.db.core import PaginatorOptions
from flask_cors import CORS
from mlcomp.server.back.supervisor import register_supervisor
import atexit

PORT = 4201

app = Flask(__name__)
CORS(app)

@app.route('/projects')
def projects():
    args = request.args
    options = PaginatorOptions(sort_column=args.get('sort_column'),
                               sort_descending=args.get('sort_descending') == 'true',
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
    register_supervisor()
    while True:
        pass
    # Thread(target=supervisor, daemon=True).start()
    # app.run(debug=True, port=PORT)


@base.command()
def stop():
    requests.post(f'http://localhost:{PORT}/shutdown')


if __name__ == '__main__':
    base()
