from flask import Flask, request
import json
import click
import requests
from db.providers import ProjectProvider
from db.core import PaginatorOptions

PORT = 4201

app = Flask(__name__)


@app.route('/projects')
def projects():
    args = request.args
    options = PaginatorOptions(sort_column=args.get('sortColumn'),
                               sort_descending=args.get('sortDescending'),
                               page_number=int(args['pageNumber']) if args.get('pageNumber') else None,
                               page_size=int(args['pageSize'])  if args.get('pageSize') else None,
                               )

    provider = ProjectProvider()
    res = provider.get(options)


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
    app.run(debug=True, port=PORT)


@base.command()
def stop():
    requests.post(f'http://localhost:{PORT}/shutdown')


if __name__ == '__main__':
    base()
