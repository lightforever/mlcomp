from mlcomp.server.back.app import start_server as _start_server
from mlcomp.server.back.app import stop_server as _stop_server
import click


@click.group()
def main():
    pass


@main.command()
def start():
    _start_server()


@main.command()
def stop():
    _stop_server()


if __name__ == '__main__':
    main()
