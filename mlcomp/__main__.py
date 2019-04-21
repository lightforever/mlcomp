import click
from db.providers import ProjectProvider

@click.group()
def main():
    pass


@main.command()
@click.argument('action', type=click.Choice(['start', 'stop']))
def server(action):
    print(action)

@main.command()
@click.argument('name')
def project(name):
    provider = ProjectProvider()
    provider.add(name)

@main.command()
@click.argument('--name')
def task(name):
    print(name)

if __name__ == '__main__':
    main()

