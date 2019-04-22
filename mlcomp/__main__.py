import click
from mlcomp.db.providers import *
from mlcomp.db.enums import *
import os
from mlcomp.task.storage import Storage

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
@click.option('--name')
@click.option('--project', type=int)
@click.option('--parent_task', type=int)
@click.option('--computer', type=int)
@click.option('--gpu', type=int, default=0)
@click.option('--cpu', type=int, default=0)
@click.option('--type', type=int, default=0)
@click.option('--folder', type=str, default='./')
def task(name: str, project: int, parent_task: int, computer: int, gpu: int, cpu: int, type: int):
    provider = TaskProvider()
    task = Task(name=name, project=project, parent_task=parent_task, computer=computer,
                gpu=gpu, cpu=cpu, type=type, status=TaskStatus.NotRan.value
                )
    provider.add(task)

    folder = os.path.join(os.getcwd(), folder)
    Storage.upload(folder)



if __name__ == '__main__':
    main()
