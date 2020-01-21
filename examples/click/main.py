import click
import time
from tqdm import tqdm


@click.group()
def base():
    pass


@base.command()
@click.option('--count', type=int, default=1000)
def work(count: int):
    print('start')
    items = list(range(count))
    bar = tqdm(items)
    for item in bar:
        bar.set_description(f'item={item}')
        time.sleep(0.01)
    print('end')


if __name__ == '__main__':
    base()
