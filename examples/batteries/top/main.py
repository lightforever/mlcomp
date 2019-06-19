import pickle

import click
import pandas as pd
import cv2
import os

@click.group()
def base():
    pass


@base.command()
def cathode_count():
    file = 'data/fold_test.csv'
    df = pd.read_csv(file)
    cathode_count = []
    image_type_by_id = pickle.load(open('data/image_type_by_id.p', 'rb'))
    for name in df['image']:
        id = name.split('-')[0]
        if not id.isnumeric():
            raise Exception(f'name = {name} id is not numeric')
        id = int(id)
        if id not in image_type_by_id:
            print(f'name = {name} id not in image_type_by_id')
        cathode_count.append(image_type_by_id.get(id, 0))
    df['cathode_count'] = cathode_count
    df.to_csv(file, index=False)

if __name__=='__main__':
    base()