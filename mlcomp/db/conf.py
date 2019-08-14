import os

from mlcomp import DB_FOLDER

DB_TYPE = os.getenv('DB_TYPE')
if DB_TYPE == 'POSTGRESQL':
    DATABASE = {
        'dbname': os.getenv('POSTGRES_DB'),
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD'),
        'host': os.getenv('POSTGRES_HOST'),
        'port': int(os.getenv('POSTGRES_PORT')),
    }

    SA_CONNECTION_STRING = f"postgresql+psycopg2://{DATABASE['user']}:" \
        f"{DATABASE['password']}@{DATABASE['host']}:" \
        f"{DATABASE['port']}/{DATABASE['dbname']}"
elif DB_TYPE == 'SQLITE':
    SA_CONNECTION_STRING = f'sqlite:///{DB_FOLDER}/sqlite3.sqlite'
else:
    raise Exception(f'Unknown DB_TYPE = {DB_TYPE}')

__all__ = ['SA_CONNECTION_STRING', 'DB_TYPE']
