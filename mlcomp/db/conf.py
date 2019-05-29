import os

DB_DRIVER = os.getenv('DB_DRIVER', 'postgresql+psycopg2')
DATABASE = {
    'dbname': os.getenv('POSTGRES_DB', 'mlcomp'),
    'user': os.getenv('POSTGRES_USER', 'mlcomp'),
    'password': os.getenv('POSTGRES_PASSWORD', '12345'),
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
}
if DB_DRIVER=='sqlite':
    f = os.path.dirname(__file__)
    f = os.path.join(f, '../migration')
    f = os.path.abspath(f)
    SA_CONNECTION_STRING = f'sqlite:////{f}/mlcomp.db'
else:
    SA_CONNECTION_STRING = f"{DB_DRIVER}://{DATABASE['user']}:{DATABASE['password']}@{DATABASE['host']}:{DATABASE['port']}/{DATABASE['dbname']}"

__all__ = [
    'SA_CONNECTION_STRING'
]