import os

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


__all__ = ['SA_CONNECTION_STRING']
