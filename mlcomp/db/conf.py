import os

DB_DRIVER = os.getenv('DB_DRIVER', 'postgresql+psycopg2')
DATABASE = {
    'dbname': os.getenv('POSTGRES_DB', 'mlcomp'),
    'user': os.getenv('POSTGRES_USER', 'mlcomp'),
    'password': os.getenv('POSTGRES_PASSWORD', '12345'),
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
}
# SA_CONNECTION_STRING = 'Data Source=mlcomp.db;Version=3;'
SA_CONNECTION_STRING = f"{DB_DRIVER}://{DATABASE['user']}:{DATABASE['password']}@{DATABASE['host']}:{DATABASE['port']}/{DATABASE['dbname']}"
