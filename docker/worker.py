import os
from multiprocessing import cpu_count

pr = os.getenv('CPU', cpu_count())
text = [
    '[supervisord]',
    'nodaemon=true',
    '',
    '[supervisor]',
    'command=python __main__.py worker-supervisor'
]
for p in range(pr):
    text.append(f'[program:worker{p}]')
    text.append(f'command=python __main__.py worker {p}')
    text.append('autostart=true')
    text.append('autorestart=true')
    text.append('')

with open('supervisord.conf', 'w') as f:
    f.writelines('\n'.join(text))