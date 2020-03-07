pip install -r requirements.txt

pip install flake8 flake8-quotes yapf

pip install -U pytest pytest-xdist

pytest -v --forked --numprocesses=auto