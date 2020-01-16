echo "pip install -r requirements.txt"
pip install -r requirements.txt

echo "pip install flake8 flake8-quotes yapf"
pip install flake8 flake8-quotes yapf

echo "pip install -U pytest pytest-xdist"
pip install -U pytest pytest-xdist

echo "pytest -v --forked --numprocesses=auto"
pytest -v --forked --numprocesses=auto