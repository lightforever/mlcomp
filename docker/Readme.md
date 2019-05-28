**Test it into docker**:

Worker:


docker run --net=host -it mlcomp-worker /bin/bash 

PYTHONPATH=../ python __main__.py worker 0