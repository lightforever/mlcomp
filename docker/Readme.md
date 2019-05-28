**Test it into docker**:

Worker:

docker build -f docker/worker-dev -t mlcomp-worker .

docker run --net=host -v /opt/mlcomp/:/opt/mlcomp -it mlcomp-worker /bin/bash 

PYTHONPATH=../ python __main__.py worker 0


Server:

docker build -f docker/server-dev -t mlcomp-server .

docker run --net=host -p 4201:4201 -it mlcomp-server /bin/bash 

PYTHONPATH=../ python __main__.py start-server