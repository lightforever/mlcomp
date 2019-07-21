export $(cat docker/.env | xargs)

PYTHONPATH=. python mlcomp/worker/__main__.py supervisor

PYTHONPATH=. python2 /usr/bin/supervisord -c docker/supervisord.conf 


**Test it into docker**:

Worker: 

docker build -f docker/worker -t mlcomp-worker .

docker run --net=host -v /opt/mlcomp/:/opt/mlcomp -it mlcomp-worker /bin/bash 

docker-compose -f docker/worker-compose.yml up --build

Server:

docker build -f docker/server -t mlcomp-server .

docker run --net=host -it mlcomp-server /bin/bash 

docker-compose -f docker/server-compose.yml up  --build                                                                                                                                                                             