Installation
======================================

.. toctree::
   :maxdepth: 2

1. Install mlcomp package

    ::

        pip install mlcomp

2. Setup your environment

    A Configuration file automatically created at ~/mlcomp/configs/.env

    Environment variables:

    - ROOT_FOLDER - folder to save mlcomp files: configs, db, tasks, etc
    - TOKEN - site security token. Please change it to any string
    - DB_TYPE. Either SQLITE or POSTGRESQL
    - POSTGRES_DB. PostgreSql db name
    - POSTGRES_USER. PostgreSql user
    - POSTGRES_PASSWORD. PostgreSql password
    - POSTGRES_HOST. PostgreSql host
    - PGDATA. PostgreSql db files location
    - REDIS_HOST. Redis host
    - REDIS_PORT. Redis port
    - REDIS_PASSWORD. Redis password
    - WEB_HOST. mlcomp site host. 0.0.0.0 means it is available from everywhere
    - WEB_PORT. mlcomp site port
    - CONSOLE_LOG_LEVEL. log level for output to the console
    - DB_LOG_LEVEL. log level for output to the database
    - IP. Ip of a worker. The worker must be accessible from other workers by these IP/PORT
    - PORT. Port of a worker. The worker must be accessible from other workers by these IP/PORT (SSH protocol)
    - MASTER_PORT_RANGE. distributed port range for a worker. 29500-29510 means that if this worker will be a master in a distributed learning, it will use the first free port from this range. Ranges of different workers must be not overlapping.
    - NCCL_SOCKET_IFNAME. NCCL network interface.

3. Run db, redis, mlcomp-server, mlcomp-workers:

    **Variant 1: minimal(if you have 1 computer)**

    Run all necessary(mlcomp-server, mlcomp-workers, redis-server), it uses SQLITE:

        ::

            mlcomp-server start

    **Variant 2: full**

    a. Change your Environment variables to use PostgreSql

    b. Install rsync on each worker computer

    ::

        sudo apt-get install rsync


    Ensure every computer is available by SSH protocol with IP/PORT you specified in the .env file.

    rsync will perform the following commands:

     to upload

        ::

         rsync -vhru -e "ssh -p {target.port} -o StrictHostKeyChecking=no" \
         {folder}/ {target.user}@{target.ip}:{folder}/ --perms  --chmod=777

     to download

        ::

         rsync -vhru -e "ssh -p {source.port} -o StrictHostKeyChecking=no" \
         {source.user}@{source.ip}:{folder}/ {folder}/ --perms  --chmod=777


    c. Install `apex <https://github.com/NVIDIA/apex#quick-start>`_ for distributed learning

    d. To Run postgresql, redis-server, mlcomp-server, execute on your server-computer:

    ::

        cd ~/mlcomp/configs/
        docker-compose -f server-compose.yml up -d


    e. Run on each worker computer:

    ::

        mlcomp-worker start
