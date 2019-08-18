# MLComp

[![Build Status](https://travis-ci.org/lightforever/mlcomp.svg?branch=master)](https://travis-ci.org/lightforever/mlcomp) 
[![License](https://img.shields.io/github/license/lightforever/mlcomp.svg)](LICENSE)
[![Pipi version](https://img.shields.io/pypi/v/mlcomp.svg)](https://pypi.org/project/mlcomp/)
[![Docs](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fmlcomp%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://lightforever.github.io/mlcomp/index.html)

MLComp is a distributed DAG (Directed acyclic graph) framework for machine learning with UI.

The goal of MLComp is to provide tools for training, inference, creating complex pipelines
(especially for computer vision) in a rapid, well manageable, way.

MLComp is compatible with: Python 3.6+, Unix operation system.

![MLComp logo](https://github.com/lightforever/mlcomp/raw/master/mlcomp/server/front/src/assets/img/mlcomp_logo.jpg)

**Features**

- Amazing UI
- [Catalyst](https://github.com/catalyst-team/catalyst) support
- Distributed training
- Supervisor that controls computational resources
- Synchronization of both code and data
- Resources monitoring
- Full-functionally pause and continue on UI
- Auto control of the requirements
- Code dumping(with syntax highlight on UI)
- [Kaggle](https://www.kaggle.com/) integration
- Hierarchical logging
- Grid search
- Experiments comparison
- Customizing layouts

Contents

[Screenshots](#screenshots)

[Installation](#installation)

[Environment variables](#environment-variables)

[UI](#ui)

[Usage](#usage)

[Docs and examples](#docs-and-examples)

# Screenshots

![graph](docs/imgs/graph.jpg)

# Installation

1. Install mlcomp package

    ```bash
    pip install mlcomp
    ```

2. Setup your environment. Please consider [Environment variables](#environment-variables) section

3. Run db, redis, mlcomp-server, mlcomp-workers:

    **Variant 1: minimal(if you have 1 computer)**
    
    Run all necessary(mlcomp-server, mlcomp-workers, redis-server), it uses SQLITE:
    
    ```bash
    mlcomp-server start
    ```
   
    **Variant 2: full**
    
    a. Change your [Environment variables](#environment-variables) to use PostgreSql
    
    b. Install rsync
    
    ```.env
    sudo apt-get install rsync
    ```
   
    Ensure every computer is available by SSH protocol with IP/PORT you specified
     in the [Environment variables](#environment-variables) file
     
     rsync will perform the following commands:
     
     to upload
     ```bash
     rsync -vhru -e ' "ssh -p {target.port} -o StrictHostKeyChecking=no" \
     {folder}/ {target.user}@{target.ip}:{folder}/ --perms  --chmod=777
     ```
     to download
     
     ```.env
     rsync -vhru -e "ssh -p {source.port} -o StrictHostKeyChecking=no" \
     {source.user}@{source.ip}:{folder}/ {folder}/ --perms  --chmod=777
     ```
    
    c. To Run postgresql, redis-server, mlcomp-server, execute on your server-computer:
    
     ```bash
    cd ~/mlcomp/configs/
    docker-compose -f server-compose.yml up -d
    ```
    
    d. Run on each worker computer:
    
    ```bash
    mlcomp-worker start
    ```
   
    e. Install [apex](https://github.com/NVIDIA/apex#quick-start) for distributed learning
    
# Environment variables

The single file to setup your server/worker environment is located at ~/mlcomp/configs/.env

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
- MASTER_PORT_RANGE. distributed port range for a worker. 29500-29510 means that if
this worker will be a master in a distributed learning, it will use first free port
from this range. Ranges of different workers must be not overlapping.
- NCCL_SOCKET_IFNAME. NCCL network interface. 
You can see your network interfaces with `ifconfig` command.
 Please consider [nvidia doc](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html)
 
 # UI
 
Web site is available at http://{WEB_HOST}:{WEB_PORT}

By default, it is http://localhost:4201

The front is built with AngularJS.

In case you desire to change it, please consider [front's Readme page](mlcomp/server/front/README.md)
 
 # Usage
 
In your folder:
 ```bash
mlcomp dag PATH_TO_CONFIG.yml
```

This command copies files of the directory to the database.

Then, the server schedules the dag considering free resources. 
 
# Docs and examples
 
API documentation and an overview of the library can be
 found here [![Docs](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fmlcomp%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://lightforever.github.io/mlcomp/index.html)

In the [examples](examples/) folder of the repository, you can find advanced tutorials and MLComp best practices.