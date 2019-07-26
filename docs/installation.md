### Development computer

```bash
pip install mlcomp
```

### Server

- Run databases

```bash
git clone git@github.com:lightforever/mlcomp.git

cd mlcomp

docker-compose -f docker/db-compose.yml up -d
```

- Install mlcomp

```bash
pip install mlcomp
```

- Run mlcomp server

```bash
mlcomp-server start
```

### Worker

**Common preparations**:
- Create pub/private keys

```.bash
ssh-keygen -f ~/id_rsa
```

You will see this text:
```Generating public/private rsa key pair.
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
```

Press Enter two times.

As a results, two files will be created: ~/id_rsa and ~/id_rsa.pub


**In each worker-computer**:

- Create service folders

```bash
# the library folders
sudo mkdir /opt/mlcomp/
sudo mkdir /opt/mlcomp/data
sudo mkdir /opt/mlcomp/models
sudo mkdir /opt/mlcomp/tasks

# user 'mlcomp' must have the rights for reading/writing
sudo chmod 777 -R /opt/mlcomp/data
sudo chmod 777 -R /opt/mlcomp/models
sudo chmod 777 -R /opt/mlcomp/tasks
```

- Install supervisor

```bash
sudo apt-get install supervisor
```

- Install rsync

```bash
sudo apt-get install rsync
```

- Install NVIDIA/apex:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

- Create a user

```.bash
sudo adduser mlcomp
```

- Add the keys to ~/.ssh

```bash
su - mlcomp
mdkir .ssh
nano authorized_keys
# Add mlcomp pub key(id_rsa.pub) here and save

nano id_rsa
# Add mlcomp private key(id_rsa) here and save
```

- Install mlcomp

```bash
pip install mlcomp
```

- Run mlcomp worker

```bash
mlcomp-worker start
```