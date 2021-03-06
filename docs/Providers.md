Logs

```python
from mlcomp.db.providers import LogProvider
from mlcomp.db.models import Log

provider = LogProvider()
provider.commit()
for log in provider.query(Log).all():
    print('Time = ', log.time ,'Message = ', log.message)
```

Tasks

```python
from mlcomp.db.providers import TaskProvider
from mlcomp.db.models import Task

provider = TaskProvider()
provider.commit()
for task in provider.query(Task).order_by(Task.id).all():
    print('Time = ', task.started ,'Status = ', task.status)
```

Computers

```python
from mlcomp.db.providers import ComputerProvider
from mlcomp.db.models import Computer

provider = ComputerProvider()
provider.commit()
for computer in provider.query(Computer).all():
    for k, v in computer.__dict__.items():
        print(k, v)
```

Auxiliary

```python
from mlcomp.db.providers import AuxiliaryProvider
from mlcomp.db.models import Auxiliary

provider = AuxiliaryProvider()
provider.commit()
for obj in provider.query(Auxiliary).all():
    print(obj.name, obj.data)
```

To see celery task statuses:

1. set env variable CELERY_RESULT_BACKEND=your url 

2. 
```python
from celery.result import AsyncResult
res = AsyncResult("5456eec7-be49-49ee-9a09-268699b31782")
print(res.status)


```