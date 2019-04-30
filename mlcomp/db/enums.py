from enum import Enum

class TaskStatus(Enum):
    NotRan = 0
    Queued = 1
    InProgress = 2
    Failed = 3
    Stopped = 4
    Success = 5

class TaskType(Enum):
    Train = 0
    Infer = 1
    User = 2

class StepStatus(Enum):
    InProgress = 0
    Failed = 1
    Stopped = 2
    Successed = 3