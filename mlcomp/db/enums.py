from enum import Enum

class TaskStatus(Enum):
    NotRan = 0
    InProgress = 1
    Failed = 2
    Success = 3

class TaskType(Enum):
    Train = 0
    Infer = 1
    User = 2