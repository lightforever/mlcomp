import enum

class TaskStatus(enum):
    NotRan = 0
    InProgress = 1
    Failed = 2
    Success = 3

class TaskType(enum):
    Train = 0
    Infer = 1
    User = 2