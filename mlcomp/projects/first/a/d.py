from c import report

from mlcomp.task.executors import Download, Executor

@Executor.register
class Download2(Download):
    pass

report('d')