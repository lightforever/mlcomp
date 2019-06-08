from c import report

from mlcomp.worker.executors import Download, Executor

@Executor.register
class Download2(Download):
    pass

report('d')