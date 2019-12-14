import subprocess

from mlcomp.worker.executors import Executor


@Executor.register
class Bash(Executor):
    def __init__(self, command: str, **kwargs):
        super().__init__(**kwargs)

        self.command = command

    def work(self):
        process = subprocess.Popen(self.command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True
                                   )
        while True:
            line = process.stdout.readline()
            if not line:
                break
            self.info(line.decode().strip())

        error = []
        while True:
            line = process.stderr.readline()
            if not line:
                break
            error.append(line.decode().strip())
        if len(error) > 0:
            raise Exception('\n'.join(error))


__all__ = ['Bash']
