import importlib
import sys
from os import getcwd
from os.path import join

from mlcomp.worker.executors import Executor


@Executor.register
class Click(Executor):
    def __init__(self, module: str, command: str = None, **kwargs):
        super().__init__(**kwargs)

        self.module = module
        self.command = command

    def work(self):
        self.info('click. creating module')
        spec = importlib.util.spec_from_file_location(
            self.module,
            join(getcwd(), self.module+'.py')
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        self.info('click. module created')

        command = getattr(m, self.command)
        self.info('click. command get')

        m.tqdm = self.tqdm

        self.info('click. tqdm set')

        for p in command.params:
            if p.name in self.kwargs:
                p.default = self.kwargs[p.name]

        sys.argv = sys.argv[:1]

        self.info('click. setup finished. executing command')

        stdout = sys.stdout
        sys.stdout = self
        self.info('click. stdout set')

        command(standalone_mode=False)

        sys.stdout = stdout

        self.info('click. command finished')


__all__ = ['Click']
