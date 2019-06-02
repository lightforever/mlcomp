from mlcomp.task.executors.catalyst.base import BaseCallback
import pickle
from mlcomp.db.models import ReportImg
from catalyst.dl.state import RunnerState
import numpy as np


class F1Callback(BaseCallback):
    def on_batch_end(self, state: RunnerState):
        if state.loader_name != 'valid':
            return

        data = self.data[state.loader_name]
        target = state.input['targets'].detach().cpu().numpy()
        pred = state.output['logits'].detach().cpu().numpy()
        data['target'].append(target)
        data['output'].append(pred)

    def on_epoch_end(self, state: RunnerState):
        target = np.vstack(self.data['valid']['target'])
        output = np.vstack(self.data['valid']['output'])
        img = self.info.plot(target, output.argmax(1))
        content = {'img': img}
        obj = ReportImg(group=self.info.name, epoch=state.epoch, task=self.task.id,
                        img=pickle.dumps(content),
                        project=self.dag.project,
                        dag=self.task.dag
                        )

        self.img_provider.add(obj)
        self.img_provider.remove_lower(self.task.id, self.info.name, state.epoch)

        super(self.__class__, self).on_epoch_end(state)
