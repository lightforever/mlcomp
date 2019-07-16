import pickle

import numpy as np

from scipy.special import softmax

from catalyst.dl import RunnerState

from mlcomp.worker.executors.catalyst.base import BaseCallback
from mlcomp.db.models import ReportImg


class PrecisionRecallCallback(BaseCallback):
    def on_batch_end(self, state: RunnerState):
        if state.loader_name != 'valid':
            return

        data = self.data[state.loader_name]
        target = state.input['targets'].detach().cpu().numpy()
        pred = state.output['logits'].detach().cpu().numpy()
        data['target'].append(target)
        data['output'].append(pred)

    def on_epoch_end(self, state: RunnerState):
        target = np.hstack(self.data['valid']['target'])
        output = np.vstack(self.data['valid']['output'])

        output_soft = softmax(output, axis=1)
        img = self.info.plot(target, output_soft[:,1])
        content = {'img': img}
        content = pickle.dumps(content)
        obj = ReportImg(group=self.info.name, epoch=state.epoch,
                        task=self.task.id, img=content,
                        project=self.dag.project,
                        dag=self.task.dag,
                        part=state.loader_name
                        )

        self.img_provider.add(obj)
        self.img_provider.remove_lower(self.task.id,
                                       self.info.name,
                                       state.epoch)

        super(self.__class__, self).on_epoch_end(state)
