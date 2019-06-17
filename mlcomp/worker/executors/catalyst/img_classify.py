from typing import Union

from catalyst.dl.state import RunnerState
from mlcomp.db.models import ReportImg
from scipy.special import softmax
import numpy as np
import cv2
import pickle
from mlcomp.utils.misc import adapt_db_types
from mlcomp.worker.executors.catalyst.base import BaseCallback
from sklearn.metrics import confusion_matrix
from numbers import Integral
import sys

class ImgClassifyCallback(BaseCallback):
    def on_batch_end(self, state: RunnerState):
        if state.loader_name == 'train' and not self.info.train:
            return

        save = (state.epoch + 1) % self.info.epoch_every == 0
        if not save:
            return

        necessary_fields = ['y_pred', 'metric_diff']
        targets = state.input['targets'].detach().cpu().numpy()
        preds = state.output['logits'].detach().cpu().numpy()
        inputs = state.input['features'].detach().cpu().numpy()
        metas = [{k: state.input['meta'][k][i] for k in state.input['meta']} for i in range(len(inputs))]

        data = self.data[state.loader_name]

        for i in range(len(targets)):
            input = inputs[i]
            pred = preds[i]
            target = targets[i]
            meta = metas[i]

            target_int = self.target(target, meta)
            prob = self.prob(pred, meta)

            data['target'].append(target_int)
            data['output'].append(prob)

            if self.added[state.loader_name][target_int] >= self.info.count_class_max:
                continue
            prep = self.classify_prepare(input, pred, target, prob, target_int, meta)
            adapt_db_types(prep)
            for field in necessary_fields:
                assert prep[field] is not None, f'{field} is None after classify_prepare'

            img = cv2.imencode('.jpg', prep['img'])[1].tostring()
            content = {'img': img}
            content = pickle.dumps(content)
            obj = ReportImg(group=self.info.name, epoch=state.epoch, task=self.task.id,
                            img=content,
                            project=self.dag.project,
                            dag=self.task.dag,
                            y=target_int,
                            y_pred=prep['y_pred'],
                            metric_diff=prep['metric_diff'],
                            attr1=prep.get('attr1'),
                            attr2=prep.get('attr2'),
                            attr3=prep.get('attr3'),
                            part=state.loader_name
                            )
            self.img_provider.add(obj, commit=False)
            self.added[state.loader_name][target_int] += 1

        self.img_provider.commit()

    def on_epoch_end(self, state: RunnerState):
        if self.info.epoch_every is None and self.is_best:
            self.img_provider.remove_lower(self.task.id, self.info.name, state.epoch)

        for name, value in self.data.items():
            targets = np.array(self.data[name]['target'])
            outputs = np.array(self.data[name]['output'])

            matrix = confusion_matrix(targets, outputs.argmax(1), labels=np.arange(outputs.shape[1]))

            c = {'data': matrix}
            obj = ReportImg(group=self.info.name + '_confusion', epoch=state.epoch, task=self.task.id,
                            img=pickle.dumps(c),
                            project=self.dag.project,
                            dag=self.task.dag,
                            part=name
                            )
            self.img_provider.add(obj)

        super(ImgClassifyCallback, self).on_epoch_end(state)

    def prob(self, pred: np.array, meta: dict) -> np.array:
        return softmax(pred)

    def target(self, target: Integral, meta: dict) -> int:
        assert isinstance(target, Integral), 'Target is not a number'
        return int(target)

    def img(self, input: np.array, pred: np.array, target: Union[np.array, int], meta: dict):
        return self.experiment.denormilize(input)

    def classify_prepare(self, input: np.array, pred: np.array,
                         target: Union[np.array, int], prob: np.array,
                         target_int: int, meta: dict):

        return {'y_pred': prob.argmax(),
                'img': self.img(input, pred, target, meta),
                'metric_diff': 1 - prob[target_int]}
