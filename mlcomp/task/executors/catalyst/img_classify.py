from catalyst.dl.state import RunnerState
from mlcomp.db.models import ReportImg
from scipy.special import softmax
import numpy as np
import cv2
import pickle
from mlcomp.utils.misc import adapt_db_types
from mlcomp.task.executors.catalyst.base import BaseCallback


class ImgClassifyCallback(BaseCallback):
    def on_batch_end(self, state: RunnerState):
        if state.loader_name == 'train' and not self.info.train:
            return

        save = (state.epoch+1) % self.info.epoch_every == 0 or self.is_best
        if not save:
            return

        necessary_fields = ['y_pred', 'metric_diff']
        targets = state.input['targets'].detach().cpu().numpy()
        preds = state.output['logits'].detach().cpu().numpy()
        inputs = state.input['features'].detach().cpu().numpy()

        for i in range(len(targets)):
            input = inputs[i]
            pred = preds[i]
            target = targets[i]

            target = int(target)
            if self.added[state.loader_name][target] >= self.info.count_class_max:
                continue
            prep = self.classify_prepare(input, pred, target)
            adapt_db_types(prep)
            for field in necessary_fields:
                assert prep[field] is not None, f'{field} is None after classify_prepare'

            img = cv2.imencode('.jpg', prep['img'])[1].tostring()
            content = {'img': img}
            obj = ReportImg(group=self.info.name, epoch=state.epoch, task=self.task.id,
                            img=pickle.dumps(content),
                            project=self.dag.project,
                            dag=self.task.dag,
                            y=target,
                            y_pred=prep['y_pred'],
                            metric_diff=prep['metric_diff'],
                            attr1=prep.get('attr1'),
                            attr2=prep.get('attr2'),
                            attr3=prep.get('attr3'),
                            part=state.loader_name
                            )
            self.img_provider.add(obj, commit=False)
            self.added[state.loader_name][target] += 1

        self.img_provider.commit()

    def on_epoch_end(self, state: RunnerState):
        if self.info.epoch_every is None and self.is_best:
            self.img_provider.remove_lower(self.task.id, self.info.name, state.epoch)
        super(self.__class__, self).on_epoch_end(state)

    def classify_prepare(self, input: np.array, pred: np.array, target: int):
        res = {
            'y_pred': pred.argmax(),
            'img': self.experiment.denormilize(input)
        }

        pred_soft = softmax(pred)
        res['metric_diff'] = 1 - pred_soft[target]
        return res
