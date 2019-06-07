from mlcomp.task.executors.catalyst.img_classify import ImgClassifyCallback
import numpy as np


class TopClassifyCallback(ImgClassifyCallback):
    def pred_prob(self, pred: np.array) -> np.array:
        pass

    def target(self, target):
        pass

    def img(self, input: np.array, pred: np.array, target: np.array):
        return self.experiment.denormilize(input)
