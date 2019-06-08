from mlcomp.worker.executors.catalyst.img_classify import ImgClassifyCallback
import numpy as np
import cv2
from processor import ImgProcessor


class TopClassifyCallback(ImgClassifyCallback):
    def prob(self, pred: np.array, meta: dict) -> np.array:
        proc = ImgProcessor(None, meta['cathode_count'], anode_threshold=self.info.threshold['anode'],
                            cathode_threshold=self.info.threshold['cathode'])
        proc.add_cathode_points(pred[2])
        proc.add_anode_points(pred[1])
        if len(proc.cathode_points) != proc.cathode_count or len(proc.anode_points) != proc.cathode_count + 1:
            return np.array([0, 1])

        report = proc.broken_report()

        return np.array([1-report['prob_broken'], report['prob_broken']])

    # noinspection PyUnresolvedReferences
    def target(self, target: np.array, meta: dict) -> int:
        proc = ImgProcessor(None, meta['cathode_count'])
        proc.add_cathode_points(target[2])
        proc.add_anode_points(target[1])

        if len(proc.cathode_points) != proc.cathode_count or len(proc.anode_points) != proc.cathode_count + 1:
            return 1

        report = proc.broken_report()

        return int(report['broken'])

    def img(self, input: np.array, pred: np.array, target: np.array, meta: dict):
        parts = []
        # Orig
        img = self.experiment.denormilize(input)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        parts.append(img)

        # Target
        proc = ImgProcessor(img, meta['cathode_count'])
        proc.add_cathode_points(target[2])
        proc.add_anode_points(target[1])
        proc.draw_points(proc.cathode_points, proc.cathode_color)
        proc.draw_points(proc.anode_points, proc.anode_color)
        parts.append(proc.img_draw)

        # Pred
        pred_img = np.zeros(img.shape, dtype=np.uint8)
        for l, t , color in [
            (pred[1], self.info.threshold['anode'], proc.anode_color),
            (pred[2], self.info.threshold['cathode'], proc.cathode_color)
        ]:
            pred_img[l>t] = color

        parts.append(pred_img)

        # Final
        proc = ImgProcessor(img, meta['cathode_count'], anode_threshold=self.info.threshold['anode'],
                            cathode_threshold=self.info.threshold['cathode'])
        proc.add_cathode_points(pred[2])
        proc.add_anode_points(pred[1])
        proc.draw_points(proc.cathode_points, proc.cathode_color)
        proc.draw_points(proc.anode_points, proc.anode_color)
        parts.append(proc.img_draw)

        return np.vstack(parts)

