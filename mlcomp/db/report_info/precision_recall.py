import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve

from mlcomp.db.report_info.item import ReportLayoutItem
from mlcomp.utils.plot import figure_to_binary


class ReportLayoutPrecisionRecall(ReportLayoutItem):
    def plot(self, y: np.array, pred: np.array):
        p, r, t = precision_recall_curve(y, pred)
        fig, ax = plt.subplots(figsize=(4.2, 2.7))
        ax2 = ax.twinx()

        t = np.hstack([t, t[-1]])

        ax.plot(r, p)

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax2.set_ylabel('Threashold')
        ax2.plot(r, t, c='red')
        return figure_to_binary(fig)


__all__ = ['ReportLayoutPrecisionRecall']
