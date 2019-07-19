import numpy as np

from sklearn.metrics import classification_report

from mlcomp.db.report_info.item import ReportLayoutItem
from mlcomp.utils.plot import figure_to_binary, plot_classification_report


class ReportLayoutF1(ReportLayoutItem):
    def plot(self, y: np.array, pred: np.array):
        report = classification_report(y, pred)
        fig = plot_classification_report(report)
        return figure_to_binary(fig, dpi=70)


__all__ = ['ReportLayoutF1']