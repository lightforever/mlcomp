from torch.nn import CrossEntropyLoss
from torch.nn.functional import nll_loss, log_softmax


class LabelSmoothingCrossEntropy(CrossEntropyLoss):
    def __init__(self, eps: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        nl = nll_loss(log_preds, target, reduction=self.reduction)
        return loss * self.eps / c + (1 - self.eps) * nl


__all__ = ['LabelSmoothingCrossEntropy']
