import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.loss import CrossEntropyLoss


class RingLoss(nn.Module):
    def __init__(self,
                 type='auto',
                 loss_weight=1.0,
                 softmax_loss_weight=1.0):
        """
        :param type: type of loss ('l1', 'l2', 'auto')
        :param loss_weight: weight of loss, for 'l1' and 'l2', try with 0.01.
            For 'auto', try with 1.0.

        Source: https://github.com/Paralysis/ringloss
        """
        super().__init__()
        self.radius = Parameter(torch.Tensor(1))
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight
        self.type = type
        self.softmax = CrossEntropyLoss()
        self.softmax_loss_weight = softmax_loss_weight

    def forward(self, x, y):
        softmax = self.softmax(x, y).mul_(self.softmax_loss_weight)
        x = x.pow(2).sum(dim=1).pow(0.5)
        if self.radius.data[0] < 0:
            self.radius.data.fill_(x.mean().data)
        if self.type == 'l1':
            loss1 = F.smooth_l1_loss(x, self.radius.expand_as(x)). \
                mul_(self.loss_weight)
            loss2 = F.smooth_l1_loss(self.radius.expand_as(x), x). \
                mul_(self.loss_weight)
            ringloss = loss1 + loss2
        elif self.type == 'auto':
            diff = x.sub(self.radius.expand_as(x)) / \
                   (x.mean().detach().clamp(min=0.5))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        else:  # L2 Loss, if not specified
            diff = x.sub(self.radius.expand_as(x))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        return softmax + ringloss


__all__ = ['RingLoss']
