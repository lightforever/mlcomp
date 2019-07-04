from torch.optim.lr_scheduler import CosineAnnealingLR


class OneCycleCosineAnnealLR(CosineAnnealingLR):
    def __init__(self, *args, **kwargs):
        self.start_epoch = None
        self.last_epoch = None
        super().__init__(*args, **kwargs)

    def step(self, epoch=None):
        if self.last_epoch is not None:
            if self.start_epoch is None:
                self.start_epoch = self.last_epoch
                self.last_epoch = 0
                for i in range(len(self.base_lrs)):
                    self.optimizer.param_groups[i]['lr'] = self.base_lrs[0]

            if self.last_epoch >= self.T_max - 1:
                self.start_epoch = self.last_epoch
                self.last_epoch = -1
                for i in range(len(self.base_lrs)):
                    self.optimizer.param_groups[i]['lr'] = self.base_lrs[0]

        super().step(epoch)


__all__ = ['OneCycleCosineAnnealLR']
