import numpy as np

from torch import softmax
from torch.jit import load
from torch.utils.data import DataLoader

from mlcomp.worker.interfaces.base import Interface


@Interface.register
class Torch(Interface):
    def __init__(self, file: str, batch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = load(file)
        self.batch_size = batch_size

    def __call__(self, x: dict) -> dict:
        loader = DataLoader(
            x['dataset'], batch_size=self.batch_size, shuffle=False
        )
        pred = []
        for batch in loader:
            logits = self.model(batch['features'])
            p = softmax(logits, 1).detach().cpu().numpy()
            pred.append(p)
        pred = np.vstack(pred)
        return {'prob': pred}
