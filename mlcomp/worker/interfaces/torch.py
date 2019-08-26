import numpy as np
from scipy.special import expit, softmax
from tqdm import tqdm

from torch.jit import load
from torch.utils.data import DataLoader

from mlcomp.worker.interfaces.base import Interface


@Interface.register
class Torch(Interface):
    def __init__(self, file: str, batch_size: int, batch_mode: bool = False,
                 use_logistic: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = load(file).cuda()

        self.batch_size = batch_size
        self.use_logistic = use_logistic
        self.batch_mode = batch_mode

    def __call__(self, x: dict) -> dict:
        loader = DataLoader(
            x['dataset'], batch_size=self.batch_size, shuffle=False
        )
        pred = []
        for batch in tqdm(loader, total=len(loader)):
            features = batch['features'].cuda()
            logits = self.model(features).detach().cpu().numpy()
            if self.use_logistic:
                p = expit(logits)
            else:
                p = softmax(logits, 1)

            if self.batch_mode:
                yield {'prob': p, 'count': len(p), **batch}
            else:
                pred.append(p)

        if self.batch_mode:
            return

        pred = np.vstack(pred)
        return {'prob': pred}
