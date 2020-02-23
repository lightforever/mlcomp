from typing import List

import cv2
import torch
import albumentations as A
import numpy as np
from tqdm import tqdm

from mlcomp.contrib.transform.albumentations import ChannelTranspose
from mlcomp.contrib.model import Pretrained
from mlcomp.contrib.model.timm import Timm


def infer(
        files: List[str],
        checkpoint: str,
        model=None,
        class_='Pretrained',
        variant: str = 'resnet34',
        activation: str = 'softmax',
        num_classes: int = 2,
        batch_size: int = 16,
        device: str = 'cuda',
        transforms=None):
    assert activation in ['softmax', 'sigmoid', None]

    if model is None:
        if class_ == 'Pretrained':
            model = Pretrained(variant=variant, num_classes=num_classes)
        elif class_ == 'Timm':
            model = Timm(variant=variant, num_classes=num_classes)
        else:
            raise Exception('unknown model class')
    if transforms is None:
        transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)
                        ),
            ChannelTranspose()
        ])

    checkpoint = torch.load(checkpoint,
                            map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    imgs = []
    preds = []

    def predict_batch(imgs):
        if len(imgs) == 0:
            return

        imgs_t = np.array(
            [transforms(image=img)['image'] for img in imgs]
        )
        tensor = torch.from_numpy(imgs_t).to(device)
        pred = model(tensor)
        if activation == 'softmax':
            pred = torch.softmax(pred, dim=1)
        elif activation == 'sigmoid':
            pred = torch.sigmoid(pred)
        pred = pred.detach().cpu().numpy()
        preds.extend(pred.tolist())

        imgs.clear()

    for file in tqdm(files):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

        if len(imgs) >= batch_size:
            predict_batch(imgs)

    predict_batch(imgs)
    return np.array(preds)


if __name__ == '__main__':
    print(infer([
        '/home/light/projects/deepfake/data/eyes/dfdc_train_part_49/uowiocuqqt.mp4/0.tar/0_0.jpg'
    ], checkpoint='/home/light/mlcomp/models/deepfake/best.pth'))
