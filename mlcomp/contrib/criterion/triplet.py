import torch
import torch.nn.functional as F
from catalyst.contrib.nn.criterion.functional import cosine_distance, \
    batch_all, _EPS


def triplet_loss(
        embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3,
        reduction='mean'
) -> torch.Tensor:
    cosine_dists = cosine_distance(embeddings)
    mask = batch_all(labels)

    anchor_positive_dist = cosine_dists.unsqueeze(2)
    anchor_negative_dist = cosine_dists.unsqueeze(1)
    triplet_loss_value = \
        F.relu(anchor_positive_dist - anchor_negative_dist + margin)
    triplet_loss_value = torch.mul(triplet_loss_value, mask)

    if reduction == 'mean':
        num_positive_triplets = torch.gt(
            triplet_loss_value,
            _EPS).sum().float()
        triplet_loss_value = (
                triplet_loss_value.sum() / (num_positive_triplets + _EPS)
        )
    elif reduction == 'none':
        triplet_loss_value = torch.sum(triplet_loss_value, dim=[1, 2])
    else:
        raise Exception(f'Unknown reduction scheme {reduction}')
    return triplet_loss_value
