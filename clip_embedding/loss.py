import torch
import torch.nn.functional as F


def cosine_similarity(x, y):
    return F.cosine_similarity(x, y, dim=-1)


# L2 损失
def mse_loss(pred, target):
    return F.mse_loss(pred, target)


def cosine_similarity_loss(pred, target):
    return 1 - F.cosine_similarity(pred, target, dim=-1).mean()
