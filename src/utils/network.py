import torch.nn.functional as F


def l2norm(t):
    return F.normalize(t, dim=-1)
