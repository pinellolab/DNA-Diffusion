from typing import List

import torch
import torch.nn.functional as F

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()

def pearson_corr_coef(x, y, dim = 1, reduce_dims = (-1,)):
    x_centered = x - x.mean(dim = dim, keepdim = True)
    y_centered = y - y.mean(dim = dim, keepdim = True)
    return F.cosine_similarity(x_centered, y_centered, dim = dim).mean(dim = reduce_dims)

def compute_metrics(model, data, track_idx: List[int]):
    model.eval()

    seq, target = data['sequence'], data['target']
    with torch.no_grad():
        pred = model(seq, head='human')

    if track_idx:
        pred = pred[:, track_idx]
        target = target[:, track_idx]

    pl = poisson_loss(pred.T, target.T)
    pcc = pearson_corr_coef(pred.T, target.T)

    return pl, pcc

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from enformer_pytorch import Enformer

    data = torch.load('enformer_lucidrains_pytorch/data/test-sample.pt', map_location=torch.device('cpu'))
    enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
    import pdb; pdb.set_trace()
    pl, pcc = compute_metrics(enformer, data, track_idx=[0, 2])