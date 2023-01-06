import torch
import torch.nn.functional as F

from enformer_pytorch import Enformer

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()

def pearson_corr_coef(x, y, dim = 1, reduce_dims = (-1,)):
    x_centered = x - x.mean(dim = dim, keepdim = True)
    y_centered = y - y.mean(dim = dim, keepdim = True)
    return F.cosine_similarity(x_centered, y_centered, dim = dim).mean(dim = reduce_dims)

if __name__ == "__main__":
    data = torch.load('enformer_lucidrains_pytorch/data/test-sample.pt', map_location=torch.device('cpu'))
    seq, target = data['sequence'], data['target']
    
    enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
    enformer.eval()

    enformer_output = enformer(seq, head='human')
    loss = poisson_loss(enformer_output, target)
    pcc = pearson_corr_coef(enformer_output, target)