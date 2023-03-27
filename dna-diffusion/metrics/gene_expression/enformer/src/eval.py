import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import pandas as pd


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()


def pearson_corr_coef(x, y, dim=1, reduce_dims=(-1,)):
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    return F.cosine_similarity(x_centered, y_centered, dim=dim).mean(dim=reduce_dims)


def scatter_evaluation():
    enf_files = os.listdir('../outputs/enformer_bedgraphs/')
    exp_files = os.listdir('../outputs/experimental_bedgraphs/')
    fig, ax = plt.subplots()
    for exp_file in exp_files:
        cell_type = exp_file.split('_')[0]
        for enf_file in enf_files:
            cell_type_enf = enf_file.split('_')[1]
            if cell_type in cell_type_enf:
                df = pd.read_csv(f'../outputs/enformer_bedgraphs/{enf_file}', sep='\t', header=None)
                pred = list(df[3].values)
                df = pd.read_csv(f'../outputs/experimental_bedgraphs/{exp_file}', sep='\t', header=None)
                target = list(df[3].values)
                ax.scatter(pred, target, label=f"{cell_type}", alpha=0.5)

    ax.set_xlabel('Enformer prediction')
    ax.set_ylabel('DNAse experimental signal')
    ax.legend()
    plt.show()
