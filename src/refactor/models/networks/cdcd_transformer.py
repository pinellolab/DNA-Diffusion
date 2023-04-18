from einops import rearrange
from src.refactor.utils.misc import default
from src.refactor.utils.network import CDCDTransformerEncoder, LearnedSinusoidalPosEmb
import torch
from torch import nn


class CDCDTransformer(nn.Module):
    """
    Refer to the main paper for the architecture details https://arxiv.org/pdf/2208.04202.pdf
    """

    def __init__(
            self,
            dim,
            init_dim=200,
            out_dim=4,
            channels=1,
            learned_sinusoidal_dim=18,
            num_classes=10, 
            transformer_num_layers=1,
            transformer_num_attention_heads=8, 
            transformer_hidden_dim=128,
            transformer_mlp_hidden_dim=512,
            
    ):
        super().__init__()

        self.channels = channels
        self.nucleotide_embeddings = torch.nn.Embedding(4, embedding_dim=dim)

        init_dim = default(init_dim, dim)
        input_dim = init_dim*3
        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            nn.GELU()
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        self.transformer_encoder = CDCDTransformerEncoder(
            init_dim=input_dim,
            time_dim=time_dim,
            num_layers=transformer_num_layers, 
            num_attention_heads=transformer_num_attention_heads, 
            hidden_dim=transformer_hidden_dim, 
            mlp_hidden_dim=transformer_mlp_hidden_dim,
            embed_dim=init_dim
        )

        self.linear_out = nn.Linear(init_dim, out_dim) # BS * seq_len * emb_dim -> BS * seq_len * 4


    def forward(self, x, time, classes):
        t_start = self.time_mlp(time)

        if classes is not None:
            t_start += self.label_emb(classes)

        x = self.transformer_encoder(x, t_start)
        logits = self.linear_out(x)

        logits = rearrange(logits, "b w h v -> b w v h")
        return logits
    