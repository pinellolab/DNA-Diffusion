import torch
import pandas as pd
from enformer_lucidrains_pytorch.enformer_pytorch import Enformer
from dataloader import EnformerDataloaderABC, EnformerDataloaderDNAse


class EnformerModel:
    def __init__(self, data_path: str, model_path="EleutherAI/enformer-official-rough"):
        if torch.cuda.is_available():
            print("Using NVIDIA GPU")
            device = torch.device("cuda")
        else:
            print("Using CPU")
            device = torch.device("cpu")

        self.device = device
        self.model = Enformer.from_pretrained(model_path).to(device)
        self.data = EnformerDataloaderDNAse(data_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))
