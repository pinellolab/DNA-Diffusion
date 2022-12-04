import torch
import pandas as pd
from enformer_pytorch import Enformer
from dataloader import EnformerDataLoader


class EnformerInference:
    def __init__(self, data_path: str, model_path="EleutherAI/enformer-official-rough"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.device = device
        self.model = Enformer.from_pretrained(model_path)
        self.data = EnformerDataLoader(pd.read_csv(data_path, sep="\t"))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))

    def forward(self, x):
        seq = torch.randint(0, 5, (1, 196_608))
        one_hot = seq_indices_to_one_hot(seq)

        output = self.model(one_hot)

        return output
