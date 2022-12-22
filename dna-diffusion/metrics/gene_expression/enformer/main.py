from dataloader import EnformerDataLoader
from enformer_inference import EnformerInference
import pandas as pd
from Bio import SeqIO
import os

if __name__ == "__main__":
    data_path = "abc_data/K562.PositivePredictions.txt"
    model = EnformerInference(data_path)
    # data = EnformerDataLoader(pd.read_csv("abc_data/K562.PositivePredictions.txt", sep="\t"))
    one_hot_seqs = model.data.fetch_sequence()  # this is a dictionary with key being Ensembl ID|Gene Name and the value
                                                # being the one hot encoded sequence as a torch.Tensor

    for gene, seq in one_hot_seqs.items():
        output = model.forward(seq)
        print(output)
