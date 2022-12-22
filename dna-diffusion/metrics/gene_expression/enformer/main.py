from enformer_inference import EnformerInference

if __name__ == "__main__":
    data_path = "abc_data/K562.PositivePredictions.txt"
    model = EnformerInference(data_path)
    one_hot_seqs = model.data.fetch_sequence()  # this is a dictionary with key being Ensembl ID|Gene Name and the value
                                                # being the one hot encoded sequence as a torch.Tensor

    for gene, seq in one_hot_seqs.items():
        output = model.forward(seq)
        print(output)
