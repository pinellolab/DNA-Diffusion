from enformer_inference import EnformerInference

if __name__ == "__main__":
    data_path = "abc_data/K562.PositivePredictions.txt"
    model = EnformerInference(data_path)
    one_hot_seqs = model.data.fetch_sequence()  # this is a dictionary with key being Ensembl ID|Gene Name and the value
                                                # being the one hot encoded sequence as a torch.Tensor

    for gene, seq in one_hot_seqs.items():
        output = model.forward(seq)
        print(output)
        output_human = output['human']
        '''
        get the shape of the output_human tensor 
        '''
        print(output_human.shape)

        # TODO 1: Import supplementary data table 2 from paper in order to get the cell types and genomic track type.

        # TODO 2: Perform sanity check on DNA diff test data see:
        # TODO 2: https://discord.com/channels/850068776544108564/1024646567833112656/1055581251483996210

