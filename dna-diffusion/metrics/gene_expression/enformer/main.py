from inference import EnformerModel
from utils import inference

if __name__ == "__main__":
    data_path = "data/abc_data/K562.PositivePredictions.txt"
    model = EnformerModel(data_path)
    one_hot_seqs = model.data.fetch_sequence()  # this is a dictionary with key being Ensembl ID|Gene Name and the value
    # being the one hot encoded sequence as a torch.Tensor
    inference(one_hot_seqs, model)  # saves output to outputs folder as a pickle file

    # TODO 1: Perform sanity check on DNA diff test data see:
    # TODO 1: https://discord.com/channels/850068776544108564/1024646567833112656/1055581251483996210`
