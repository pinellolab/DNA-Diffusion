from inference import EnformerModel
from utils import inference

if __name__ == "__main__":
    data_path = "data/selected_K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt"
    model = EnformerModel(data_path)
    one_hot_seqs = model.data.fetch_sequences()
    inference(one_hot_seqs, model)  # saves output to outputs folder as a pickle file

    # TODO 1: Perform sanity check on DNA diff test data see:
    # TODO 1: https://discord.com/channels/850068776544108564/1024646567833112656/1055581251483996210`

    # TODO 2: Install pybedtools on HPC via pip after trying via conda. If this does not work, go to TODO 3.

    # TODO 3: Add pybedtools and all associated dependencies to HPC manually. Step 1: find pybedtools in local conda env
    #  Step 3: find all dependencies of pybedtools in local conda env. Step 3: add all dependencies to HPC manually.
