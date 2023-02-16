from inference import EnformerModel
from utils import inference, create_enformer_bedgraph

if __name__ == "__main__":
    data_path = "data/selected_K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt"
    enformer_bed = "data/chr1_dnase_enformer.bed"
    cell_types = ["K562", "H1-hESC", "HepG2", "GM12878"]
    assay_type = "DNASE"

    model = EnformerModel(data_path)
    one_hot_seqs = model.data.fetch_sequences()
    inference(one_hot_seqs, model)  # saves output to outputs folder as a pickle file
    create_enformer_bedgraph(enformer_bed, cell_types, assay_type)

    # TODO 1: Perform sanity check on DNA diff test data see:
    # TODO 1: https://discord.com/channels/850068776544108564/1024646567833112656/1055581251483996210`

