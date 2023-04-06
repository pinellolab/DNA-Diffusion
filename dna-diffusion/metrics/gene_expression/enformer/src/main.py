import os

from inference import EnformerModel
from utils import inference, create_enformer_bedgraph, plot_tracks
from eval import scatter_evaluation

if __name__ == "__main__":
    data_path = "../data/selected_K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt"
    enformer_bed = "../data/chr1_dnase_enformer.bed"
    cell_types = ["K562", "H1-hESC", "HepG2", "GM12878"]
    assay_type = "DNASE"
    chromosome = "chr1"

    model = EnformerModel(data_path)
    one_hot_seqs = model.data.fetch_sequences()
    inference(one_hot_seqs, model)
    create_enformer_bedgraph(enformer_bed, cell_types, assay_type, chromosome)

    plot_tracks()
    scatter_evaluation()
