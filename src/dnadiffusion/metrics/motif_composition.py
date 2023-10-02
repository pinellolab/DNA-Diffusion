import pandas as pd
import re

from dnadiffusion.utils.sample_util import extract_motifs

def motif_matrix(sequence_file_path: str, tag: str, cell_type: str) -> pd.Dataframe:
    """Given an input sequence file path, tag, and cell type, return a matrix of motif counts.

    Args:
        sequence_file_path (str): Path to the input sequence file.
        tag (str): Tag for the sequence file. Possible options for our dataset are
            "GENERATED", "PROMOTERS", "RANDOM_GENOME_REGIONS", "test", "training", "validation".
        cell_type (str): Cell type for the sequence file. Possible options for our dataset are
            "GM12878", "HepG2", "K562", "hESCT0", "NO"

    Returns:
        pd.Dataframe: Matrix of motif counts.
    """
    # Extract motifs from sequence file
    df_motifs = extract_motifs(sequence_file_path, tag, cell_type)
    motifs = []
    with open('JASPAR2020_vertebrates.pfm') as f:
        for line in f:
            if re.match('>', line):
                motif = line.strip().replace('>', '')
                motifs.append(motif)
    motifs_dict = {k:v for v,k in enumerate(motifs)}
    df_motifs["motifs_id_number"] = df_motifs["motifs"].apply(lambda x: motifs_dict[x])
    motif_count = []
    full_motif_list = df_motifs[0].unique().tolist()
    for k, v_df in df_motifs.groupby([0]):
        partial_motif_count = [0] * len(full_motif_list)
        for i in v_df["motifs_id_number"].values:
            partial_motif_count[i] += 1
        full_motif_count = [k] + partial_motif_count
        motif_count.append(full_motif_count)

    # Getting absence
    for x_abs in df_motifs["ID"]:
        if x_abs not in full_motif_list:
            partial_motif_count = [0] * len(full_motif_list)
            full_motif_count = [x_abs] + partial_motif_count
            motif_count.append(full_motif_count)
    df_captured_motifs = pd.DataFrame(motif_count)
    df_captured_motifs.columns = ["ID"] + [x for x in motifs_dict.keys()]
    df_motifs = df_motifs.set_index("ID")
    df_captured_motifs = df_captured_motifs.set_index("ID")
    output_df = pd.concat([df_motifs[[x for x in df_motifs.columns if x != "ID"]], df_captured_motifs.loc[df_motifs["ID"].values]], axis=1)
    return output_df