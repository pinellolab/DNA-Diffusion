import os
import re

import pandas as pd

from dnadiffusion import DATA_DIR
from dnadiffusion.utils.data_util import motif_composition_helper, seq_extract


def motif_composition_matrix(df_file_path: str, tag: str, cell_type: str, download_data: bool = False) -> pd.DataFrame:
    """Given an input df file path, tag, and cell type, return a matrix of motif counts for specified tag/cell type.

    Args:
        df_file_path (str): Path to the input txt file that a series of sequences across various tags
        tag (str): Tag for the sequence file. Possible options for our dataset are
            "GENERATED", "PROMOTERS", "RANDOM_GENOME_REGIONS", "test", "training", "validation".
        cell_type (str): Cell type for the sequence file. Possible options for our dataset are
            "GM12878", "HepG2", "K562", "hESCT0", "NO"

    Returns:
        pd.Dataframe: Matrix of motif counts.
    """
    if download_data:
        # Download JASPAR2020_vertebrates.pfm
        print("Downloading JASPAR2020_vertebrates.pfm...")
        os.system(
            f"wget 'https://raw.githubusercontent.com/vanheeringen-lab/gimmemotifs/master/data/motif_databases/JASPAR2020_vertebrates.pfm' - O {DATA_DIR}/JASPAR2020_vertebrates.pfm"
        )

    # Subselect desired tag/cell type from the dataframe
    main_df = seq_extract(df_file_path, tag, cell_type)

    # Extract motifs from sequence file
    df_motifs = motif_composition_helper(main_df)
    motifs = []
    with open(f"{DATA_DIR}/JASPAR2020_vertebrates.pfm") as f:
        for line in f:
            if re.match(">", line):
                motif = line.strip().replace(">", "")
                motifs.append(motif)

    # Sorting motifs
    motifs = sorted(motifs)
    motifs_dict = {k: v for v, k in enumerate(motifs)}
    df_motifs["motifs_id_number"] = df_motifs["motifs"].apply(lambda x: motifs_dict[x])
    motif_count = []
    full_motif_list = df_motifs[0].unique().tolist()
    for k, v_df in df_motifs.groupby([0]):
        partial_motif_count = [0] * len(motifs_dict)
        for i in v_df["motifs_id_number"].values:
            partial_motif_count[i] = partial_motif_count[i] + 1
        full_motif_count = [k[0], *partial_motif_count]
        motif_count.append(full_motif_count)

    # Getting absence
    for x_abs in main_df["ID"]:
        if x_abs not in full_motif_list:
            partial_motif_count = [0] * len(motifs_dict)
            full_motif_count = [x_abs, *partial_motif_count]
            motif_count.append(full_motif_count)
    df_captured_motifs = pd.DataFrame(motif_count)
    df_captured_motifs.columns = ["ID", *list(motifs_dict.keys())]
    main_df = main_df.set_index("ID", drop=False)
    df_captured_motifs = df_captured_motifs.set_index("ID", drop=False)
    output_df = pd.concat(
        [main_df[[x for x in main_df.columns if x != "ID"]], df_captured_motifs.loc[main_df["ID"].values]], axis=1
    )
    return output_df
