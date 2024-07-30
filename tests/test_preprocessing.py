import os

import pandas as pd

from dnadiffusion.data.preprocessing import FilteringData


def test_filtering_data():
    data_path = "tests/test_data/preprocessing"
    df_path = "/test_dataset.ftr"
    cell_list = [
        "K562_ENCLB843GMH",
        "hESCT0_ENCLB449ZZZ",
        "HepG2_ENCLB029COU",
        "GM12878_ENCLB441ZZZ",
    ]
    # Running filtering data and saving the output to a temporary directory
    FilteringData(data_path, df_path, cell_list).filter_exclusive_replicates(
        sort=True, balance=True
    )

    # Assert file path exists
    assert os.path.exists(
        data_path + "/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt"
    )

    # Loading the output
    df = pd.read_csv(
        data_path + "/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt",
        sep="\t",
    )

    # Checking filtering
    assert df["TAG"].unique().tolist().sort() == cell_list.sort()

    # Assert occurence of each TAG is equal
    assert len(set(df.groupby("TAG").value_counts().tolist())) == 1

    # Remove output file
    os.remove(
        data_path + "/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt"
    )
