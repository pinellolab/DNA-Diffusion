import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from dnadiffusion import DATA_DIR
from dnadiffusion.utils.data_util import ReferenceGenome, add_sequence_column


def preprocess_data(
    data_path: Path = DATA_DIR,
    df_path: str = "/master_dataset.ftr",
    cell_list: List = ["K562_ENCLB843GMH", "hESCT0_ENCLB449ZZZ", "HepG2_ENCLB029COU", "GM12878_ENCLB441ZZZ"],
    download_data_bool: bool = True,
    create_master_dataset_bool: bool = True,
    filter_data_bool: bool = True,
    sort_replicates: bool = True,
    balance_replicates: bool = True,
) -> None:
    """
    Preprocesses the data for the DNA diffusion project.
    """
    if download_data_bool:
        download_data(data_path)
    if create_master_dataset_bool:
        create_master_dataset(data_path)
    if filter_data_bool:
        FilteringData(data_path, df_path, cell_list).filter_exclusive_replicates(
            sort=sort_replicates, balance=balance_replicates
        )

    print("Finished preprocessing data")


def download_data(data_path: str) -> None:
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(data_path + "/tmp"):
        os.makedirs(data_path + "/tmp")

    # Creating tmp directory variable
    tmp_dir = data_path + "/tmp"

    # Downloading the reference genome
    os.system(f"wget 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz' -O {tmp_dir}/hg38.fa.gz")
    os.system(f"gunzip {tmp_dir}/hg38.fa.gz")

    # Download DHS metadata and load into dataframe
    os.system(
        f"wget 'https://www.meuleman.org/DHS_Index_and_Vocabulary_metadata.tsv' -O {tmp_dir}/DHS_Index_and_Vocabulary_metadata.tsv"
    )
    # Collect basis arrays from NMF
    basis_array = os.system(
        f"wget 'https://zenodo.org/record/3838751/files/2018-06-08NC16_NNDSVD_Basis.npy.gz?download=1' -O {tmp_dir}/2018-06-08NC16_NNDSVD_Basis.npy.gz"
    )
    # Extacting the gzip
    os.system(f"gzip -d {tmp_dir}/2018-06-08NC16_NNDSVD_Basis.npy.gz")

    # Converting npy file to csv
    basis_array = np.load(f"{tmp_dir}/2018-06-08NC16_NNDSVD_Basis.npy")
    np.savetxt(f"{tmp_dir}/2018-06-08NC16_NNDSVD_Basis.csv", basis_array, delimiter=",")

    # Creating nmf_loadings matrix from csv
    nmf_loadings = pd.read_csv(f"{tmp_dir}/2018-06-08NC16_NNDSVD_Basis.csv", header=None)
    nmf_loadings.columns = ["C" + str(i) for i in range(1, 17)]

    # Downloading mixture array that contains 3.5M x 16 matrix of peak presence/absence decomposed into 16 components
    mixture_array = os.system(
        f"wget 'https://zenodo.org/record/3838751/files/2018-06-08NC16_NNDSVD_Mixture.npy.gz?download=1' -O {tmp_dir}/2018-06-08NC16_NNDSVD_Mixture.npy.gz"
    )
    # Extacting the gzip
    os.system(f"gzip -d {tmp_dir}/2018-06-08NC16_NNDSVD_Mixture.npy.gz")

    # Turning npy file into csv
    mixture_array = np.load(f"{tmp_dir}/2018-06-08NC16_NNDSVD_Mixture.npy").T
    np.savetxt(f"{tmp_dir}/2018-06-08NC16_NNDSVD_Mixture.csv", mixture_array, delimiter=",")

    # Loading in DHS_Index_and_Vocabulary_metadata that contains the following information:
    # seqname, start, end, identifier, mean_signal, numsaples, summit, core_start, core_end, component
    os.system(
        f"wget 'https://www.meuleman.org/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz' -O {tmp_dir}/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz"
    )
    os.system(f"gunzip -d {tmp_dir}/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz")

    # Downloading binary peak presence/absence matrix
    os.system(
        f"wget 'https://dl.dropboxusercontent.com/scl/fi/kklr3u4j7fdpd9iv1la9v/dat_bin_FDR01_hg38.txt.gz?rlkey=0i8j7o75a1n893ixg1ozssnf0&dl=1' -O {tmp_dir}/dat_bin_FDR01_hg38.txt.gz"
    )
    os.system(f"gunzip -d {tmp_dir}/dat_bin_FDR01_hg38.txt.gz")

    print("Finished downloading data")


def create_master_dataset(
    data_path: str,
) -> None:
    tmp_dir = data_path + "/tmp"
    # Creating master dataset from downloaded data (takes about 10 minutes)
    print("Creating master dataset")
    # Query the reference genome
    genome = ReferenceGenome.from_path(f"{tmp_dir}/hg38.fa")
    # Redefine component columns
    component_columns = ["C" + str(i) for i in range(1, 17)]
    DHS_Index_and_Vocabulary_metadata = pd.read_table(f"{tmp_dir}/DHS_Index_and_Vocabulary_metadata.tsv").iloc[:-1]

    # Component columns names
    component_columns = ["C" + str(i) for i in range(1, 17)]

    # Creating nmf_loadings matrix from csv
    basis_nmf_loadings = pd.read_csv(f"{tmp_dir}/2018-06-08NC16_NNDSVD_Basis.csv", header=None)
    basis_nmf_loadings.columns = component_columns

    # Joining metadata with component presence matrix
    DHS_Index_and_Vocabulary_metadata = pd.concat([DHS_Index_and_Vocabulary_metadata, basis_nmf_loadings], axis=1)

    DHS_Index_and_Vocabulary_metadata["component"] = (
        DHS_Index_and_Vocabulary_metadata[component_columns].idxmax(axis=1).apply(lambda x: int(x[1:]))
    )

    # Loading sequence metadata
    sequence_metadata = pd.read_table(f"{tmp_dir}/DHS_Index_and_Vocabulary_hg38_WM20190703.txt", sep="\t")

    # Dropping component column that contains associated tissue rather than component number (We will use the component number from DHS_Index_and_Vocabulary_metadata)
    sequence_metadata = sequence_metadata.drop(columns=["component"], axis=1)

    # Creating nmf_loadings matrix from csv and renaming columns
    mixture_nmf_loadings = pd.read_csv(
        f"{tmp_dir}/2018-06-08NC16_NNDSVD_Mixture.csv", header=None, names=component_columns
    )
    # Join metadata with component presence matrix
    df = pd.concat([sequence_metadata, mixture_nmf_loadings], axis=1, sort=False)
    # Recreating some of the columns from our original dataset
    df["component"] = df[component_columns].idxmax(axis=1).apply(lambda x: int(x[1:]))
    df["proportion"] = df[component_columns].max(axis=1) / df[component_columns].sum(axis=1)
    df["total_signal"] = df["mean_signal"] * df["numsamples"]
    df["proportion"] = df[component_columns].max(axis=1) / df[component_columns].sum(axis=1)
    df["dhs_id"] = df[["seqname", "start", "end", "summit"]].apply(lambda x: "_".join(map(str, x)), axis=1)
    df["DHS_width"] = df["end"] - df["start"]

    # Creating sequence column
    df = add_sequence_column(df, genome, 200)

    # Changing seqname column to chr
    df = df.rename(columns={"seqname": "chr"})

    # Reordering and unselecting columns
    df = df[
        [
            "dhs_id",
            "chr",
            "start",
            "end",
            "DHS_width",
            "summit",
            "numsamples",
            "total_signal",
            "component",
            "proportion",
            "sequence",
        ]
    ]

    # Opening file
    binary_matrix = pd.read_table(f"{tmp_dir}/dat_bin_FDR01_hg38.txt", header=None)

    # Collecting names of cells into a list with fromat celltype_encodeID
    celltype_encodeID = [
        row["Biosample name"] + "_" + row["DCC Library ID"] for _, row in DHS_Index_and_Vocabulary_metadata.iterrows()
    ]

    # Renaming columns using celltype_encodeID list
    binary_matrix.columns = celltype_encodeID

    # Concatenating binary matrix with master dataset
    master_dataset = pd.concat([df, binary_matrix], axis=1, sort=False)

    # Save as feather file
    master_dataset.to_feather(f"{data_path}/master_dataset.ftr")

    # Remove tmp files folder
    shutil.rmtree(tmp_dir)

    print("Finished creating master dataset")


class FilteringData:
    def __init__(self, data_path: str, df_path: str, cell_list: list):
        self.data_path = data_path
        self.df = pd.read_feather(data_path + df_path)
        self.cell_list = cell_list
        self.output_path = data_path + "/test_K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt"
        self._test_data_structure()

    def _test_data_structure(self):
        # Ensures all columns after the 11th are named cell names
        assert all("_ENCL" in x for x in self.df.columns[11:]), "_ENCL not in all columns after 11th"

    def filter_exclusive_replicates(self, sort: bool = False, balance: bool = True):
        """Given a specific set of samples (one per cell type),
        capture the exclusive peaks of each samples (the ones matching just one sample for the whole set)
        and then filter the dataset to keep only these peaks.

        Returns:
            pd.DataFrame: The original dataframe plus a column for each cell type with the exclusive peaks
        """
        print("Filtering exclusive peaks between replicates")
        # Selecting the columns corresponding to the cell types
        subset_cols = self.df.columns[:11].tolist() + self.cell_list
        # Creating a new dataframe with only the columns corresponding to the cell types
        df_subset = self.df[subset_cols]
        # Creating a new column for each cell type with the exclusive peaks or 'NO_TAG' if not exclusive
        df_subset["TAG"] = df_subset[self.cell_list].apply(lambda x: "NO_TAG" if x.sum() != 1 else x.idxmax(), axis=1)

        # Creating a new dataframe with only the rows with exclusive peaks
        new_df_list = []
        for k, v in df_subset.groupby("TAG"):
            if k != "NO_TAG":
                cell, replicate = "_".join(k.split("_")[:-1]), k.split("_")[-1]
                v["additional_replicates_with_peak"] = (
                    self.df[self.df.filter(like=cell).columns].apply(lambda x: x.sum(), axis=1) - 1
                )
                temp_df = self.df.filter(like=cell)
                print(f"Cell type: {cell}, Replicate: {replicate}, Number of exclusive peaks: {v.shape[0]}")
            else:
                v["additional_replicates_with_peak"] = 0
            new_df_list.append(v)
        new_df = pd.concat(new_df_list).sort_index()
        new_df["other_samples_with_peak_not_considering_reps"] = (
            new_df["numsamples"] - new_df["additional_replicates_with_peak"] - 1
        )

        # Sorting the dataframe by the number of samples with the peak
        if sort:
            new_df = pd.concat(
                [
                    x_v.sort_values(
                        by=["additional_replicates_with_peak", "other_samples_with_peak_not_considering_reps"],
                        ascending=[False, True],
                    )
                    for x_k, x_v in new_df.groupby("TAG")
                ],
                ignore_index=True,
            )

        # Balancing the dataset
        if balance:
            lowest_peak_count = new_df.groupby("TAG").count()["sequence"].min()
            new_df = pd.concat(
                [v_bal.head(lowest_peak_count) for k_bal, v_bal in new_df.groupby("TAG") if k_bal != "NO_TAG"]
            )

        print("Saving filtered dataset")
        new_df.to_csv(self.output_path, sep="\t", index=False)


if __name__ == "__main__":
    preprocess_data()
