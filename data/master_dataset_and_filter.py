# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "biopython",
#     "gdown",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "requests",
# ]
# ///

import argparse
import gzip
import os
import shutil
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
import requests
from Bio import SeqIO


class DataSource:
    """Base class for data sources."""

    def __init__(self, data, filepath):
        self.raw_data = data
        self.filepath = filepath

    @property
    def data(self):
        return self.raw_data


class ReferenceGenome(DataSource):
    """Object for quickly loading and querying the reference genome."""

    @classmethod
    def from_path(cls, path):
        genome_dict = {record.id: str(record.seq).upper() for record in SeqIO.parse(path, "fasta")}
        return cls(genome_dict, path)

    @classmethod
    def from_dict(cls, data_dict):
        return cls(data_dict, filepath=None)

    @property
    def genome(self):
        return self.data

    def sequence(self, chrom, start, end):
        chrom_sequence = self.genome[chrom]

        assert end < len(chrom_sequence), (
            f"Sequence position bound out of range for chromosome {chrom}. "
            f"{chrom} length {len(chrom_sequence)}, requested position {end}."
        )
        return chrom_sequence[start:end]


class FilteringData:
    """Class for filtering exclusive peaks between replicates."""

    def __init__(self, df: pd.DataFrame, cell_list: list):
        self.df = df
        self.cell_list = cell_list
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
        df_subset = self.df[subset_cols].copy()
        # Creating a new column for each cell type with the exclusive peaks or 'NO_TAG' if not exclusive
        df_subset["TAG"] = df_subset[self.cell_list].apply(lambda x: "NO_TAG" if x.sum() != 1 else x.idxmax(), axis=1)

        # Creating a new dataframe with only the rows with exclusive peaks
        new_df_list = []
        for k, v in df_subset.groupby("TAG"):
            v = v.copy()
            if k != "NO_TAG":
                cell, replicate = "_".join(k.split("_")[:-1]), k.split("_")[-1]
                v["additional_replicates_with_peak"] = (
                    self.df[self.df.filter(like=cell).columns].apply(lambda x: x.sum(), axis=1).loc[v.index] - 1
                )
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

        return new_df


def download_file(url: str, filename: str, force_download: bool = False):
    """Download a file from a URL if it doesn't exist."""
    if os.path.exists(filename) and not force_download:
        print(f"File {filename} already exists, skipping download")
        return

    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename}")


def decompress_gz_file(gz_filename: str, force_decompress: bool = False):
    """Decompress a gzip file."""
    output_filename = gz_filename[:-3]  # Remove .gz extension

    if os.path.exists(output_filename) and not force_decompress:
        print(f"File {output_filename} already exists, skipping decompression")
        return output_filename

    print(f"Decompressing {gz_filename}...")
    with gzip.open(gz_filename, "rb") as f_in:
        with open(output_filename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Decompressed to {output_filename}")
    return output_filename


def sequence_bounds(summit: int, start: int, end: int, length: int):
    """Calculate the sequence coordinates (bounds) for a given DHS."""
    half = length // 2

    if (summit - start) < half:
        return start, start + length
    elif (end - summit) < half:
        return end - length, end

    return summit - half, summit + half


def add_sequence_column(df: pd.DataFrame, genome, length: int):
    """
    Query the reference genome for each DHS and add the raw sequences
    to the dataframe.
    """
    seqs = []
    for _, row in df.iterrows():
        l, r = sequence_bounds(row["summit"], row["start"], row["end"], length)
        seq = genome.sequence(row["seqname"], l, r)
        seqs.append(seq)

    df["sequence"] = seqs
    return df


def create_master_dataset(data_dir: Path = Path("./data"), force_download: bool = False):
    """Create the master dataset by downloading and processing all required files."""
    os.makedirs(data_dir, exist_ok=True)

    # Define component columns
    COMPONENT_COLUMNS = [f"C{i}" for i in range(1, 17)]

    # Step 1: Download and process genome
    print("\n=== Step 1: Downloading and processing genome ===")
    genome_gz = data_dir / "hg38.fa.gz"
    genome_path = data_dir / "hg38.fa"

    if not genome_path.exists() or force_download:
        download_file(
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz", str(genome_gz), force_download
        )
        decompress_gz_file(str(genome_gz), force_download)

    print("Loading genome...")
    genome = ReferenceGenome.from_path(str(genome_path))

    # Step 2: Download and process metadata
    print("\n=== Step 2: Downloading and processing metadata ===")
    metadata_file = data_dir / "DHS_Index_and_Vocabulary_metadata.tsv"
    download_file(
        "https://www.meuleman.org/DHS_Index_and_Vocabulary_metadata.tsv", str(metadata_file), force_download
    )

    DHS_metadata = pd.read_table(str(metadata_file)).iloc[:-1]  # Last row is empty

    # Step 3: Download and process basis array
    print("\n=== Step 3: Downloading and processing basis array ===")
    basis_gz = data_dir / "2018-06-08NC16_NNDSVD_Basis.npy.gz"
    basis_npy = data_dir / "2018-06-08NC16_NNDSVD_Basis.npy"

    if not basis_npy.exists() or force_download:
        download_file(
            "https://zenodo.org/record/3838751/files/2018-06-08NC16_NNDSVD_Basis.npy.gz?download=1",
            str(basis_gz),
            force_download,
        )
        decompress_gz_file(str(basis_gz), force_download)

    basis_array = np.load(str(basis_npy))
    nmf_loadings = pd.DataFrame(basis_array, columns=COMPONENT_COLUMNS)
    DHS_metadata = pd.concat([DHS_metadata, nmf_loadings], axis=1)
    DHS_metadata["component"] = (
        DHS_metadata[COMPONENT_COLUMNS].idxmax(axis=1).apply(lambda x: int(x[1:]))
    )

    # Step 4: Download and process mixture array
    print("\n=== Step 4: Downloading and processing mixture array (this may take 10+ minutes) ===")
    mixture_gz = data_dir / "2018-06-08NC16_NNDSVD_Mixture.npy.gz"
    mixture_npy = data_dir / "2018-06-08NC16_NNDSVD_Mixture.npy"

    if not mixture_npy.exists() or force_download:
        download_file(
            "https://zenodo.org/record/3838751/files/2018-06-08NC16_NNDSVD_Mixture.npy.gz?download=1",
            str(mixture_gz),
            force_download,
        )
        decompress_gz_file(str(mixture_gz), force_download)

    print("Loading mixture array...")
    mixture_array = np.load(str(mixture_npy)).T
    nmf_loadings = pd.DataFrame(mixture_array, columns=COMPONENT_COLUMNS)

    # Step 5: Download and process sequence metadata
    print("\n=== Step 5: Downloading and processing sequence metadata ===")
    seq_metadata_gz = data_dir / "DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz"
    seq_metadata_file = data_dir / "DHS_Index_and_Vocabulary_hg38_WM20190703.txt"

    if not seq_metadata_file.exists() or force_download:
        download_file(
            "https://www.meuleman.org/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz",
            str(seq_metadata_gz),
            force_download,
        )
        decompress_gz_file(str(seq_metadata_gz), force_download)

    print("Loading sequence metadata...")
    sequence_metadata = pd.read_table(str(seq_metadata_file), sep="\t")
    sequence_metadata = sequence_metadata.drop(columns=["component"], axis=1)

    # Join metadata with component presence matrix
    df = pd.concat([sequence_metadata, nmf_loadings], axis=1, sort=False)

    # Add additional columns
    df["component"] = df[COMPONENT_COLUMNS].idxmax(axis=1).apply(lambda x: int(x[1:]))
    df["proportion"] = df[COMPONENT_COLUMNS].max(axis=1) / df[COMPONENT_COLUMNS].sum(axis=1)
    df["total_signal"] = df["mean_signal"] * df["numsamples"]
    df["dhs_id"] = df[["seqname", "start", "end", "summit"]].apply(lambda x: "_".join(map(str, x)), axis=1)
    df["DHS_width"] = df["end"] - df["start"]

    # Add sequences
    print("Adding sequences from genome...")
    df = add_sequence_column(df, genome, 200)

    # Rename and reorder columns
    df = df.rename(columns={"seqname": "chr"})
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

    # Step 6: Download and process binary peak matrix
    print("\n=== Step 6: Downloading and processing binary peak matrix ===")
    binary_gz = data_dir / "dat_bin_FDR01_hg38.txt.gz"
    binary_file = data_dir / "dat_bin_FDR01_hg38.txt"

    # Download the file from Google Drive if it doesn't exist
    if not binary_gz.exists() or force_download:
        print("Downloading binary peak matrix from Google Drive...")
        gdown.download(
            "https://drive.google.com/uc?export=download&id=1Nel7wWOWhWn40Yv7eaQFwvpMcQHBNtJ2",
            str(binary_gz),
            quiet=False
        )

    if binary_gz.exists() and not binary_file.exists():
        decompress_gz_file(str(binary_gz), force_download)

    print("Loading binary peak matrix...")
    binary_matrix = pd.read_table(str(binary_file), header=None)

    # Create column names
    celltype_encodeID = [
        row["Biosample name"] + "_" + row["DCC Library ID"] for _, row in DHS_metadata.iterrows()
    ]
    binary_matrix.columns = celltype_encodeID

    # Create master dataset
    print("Creating master dataset...")
    master_dataset = pd.concat([df, binary_matrix], axis=1, sort=False)

    # Save as feather file
    output_file = data_dir / "master_dataset.ftr"
    print(f"Saving master dataset to {output_file}...")
    master_dataset.to_feather(str(output_file))

    return master_dataset, DHS_metadata


def filter_dataset(
    master_dataset_path: Path = Path("./data/master_dataset.ftr"),
    cell_list: list = None,
    output_path: Path = Path("./data/filtered_dataset.txt"),
    sort: bool = True,
    balance: bool = True,
):
    """Filter the master dataset for exclusive peaks between replicates."""
    if cell_list is None:
        cell_list = ["K562_ENCLB843GMH", "hESCT0_ENCLB449ZZZ", "HepG2_ENCLB029COU", "GM12878_ENCLB441ZZZ"]

    print(f"\nLoading master dataset from {master_dataset_path}...")
    df = pd.read_feather(str(master_dataset_path))

    print(f"\nFiltering for cell types: {cell_list}")
    filter_obj = FilteringData(df, cell_list)
    filtered_df = filter_obj.filter_exclusive_replicates(sort=sort, balance=balance).reset_index(drop=True)

    print(f"\nSaving filtered dataset to {output_path}...")
    filtered_df.to_csv((str(output_path)), sep="\t", index=False)

    return filtered_df


def main():
    parser = argparse.ArgumentParser(description="Create and filter DNA diffusion master dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="Directory to store data files")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of all files")
    parser.add_argument("--create-only", action="store_true", help="Only create master dataset, don't filter")
    parser.add_argument("--filter-only", action="store_true", help="Only filter existing master dataset")
    parser.add_argument(
        "--cell-list",
        nargs="+",
        default=["K562_ENCLB843GMH", "hESCT0_ENCLB449ZZZ", "HepG2_ENCLB029COU", "GM12878_ENCLB441ZZZ"],
        help="List of cell types to filter for",
    )
    parser.add_argument("--no-sort", action="store_true", help="Don't sort filtered results")
    parser.add_argument("--no-balance", action="store_true", help="Don't balance filtered dataset")

    args = parser.parse_args()

    if not args.filter_only:
        print("Creating master dataset...")
        master_dataset, metadata = create_master_dataset(args.data_dir, args.force_download)
        if master_dataset is None:
            print("\nFailed to create master dataset. Please download binary peak matrix file manually.")
            return

    if not args.create_only:
        master_dataset_path = args.data_dir / "master_dataset.ftr"
        if not master_dataset_path.exists():
            print(f"\nError: Master dataset not found at {master_dataset_path}")
            print("Please run without --filter-only flag first to create the dataset.")
            return

        output_path = args.data_dir / "filtered_dataset.txt"
        filtered_df = filter_dataset(
            master_dataset_path,
            args.cell_list,
            output_path,
            sort=not args.no_sort,
            balance=not args.no_balance,
        )

        print(f"\nFiltered dataset shape: {filtered_df.shape}")
        print(f"Cell type distribution:")
        print(filtered_df["TAG"].value_counts())


if __name__ == "__main__":
    main()
