import os

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO


def download_data(data_path: str, genome_path: str = ".local/share/genomes/hg38/hg38.fa"):
    # Download DHS metadata and load into dataframe
    os.system(
        f"wget https://www.meuleman.org/DHS_Index_and_Vocabulary_metadata.tsv -O {data_path}/DHS_Index_and_Vocabulary_metadata.tsv"
    )
    # Collect basis arrays from NMF
    basis_array = os.system(
        f"wget 'https://zenodo.org/record/3838751/files/2018-06-08NC16_NNDSVD_Basis.npy.gz?download=1' -O {data_path}/2018-06-08NC16_NNDSVD_Basis.npy.gz"
    )
    with open(f"{data_path}/2018-06-08NC16_NNDSVD_Basis.npy.gz", 'wb') as f:
        f.write(basis_array.content)
    # Extacting the gzip
    os.system(f"gzip -d {data_path}/2018-06-08NC16_NNDSVD_Basis.npy.gz")

    # Converting npy file to csv
    basis_array = np.load(f"{data_path}/2018-06-08NC16_NNDSVD_Basis.npy")
    np.savetxt(f"{data_path}/2018-06-08NC16_NNDSVD_Basis.csv", basis_array, delimiter=",")

    # Creating nmf_loadings matrix from csv
    nmf_loadings = pd.read_csv(f"{data_path}/2018-06-08NC16_NNDSVD_Basis.csv", header=None)
    nmf_loadings.columns = ['C' + str(i) for i in range(1, 17)]

    # Downloading mixture array that contains 3.5M x 16 matrix of peak presence/absence decomposed into 16 components
    mixture_array = os.system(
        f"wget 'https://zenodo.org/record/3838751/files/2018-06-08NC16_NNDSVD_Mixture.npy.gz?download=1' -O {data_path}/2018-06-08NC16_NNDSVD_Mixture.npy.gz"
    )
    with open(f"{data_path}/2018-06-08NC16_NNDSVD_Mixture.npy.gz", 'wb') as f:
        f.write(mixture_array.content)
    # Extacting the gzip
    os.system(f"gzip -d {data_path}/2018-06-08NC16_NNDSVD_Mixture.npy.gz")

    # Turning npy file into csv
    mixture_array = np.load(f"{data_path}/2018-06-08NC16_NNDSVD_Mixture.npy").T
    np.savetxt(f"{data_path}/2018-06-08NC16_NNDSVD_Mixture.csv", mixture_array, delimiter=",")

    # Loading in DHS_Index_and_Vocabulary_metadata that contains the following information:
    # seqname, start, end, identifier, mean_signal, numsaples, summit, core_start, core_end, component
    os.system(
        f"wget https://www.meuleman.org/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz - O {data_path}/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz"
    )
    os.system(f"gunzip -d {data_path}/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz")

    # Downloading binary peak presence/absence matrix
    os.system(
        f"wget 'https://dl.dropboxusercontent.com/scl/fi/kklr3u4j7fdpd9iv1la9v/dat_bin_FDR01_hg38.txt.gz?rlkey=0i8j7o75a1n893ixg1ozssnf0&dl=1' -O {data_path}/dat_bin_FDR01_hg38.txt.gz"
    )
    os.system(f"gunzip -d {data_path}/dat_bin_FDR01_hg38.txt.gz")

    print("Finished downloading data")


def create_master_dataset(
    data_path: str,
    genome_path: str,
):
    # Query the reference genome
    genome = ReferenceGenome.from_path(genome_path)
    # Redefine component columns
    component_columns = ['C' + str(i) for i in range(1, 17)]
    DHS_Index_and_Vocabulary_metadata = pd.read_table(f"{data_path}/DHS_Index_and_Vocabulary_metadata.tsv").iloc[:-1]

    # Component columns names
    component_columns = ['C' + str(i) for i in range(1, 17)]

    # Creating nmf_loadings matrix from csv
    basis_nmf_loadings = pd.read_csv('2018-06-08NC16_NNDSVD_Basis.csv', header=None)
    basis_nmf_loadings.columns = component_columns

    # Joining metadata with component presence matrix
    DHS_Index_and_Vocabulary_metadata = pd.concat([DHS_Index_and_Vocabulary_metadata, basis_nmf_loadings], axis=1)

    DHS_Index_and_Vocabulary_metadata['component'] = (
        DHS_Index_and_Vocabulary_metadata[component_columns].idxmax(axis=1).apply(lambda x: int(x[1:]))
    )

    # Loading sequence metadata
    sequence_metadata = pd.read_table(f"{data_path}/DHS_Index_and_Vocabulary_hg38_WM20190703.txt", sep='\t')

    # Dropping component column that contains associated tissue rather than component number (We will use the component number from DHS_Index_and_Vocabulary_metadata)
    sequence_metadata = sequence_metadata.drop(columns=['component'], axis=1)

    # Creating nmf_loadings matrix from csv and renaming columns
    mixture_nmf_loadings = pd.read_csv(
        f"{data_path}/2018-06-08NC16_NNDSVD_Mixture.csv", header=None, names=component_columns
    )
    # Join metadata with component presence matrix
    df = pd.concat([sequence_metadata, mixture_nmf_loadings], axis=1, sort=False)
    # Recreating some of the columns from our original dataset
    df['component'] = df[component_columns].idxmax(axis=1).apply(lambda x: int(x[1:]))
    df['proportion'] = df[component_columns].max(axis=1) / df[component_columns].sum(axis=1)
    df['total_signal'] = df['mean_signal'] * df['numsamples']
    df['proportion'] = df[component_columns].max(axis=1) / df[component_columns].sum(axis=1)
    df['dhs_id'] = df[['seqname', 'start', 'end', 'summit']].apply(lambda x: '_'.join(map(str, x)), axis=1)
    df['DHS_width'] = df['end'] - df['start']

    # Creating sequence column
    df = add_sequence_column(df, genome, 200)

    # Changing seqname column to chr
    df = df.rename(columns={'seqname': 'chr'})

    # Reordering and unselecting columns
    df = df[
        [
            'dhs_id',
            'chr',
            'start',
            'end',
            'DHS_width',
            'summit',
            'numsamples',
            'total_signal',
            'component',
            'proportion',
            'sequence',
        ]
    ]

    # Opening file
    binary_matrix = pd.read_table(f"{data_path}/dat_bin_FDR01_hg38.txt", header=None)

    # Collecting names of cells into a list with fromat celltype_encodeID
    celltype_encodeID = [
        row['Biosample name'] + "_" + row['DCC Library ID'] for _, row in DHS_Index_and_Vocabulary_metadata.iterrows()
    ]

    # Renaming columns using celltype_encodeID list
    binary_matrix.columns = celltype_encodeID

    # Concatenating binary matrix with master dataset
    master_dataset = pd.concat([df, binary_matrix], axis=1, sort=False)

    # Save as feather file
    master_dataset.to_feather(f"{data_path}/master_dataset.ftr")

    print("Finished creating master dataset")


class DataSource:
    # Sourced from https://github.com/meuleman/SynthSeqs/blob/main/make_data/source.py

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


# Functions used to create sequence column
def sequence_bounds(summit: int, start: int, end: int, length: int):
    """Calculate the sequence coordinates (bounds) for a given DHS.
    https://github.com/meuleman/SynthSeqs/blob/main/make_data/process.py
    """
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
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe of DHS annotations and NMF loadings.
    genome : ReferenceGenome(DataSource)
        A reference genome object to query for sequences.
    length : int
        Length of a DHS.

    https://github.com/meuleman/SynthSeqs/blob/main/make_data/process.py
    """
    seqs = []
    for rowi, row in df.iterrows():
        l, r = sequence_bounds(row['summit'], row['start'], row['end'], length)
        seq = genome.sequence(row['seqname'], l, r)

        seqs.append(seq)

    df['sequence'] = seqs
    return df
