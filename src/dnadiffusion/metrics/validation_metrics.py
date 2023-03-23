from collections import defaultdict

import numpy as np
import pandas as pd
from Bio import SeqIO
from sourmash import MinHash


def _create_pandas_series_from_a_fasta_file(fasta_file: str) -> pd.Series:
    """Create a pandas series from a fasta file. Input is a fasta file"""
    data = defaultdict(list)
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = record.seq
        data["sequence"].append(seq)
    df = pd.DataFrame.from_dict(data)


def _create_mini_hash_of_a_sequence(seq: str, minihash: MinHash) -> MinHash:
    """Create a minihash of a sequence. Input is a sequence and a minihash object"""
    for k in seq:
        minihash.add_sequence(k)
    return minihash


def _compare_two_sequences_and_return_similarity(seq: str, seq2: str, k: int, n: int) -> float:
    """Calculate similarity of two sequences. Input is 2 sequences, k size of kmer and n number of hashes"""
    mh1 = MinHash(n=n, ksize=k)
    mh2 = MinHash(n=n, ksize=k)
    mh1 = _create_mini_hash_of_a_sequence(seq, mh1)
    mh2 = _create_mini_hash_of_a_sequence(seq2, mh2)
    similarity = round(mh1.similarity(mh2), 5)
    return similarity


def average_jaccard_similarity(
    seq: any,
    seq2: any,
    number_of_hashes: int = 20000,
    k_sizes: list = [3, 7, 20],
    is_fasta=False,
) -> float:
    """Calculate average similarity of two sequences. Input is 2 sequences, k sizes of kmer, n number of hashes and a boolean to indicate
    if the input is a fasta file or not"""
    if is_fasta:
        seq = _create_pandas_series_from_a_fasta_file(seq)
        seq2 = _create_pandas_series_from_a_fasta_file(seq2)
    average_similarities = []
    sequence_1 = seq.tolist()
    sequence_2 = seq2.tolist()
    for k in k_sizes:
        similarity = _compare_two_sequences_and_return_similarity(sequence_1, sequence_2, k, number_of_hashes)
        average_similarities.append(similarity)
    average_similarities = np.array(average_similarities)
    average_similarity = round(average_similarities.mean(), 3)
    return average_similarity
