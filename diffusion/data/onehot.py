import pandas as pd
import numpy as np


def one_hot_encode(seq, alphabet, max_seq_len):
    """One-hot encode a sequence."""
    seq_len = len(seq)
    seq_array = np.zeros((max_seq_len, len(alphabet)))
    for i in range(seq_len):
        seq_array[i, alphabet.index(seq[i])] = 1
    return seq_array

    
def encode(seq, alphabet):
    """Encode a sequence."""
    seq_len = len(seq)
    seq_array = np.zeros(seq_len)
    for i in range(seq_len):
        seq_array[i] = alphabet.index(seq[i])
    return seq_array


if __name__ == "__main__":
    seq = ["T", "G", "C", "A"]
    alphabet = ["A", "C", "G", "T"]
    seq = ["A", "C", "G", "T"]
    one = one_hot_encode(seq, alphabet)
    print(one)
    print(one.shape)

    
    df = pd.read_csv("train_all_classifier_WM20220916_motifs.csv")
    motifs = open("unique_motifs.txt").read().splitlines()

    motif_encoded = one_hot_encode(eval(df.iloc[0]["motifs"]), motifs)
    print(motif_encoded)