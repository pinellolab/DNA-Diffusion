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
    seq_array = np.zeros(len(alphabet))
    for i in range(seq_len):
        seq_array[alphabet.index(seq[i])] = 1
    
    return seq_array


if __name__ == "__main__":
    alphabet = ["A", "C", "T", "G"]
    
    df = pd.read_csv("train_all_classifier_WM20220916.csv", sep="\t")

    x_train_seq = np.array([one_hot_encode(x, alphabet, 200) for x in df['raw_sequence'] if 'N' not in x ])

    print(x_train_seq.shape)
