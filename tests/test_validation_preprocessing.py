import random

import pytest

from dnadiffusion.data.validation_preprocessing import combine_all_seqs


@pytest.fixture
def mock_k562_sequences(tmpdir):
    # Creating 10 sets of 200 bp sequences and saving to a file
    sequence_list = []
    for i in range(10):
        sequence_list.append("".join(random.choices("ATGC", k=200)))
    k562_sequences = tmpdir.join("K562_sequences.txt")
    with open("K562_sequences.txt", "w") as f:
        for seq in sequence_list:
            f.write(seq + "\n")
    return str(k562_sequences)


@pytest.fixture
def mock_hESCT0_sequences(tmpdir):
    # Creating 10 sets of 200 bp sequences and saving to a file
    sequence_list = []
    for i in range(10):
        sequence_list.append("".join(random.choices("ATGC", k=200)))
    hESCT0_sequences = tmpdir.join("hESCT0_sequences.txt")
    with open("hESCT0_sequences.txt", "w") as f:
        for seq in sequence_list:
            f.write(seq + "\n")
    return str(hESCT0_sequences)


@pytest.fixture
def mock_HepG2_sequences(tmpdir):
    # Creating 10 sets of 200 bp sequences and saving to a file
    sequence_list = []
    for i in range(10):
        sequence_list.append("".join(random.choices("ATGC", k=200)))
    HepG2_sequences = tmpdir.join("HepG2_sequences.txt")
    with open("HepG2_sequences.txt", "w") as f:
        for seq in sequence_list:
            f.write(seq + "\n")
    return str(HepG2_sequences)


@pytest.fixture
def mock_GM12878_sequences(tmpdir):
    # Creating 10 sets of 200 bp sequences and saving to a file
    sequence_list = []
    for i in range(10):
        sequence_list.append("".join(random.choices("ATGC", k=200)))
    GM12878_sequences = tmpdir.join("GM12878_sequences.txt")
    with open("GM12878_sequences.txt", "w") as f:
        for seq in sequence_list:
            f.write(seq + "\n")
    return str(GM12878_sequences)


def test_combine_all_seqs(mock_GM12878_sequences, mock_k562_sequences, mock_HepG2_sequences, mock_hESCT0_sequences):
    # Call function with mock sample file
    cell_list = ["GM12878", "HepG2", "hESCT0", "K562"]
    sequences = combine_all_seqs(cell_list, "tests/test_data/validation_preprocessing/df_train.txt")
    # Assert format is correct
    assert sequences["SEQUENCE"].str.len().max() == 200
