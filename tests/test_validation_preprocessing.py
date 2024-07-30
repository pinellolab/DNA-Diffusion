import os
import random
import tempfile

import pytest

from dnadiffusion.data.validation_preprocessing import combine_all_seqs


@pytest.fixture
def mock_k562_sequences():
    # Creating 10 sets of 200 bp sequences and saving to a file
    sequence_list = []
    for i in range(10):
        sequence_list.append("".join(random.choices("ATGC", k=200)))
    temp_file = tempfile.NamedTemporaryFile(
        prefix="K562_", delete=False, dir=os.getcwd()
    )
    temp_path = temp_file.name
    with open(temp_path, "w") as f:
        for seq in sequence_list:
            f.write(seq + "\n")
    return temp_path


@pytest.fixture
def mock_hESCT0_sequences(tmpdir):
    # Creating 10 sets of 200 bp sequences and saving to a file
    sequence_list = []
    for i in range(10):
        sequence_list.append("".join(random.choices("ATGC", k=200)))
    temp_file = tempfile.NamedTemporaryFile(
        prefix="hESCT0_", delete=False, dir=os.getcwd()
    )
    temp_path = temp_file.name
    with open(temp_path, "w") as f:
        for seq in sequence_list:
            f.write(seq + "\n")
    return temp_path


@pytest.fixture
def mock_HepG2_sequences(tmpdir):
    # Creating 10 sets of 200 bp sequences and saving to a file
    sequence_list = []
    for i in range(10):
        sequence_list.append("".join(random.choices("ATGC", k=200)))
    temp_file = tempfile.NamedTemporaryFile(
        prefix="HepG2_", delete=False, dir=os.getcwd()
    )
    temp_path = temp_file.name
    with open(temp_path, "w") as f:
        for seq in sequence_list:
            f.write(seq + "\n")
    print(temp_path)
    return temp_path


@pytest.fixture
def mock_GM12878_sequences(tmpdir):
    # Creating 10 sets of 200 bp sequences and saving to a file
    sequence_list = []
    for i in range(10):
        sequence_list.append("".join(random.choices("ATGC", k=200)))
    temp_file = tempfile.NamedTemporaryFile(
        prefix="GM12878_", delete=False, dir=os.getcwd()
    )
    temp_path = temp_file.name
    with open(temp_path, "w") as f:
        for seq in sequence_list:
            f.write(seq + "\n")
    return temp_path


def test_combine_all_seqs(
    mock_GM12878_sequences,
    mock_k562_sequences,
    mock_HepG2_sequences,
    mock_hESCT0_sequences,
):
    # Call function with mock sample file
    cell_list = ["GM12878", "HepG2", "hESCT0", "K562"]
    sequences = combine_all_seqs(
        cell_list, "tests/test_data/validation_preprocessing/df_train.txt"
    )
    # Assert format is correct
    assert sequences["SEQUENCE"].str.len().max() == 200

    # Remove temp files
    os.remove(mock_k562_sequences)
    os.remove(mock_hESCT0_sequences)
    os.remove(mock_HepG2_sequences)
    os.remove(mock_GM12878_sequences)
