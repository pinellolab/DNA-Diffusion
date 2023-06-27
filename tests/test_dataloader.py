import numpy as np
import pytest
import torch

from dnadiffusion.data.dataloader import SequenceDataset


@pytest.fixture
def sequence_data():
    test_seqs = np.array(
        [
            [
                [-1, 1, -1, -1],
                [-1, -1, -1, 1],
                [-1, -1, 1, -1],
                [-1, -1, -1, 1],
                [-1, 1, -1, -1],
                [-1, -1, -1, 1],
                [-1, -1, 1, -1],
            ],
            [
                [-1, -1, 1, -1],
                [-1, -1, -1, 1],
                [-1, 1, -1, -1],
                [-1, -1, -1, 1],
                [-1, -1, 1, -1],
                [-1, -1, -1, 1],
                [-1, 1, -1, -1],
            ],
            [
                [-1, -1, -1, 1],
                [-1, 1, -1, -1],
                [-1, -1, -1, 1],
                [-1, -1, 1, -1],
                [-1, 1, -1, -1],
                [-1, -1, -1, 1],
                [-1, -1, 1, -1],
            ],
            [
                [-1, -1, -1, 1],
                [-1, -1, 1, -1],
                [-1, 1, -1, -1],
                [-1, -1, -1, 1],
                [-1, -1, 1, -1],
                [-1, 1, -1, -1],
                [-1, -1, -1, 1],
            ],
        ]
    )
    test_c = torch.tensor([1, 4, 2, 3])
    return test_seqs, test_c


def test_sequence_dataset(sequence_data):
    (
        seqs,
        c,
    ) = sequence_data
    dataset = SequenceDataset(seqs, c)

    assert len(dataset) == 4
    x, y = dataset[0]
    assert x.shape == (1, 7, 4)
    assert y.shape == ()

    assert torch.all(x == torch.tensor(np.array(seqs[0])))
    assert y == 1
