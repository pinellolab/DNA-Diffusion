import random

import pytest

from dnadiffusion.utils.utils import one_hot_encode


@pytest.fixture
def random_dna_sequence():
    bases = ["A", "T", "C", "G"]
    return "".join(random.choice(bases) for _ in range(200))


def test_one_hot_encode(random_dna_sequence):
    bases = ["A", "T", "C", "G"]
    seq_array = one_hot_encode(random_dna_sequence, bases, 200)
    assert seq_array.shape == (200, 4)
