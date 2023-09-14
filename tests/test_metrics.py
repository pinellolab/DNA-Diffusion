import numpy as np
import pandas as pd
import pytest

from dnadiffusion.metrics.metrics import compare_motif_list


@pytest.fixture
def df_motifs_a():
    data = {
        "motif_a": 10,
        "motif_b": 15,
        "motif_c": 4,
        "motif_d": 1,
    }
    df = pd.DataFrame(list(data.items()), columns=["motifs", "0"])
    df.set_index("motifs", inplace=True)
    return df


@pytest.fixture
def df_motifs_b():
    data = {
        "motif_a": 8,
        "motif_b": 3,
        "motif_c": 1,
        "motif_d": 2,
    }
    df = pd.DataFrame(list(data.items()), columns=["motifs", "0"])
    df.set_index("motifs", inplace=True)
    return df


def test_compare_motif_list(df_motifs_a, df_motifs_b):
    result = compare_motif_list(df_motifs_a, df_motifs_b)

    assert result.dtype == float
