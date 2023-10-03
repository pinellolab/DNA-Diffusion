import pandas as pd
import pytest

from dnadiffusion.metrics.motif_composition import motif_composition_matrix, parse_motif_file


def test_parse_motif_file():
    output = parse_motif_file("tests/test_data/metrics/test_jaspar2020.pfm")

    expected = {
        "MA0006.1_Ahr::Arnt": 0,
        "MA0007.3_Ar": 1,
        "MA0151.1_Arid3a": 2,
        "MA0634.1_ALX3": 3,
        "MA0853.1_Alx4": 4,
        "MA0854.1_Alx1": 5,
        "MA1463.1_ARGFX": 6,
    }

    assert output == expected
