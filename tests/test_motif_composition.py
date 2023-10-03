from unittest.mock import patch

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


@pytest.fixture
def mock_bed(tmpdir):
    # Defining example motif data
    test_bed = [
        'sequence_0\tpfmscan\tmisc_feature\t123\t130\t8.931999505513964\t-\t.\tmotif_name "MA0607.1_Bhlha15" ; motif_instance "ACATATGC"',
        'sequence_0\tpfmscan\tmisc_feature\t34\t44\t9.44728482904739\t+\t.\tmotif_name "MA1683.1_FOXA3" ; motif_instance "GAGTAAACAGA"',
    ]

    # Writing to a bed fille
    sample_path = tmpdir.join("syn_results_motifs.bed")
    with open("syn_results_motifs.bed", "w") as f:
        f.write("# GimmeMotifs version 0.18.0\n")
        f.write("# Input: synthetic_motifs.fasta\n")
        f.write("# Motifs: JASPAR2020_vertebrates\n")
        f.write("# FPR: 0.01 (hg38)\n")
        f.write("# Scoring: logodds score\n")
        for line in test_bed:
            f.write(line + "\n")

    return str(sample_path)


@pytest.fixture
def mock_fasta(tmpdir):
    # Creating mock sequence list
    sequence_list = ["ATGC", "CGTA"]
    synthetic_motifs_fasta = tmpdir.join("synthetic_motifs.fasta")
    with open("synthetic_motifs.fasta", "w") as f:
        for i, seq in enumerate(sequence_list):
            f.write(f">sequence_{i}\n{seq}\n")
    return str(synthetic_motifs_fasta)


def test_motif_composition_matrix(
    mock_bed,
    mock_fasta,
    path: str = "tests/test_data/metrics/motif_composition.txt",
    tag: str = "RANDOM_GENOME_REGIONS",
    cell_type: str = "NO",
    motif_pfm_path: str = "tests/test_data/metrics/test_jaspar2020.pfm",
):
    # Mock run the function to see if it works
    with patch("os.system", return_value=0):
        df = motif_composition_matrix(path, tag, cell_type, motif_pfm_path=motif_pfm_path, download_data=False)

    # assert output is a non empty dataframe
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
