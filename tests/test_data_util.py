from unittest.mock import patch

import pandas as pd
import pytest

from dnadiffusion.utils.data_util import (
    SEQ_EXTRACT,
    add_sequence_column,
    motif_composition_helper,
    seq_extract,
    sequence_bounds,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "SEQUENCE": ["ACTG", "GATC", "TAGC", "GCTA", "ATCG"],
            "CELL_TYPE": ["GM12878", "HEPG2", "HESCT0", "K562", "NO"],
            "TAG": [
                "GENERATED",
                "PROMOTERS",
                "RANDOM_GENOME_REGIONS",
                "test",
                "training",
            ],
        }
    )


def test_seq_extract(tmp_path, sample_df):
    seqs = seq_extract(sample_df, "GENERATED", "GM12878")
    assert len(seqs) == 1
    assert seqs["SEQUENCE"].tolist() == ["ACTG"]

    expected = pd.DataFrame(
        {"SEQUENCE": ["ACTG"], "CELL_TYPE": ["GM12878"], "TAG": ["GENERATED"]}
    )

    pd.testing.assert_frame_equal(seqs, expected)


class MockGenome:
    def sequence(self, seqname, start, end):
        return "ACGT"


@pytest.fixture
def mock_df():
    return pd.DataFrame(
        {
            "seqname": ["chr1", "chr2"],
            "start": [10, 20],
            "end": [100, 200],
            "summit": [50, 150],
        }
    )


def test_add_sequence_column(mock_df):
    genome = MockGenome()
    df = add_sequence_column(mock_df, genome, length=10)

    # Check sequence column added
    assert "sequence" in df.columns

    # Check sequence values
    expected_seqs = ["ACGT", "ACGT"]
    assert df["sequence"].tolist() == expected_seqs


def test_sequence_bounds():
    # Middle summit
    summit = 100
    start = 0
    end = 200
    length = 50
    expected = (75, 125)
    assert sequence_bounds(summit, start, end, length) == expected

    # Start summit
    summit = 0
    start = 0
    end = 100
    length = 10
    expected = (0, 10)
    assert sequence_bounds(summit, start, end, length) == expected

    # End summit
    summit = 199
    start = 100
    end = 200
    length = 20
    expected = (180, 200)
    assert sequence_bounds(summit, start, end, length) == expected


def test_seq_extract(
    data_path: str = "tests/test_data/data_util/seq_extract_data.txt",
):
    seqs = SEQ_EXTRACT(data_path)

    # Dict of all the tag combinations
    tag_dict = {
        "RANDOM_GENOME_REGIONS": "NO",
        "PROMOTERS": "NO",
        "training": ["GM12878", "K562", "HepG2", "hESCT0"],
        "validation": ["GM12878", "K562", "HepG2", "hESCT0"],
        "test": ["GM12878", "K562", "HepG2", "hESCT0"],
        "GENERATED": ["GM12878", "K562", "HEPG2", "HESCT0"],
    }

    # Looping through the tag combinations
    for tag, cell_type in tag_dict.items():
        if isinstance(cell_type, list):
            for cell in cell_type:
                seq_input = seqs.extract_seq(tag, cell).reset_index(drop=True)
                seq_output = pd.read_csv(
                    f"tests/test_data/data_util/{tag}_{cell}.txt",
                    sep="\t",
                    dtype=object,
                )
                # Assert the two dataframes are equal
                pd.testing.assert_frame_equal(seq_input, seq_output)
        else:
            seq_input = seqs.extract_seq(tag, cell_type).reset_index(drop=True)
            seq_output = pd.read_csv(
                f"tests/test_data/data_util/{tag}.txt", sep="\t", dtype=object
            )
            # Assert the two dataframes are equal
            pd.testing.assert_frame_equal(seq_input, seq_output)


@pytest.fixture
def mock_bed(tmpdir):
    # Defining example motif data
    test_bed = [
        'sequence_0\tpfmscan\tmisc_feature\t123\t130\t8.931999505513964\t-\t.\tmotif_name "MA0607.1_Bhlha15" ; motif_instance "ACATATGC"',
        'sequence_0\tpfmscan\tmisc_feature\t34\t44\t9.44728482904739\t+\t.\tmotif_name "MA1683.1_FOXA3" ; motif_instance "GAGTAAACAGA"',
        'sequence_0\tpfmscan\tmisc_feature\t36\t43\t6.450593195402899\t+\t.\tmotif_name "MA1489.1_FOXN3" ; motif_instance "GTAAACAG"',
        'sequence_0\tpfmscan\tmisc_feature\t35\t45\t10.249462278895773\t-\t.\tmotif_name "MA0480.1_Foxo1" ; motif_instance "AGTAAACAGAA"',
        'sequence_0\tpfmscan\tmisc_feature\t34\t44\t10.965594517325535\t+\t.\tmotif_name "MA0593.1_FOXP2" ; motif_instance "GAGTAAACAGA"',
        'sequence_0\tpfmscan\tmisc_feature\t107\t116\t11.634852301936267\t+\t.\tmotif_name "UN0330.1_ZNF513" ; motif_instance "AGAGGAAGAG"',
        'sequence_0\tpfmscan\tmisc_feature\t84\t96\t11.572869058140274\t-\t.\tmotif_name "UN0332.1_ZNF534" ; motif_instance "AATGGGCAAGAAC"',
        'sequence_1\tpfmscan\tmisc_feature\t139\t151\t9.80846582352393\t+\t.\tmotif_name "MA0800.1_EOMES" ; motif_instance "AAGGTGTTAACAT"',
        'sequence_1\tpfmscan\tmisc_feature\t101\t111\t10.183240817371155\t+\t.\tmotif_name "MA0002.2_RUNX1" ; motif_instance "GTCTGTGGTTA"',
        'sequence_1\tpfmscan\tmisc_feature\t62\t81\t12.509519971299817\t+\t.\tmotif_name "MA0080.5_SPI1" ; motif_instance "TAAAATGAGGAACTGAAGTA"',
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


@pytest.fixture
def sample_df():
    return pd.read_csv(
        "tests/test_data/data_util/motif_composition_helper_data.txt", sep="\t"
    )


def test_motif_composition_helper(sample_df, mock_fasta, mock_bed):
    # Mock os.system function
    with patch("os.system", return_value=0):
        df_motifs_count_syn = motif_composition_helper(sample_df)

    # Asserting that the output is a dataframe
    assert isinstance(df_motifs_count_syn, pd.DataFrame)

    # Asserting that the output file is not empty
    assert df_motifs_count_syn.shape[0] > 0
