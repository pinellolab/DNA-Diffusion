import pandas as pd
import pytest

from dnadiffusion.utils.data_util import SEQ_EXTRACT, add_sequence_column, sequence_bounds, seq_extract

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "SEQUENCE" :["ACTG", "GATC", "TAGC", "GCTA", "ATCG"],
        "CELL_TYPE": ["GM12878", "HEPG2", "HESCT0", "K562", "NO"],
        "TAG": ["GENERATED", "PROMOTERS", "RANDOM_GENOME_REGIONS", "test", "training"]
    })

def test_seq_extract(tmp_path, sample_df):

    seqs = seq_extract(sample_df, "GENERATED", "GM12878")
    assert len(seqs) == 1
    assert seqs["SEQUENCE"].tolist() == ["ACTG"]

    expected = pd.DataFrame({
        "SEQUENCE" :["ACTG"],
        "CELL_TYPE": ["GM12878"],
        "TAG": ["GENERATED"]
    })

    pd.testing.assert_frame_equal(seqs, expected)

class MockGenome:
    def sequence(self, seqname, start, end):
        return "ACGT"


@pytest.fixture
def mock_df():
    return pd.DataFrame({"seqname": ["chr1", "chr2"], "start": [10, 20], "end": [100, 200], "summit": [50, 150]})


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


def test_seq_extract(data_path: str = "tests/test_data/data_util/seq_extract_data.txt"):
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
                seq_output = pd.read_csv(f"tests/test_data/data_util/{tag}_{cell}.txt", sep="\t", dtype=object)
                # Assert the two dataframes are equal
                pd.testing.assert_frame_equal(seq_input, seq_output)
        else:
            seq_input = seqs.extract_seq(tag, cell_type).reset_index(drop=True)
            seq_output = pd.read_csv(f"tests/test_data/data_util/{tag}.txt", sep="\t", dtype=object)
            # Assert the two dataframes are equal
            pd.testing.assert_frame_equal(seq_input, seq_output)
