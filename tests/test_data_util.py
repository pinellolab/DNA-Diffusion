import pytest
import pandas as pd
from dnadiffusion.utils.data_util import SEQ_EXTRACT

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
                    input = seqs.extract_seq(tag, cell).reset_index(drop=True)
                    output = pd.read_csv(f"tests/test_data/data_util/{tag}_{cell}.txt", sep="\t", dtype=object)
                    # Assert the two dataframes are equal
                    pd.testing.assert_frame_equal(input, output)
            else:
                input = seqs.extract_seq(tag, cell_type).reset_index(drop=True)
                output = pd.read_csv(f"tests/test_data/data_util/{tag}.txt", sep="\t", dtype=object)
                # Assert the two dataframes are equal
                pd.testing.assert_frame_equal(input, output)