import time
import os
import torch
import pandas as pd


def inference(one_hot_seqs, model):
    start_time = time.perf_counter()
    for gene, seq in one_hot_seqs.items():
        # check if gene is already in outputs folder
        if not os.path.exists(f'outputs/{gene}.pkl'):
            print(f"Running Enformer inference for {gene}")
            output = model.forward(seq)
            torch.cuda.empty_cache()
            output_df = create_annotated_dataframe("genomic_track_type_data.xlsx", output)
            output_df.to_pickle(f'outputs/{gene}.pkl')
        else:
            print(f'Output for {gene} already exists. Skipping...')
    end_time = time.perf_counter()
    print(f'Inference took {round(end_time - start_time, 2)} seconds')


def create_annotated_dataframe(deepmind_table, output):
    """
    Creates a dataframe with the following columns:
    - assay_type
    - target (cell or tissue type that was targeted in the assay)
    - output (the output Enformer has produced for the given gene)
    Author: @kierandidi
    """
    df = pd.read_excel(deepmind_table,
                       "Supplementary Table 2", index_col=None, na_values=['NA'], usecols="I:J")
    tracks_output = pd.DataFrame([[track] for track in output['human'].T.detach().numpy()])
    df['output'] = tracks_output
    targets = df['target'].str.split(pat="/", n=1, expand=True)
    df['target'] = targets[1]
    return df
