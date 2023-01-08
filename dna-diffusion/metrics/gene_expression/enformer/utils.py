import time
import os
import torch
import subprocess
import requests
from pybedtools import BedTool
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


def sort_bed_file(bed_file):
    """
    Sorts a bed file using pybedtools
    """
    bedgraph = BedTool(bed_file)
    sorted_bedgraph = bedgraph.sort()
    merged_bedgraph = sorted_bedgraph.merge()
    df_merged = merged_bedgraph.to_dataframe(names=['chrom', 'start', 'end'])
    df_sorted = sorted_bedgraph.to_dataframe(names=['chrom', 'start', 'end', 'signal'])
    df_merged['signal'] = df_sorted['signal']
    merged_bedgraph = BedTool.from_dataframe(df_merged)
    merged_bedgraph.saveas(bed_file)


def plot_dnase_track(path_to_bw):
    """
    Plots the DNAse track for chromosome 1
    """
    start, end = get_full_chr_coords("1")
    subprocess.run(["make_tracks_file", "--trackFiles", "data/chr1_dnase.bw", "-o", "data/tracks.ini"],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    subprocess.run(["pyGenomeTracks", "--tracks", "data/tracks.ini", "--region", f"chr1:{start}-{end}",
                    "--outFileName", "plots/DNAse_chr1.pdf"])


def get_full_chr_coords(chrom: str):
    """
    Gets the full coordinates for a given chromosome
    """
    base_url = "https://rest.ensembl.org/"
    species = "homo_sapiens"

    response = requests.get(f"{base_url}info/assembly/{species}?content-type=application/json")
    if response.status_code == 200:
        data = response.json()
        chromosomes = data['top_level_region']
        for chromosome in chromosomes:
            if chromosome['name'] == chrom:
                start = 1
                end = chromosome['length']
                return start, end
            else:
                print(f"Chromosome {chrom} not found")
    else:
        print(f"Error: {response.status_code}, {response.reason}")
