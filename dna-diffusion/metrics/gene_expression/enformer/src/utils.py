import time
import os
from tqdm import tqdm
import torch
import subprocess
import requests
from pybedtools import BedTool
import pybedtools
import pandas as pd


def inference(one_hot_seqs, model):
    start_time = time.perf_counter()
    for seq_name, seq in tqdm(one_hot_seqs.items()):
        # check if gene is already in outputs folder
        if not os.path.exists(f'outputs/{seq_name}.pkl'):
            print(f"Running Enformer inference for {seq_name}")
            output = model.forward(seq)
            torch.cuda.empty_cache()
            output_df = create_annotated_dataframe("genomic_track_type_data.xlsx", output)
            output_df.to_pickle(f'outputs/{seq_name}.pkl')
        else:
            print(f'Output for {seq_name} already exists. Skipping...')
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
    tracks_output = pd.DataFrame([[track] for track in output['human'].T.detach().cpu().numpy()])
    df['output'] = tracks_output
    targets = df['target'].str.split(pat="/", n=1, expand=True)
    df['target'] = targets[1]
    return df


def sort_bed_file(bed_file):
    """
    Sorts and merges a bed file using pybedtools
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


def extend_gene_coordinates(start, end, SEQ_LEN):
    len_left = (SEQ_LEN - (end - start)) // 2
    len_right = SEQ_LEN - (end - start) - len_left
    start -= len_left
    end += len_right
    return start, end


def split_bed(bed):
    """
    Splits a bed entry into equally sized regions of length 196608 such that the result does not overlap.
    """
    bed = bed.to_dataframe().to_numpy()
    new_bed = []
    for row in bed:
        start, end = int(row[1]), int(row[2])
        if end - start > 196608:
            while end - start > 196608:
                new_bed.append([row[0], start, start + 196608])
                start += 196608 + 1
            new_bed.append([row[0], start, end])
        else:
            new_bed.append([row[0], start, end])
    return BedTool.from_dataframe(pd.DataFrame(new_bed, columns=['chrom', 'start', 'end']))


def trim_bed_file(bed_file, path_to_ref_genome):
    """
    Trims a bed file to the coordinates of the reference genome
    """
    bed = pybedtools.BedTool(bed_file)
    bed = bed.sort()
    bed = bed.merge()
    bed = split_bed(bed)

    chrom_lengths = {}

    with open(path_to_ref_genome, 'r') as f:
        for line in f:
            if line[:6] == '>chr1 ':
                fasta_header = line[1:].strip().split()
                chrom, length = fasta_header[0], int(fasta_header[1])
                chrom_lengths[chrom] = int(length)
    trimmed_bed = bed.filter(lambda x: int(x.end) < chrom_lengths[x.chrom]).saveas('data/chr1_dnase.bedGraph')
    return trimmed_bed
