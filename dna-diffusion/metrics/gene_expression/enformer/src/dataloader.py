import pandas as pd
import time
import mygene
from tqdm import tqdm
import requests
import sys
import os
import warnings
from pybedtools import BedTool
from Bio import SeqIO
import torch
import subprocess

import utils
from enformer_lucidrains_pytorch.enformer_pytorch.data import str_to_one_hot


def get_abc_data(_data_path) -> pd.DataFrame:
    df = pd.read_csv(_data_path, sep="\t")
    df = df.sort_values(by='ABC_Score', ascending=False)
    df = df.drop_duplicates(subset=['TargetGene']).head(5)
    df = df[['chr', 'start', 'end', 'TargetGene']]
    return df


class EnformerDataloaderABC:
    """
    The dataloader should take in a gene name (or list of gene names) as Ensembl IDs and return a one-hot encoded
    196,608 length sequence centred around the TSS of the gene(s).
    """

    # TODO Low Priority: Add a type checker that checks if the input genes are valid Ensembl IDs. For now we assume
    # TODO Low Priority: that we get a DataFrame as input (ABC data) and that we need to convert the gene names to
    # TODO Low Priority: Ensembl IDs.

    def __init__(self, data_path: str):
        self.SEQ_LEN = 196608

        self.data = get_abc_data(data_path)  # NOTE: This is temporary while working with abc_data. Needs changing once
        # we switch to full data.
        self.genes = self.data['TargetGene'].values
        self.gene2ensembl, self.ensembl2gene, self.ensemble_ids = self.get_ensembl_ids()
        self.gene_coordinates = self.fetch_gene_coordinates()

    def fetch_sequence(self):
        gene_ids = list(self.gene_coordinates.keys())
        for gene in gene_ids:
            start, end, chromosome, orientation = self.gene_coordinates[gene]
            if orientation == 1:
                orientation = '+'
            elif orientation == -1:
                orientation = '-'
            else:
                orientation = '.'
            with open('temp.bed', 'w') as f:
                f.write(f'{chromosome}\t{start}\t{end}\t{orientation}')
            f.close()
            bed = BedTool('temp.bed')
            try:
                fasta = bed.sequence(fi='hg38.fa', fo=f'sequences/{gene}|{self.ensembl2gene[gene]}.fa')
            except:
                warnings.warn(f'Out-of-bounds error for {gene} / {self.ensembl2gene[gene]}. This means that the gene is'
                              f' located to close to the telomeres in order to extend a 196,608 window around the TSS. '
                              f'Skipping...')
                os.remove(f'sequences/{gene}|{self.ensembl2gene[gene]}.fa')
            os.remove('temp.bed')
        print(f'Fetched all sequences and saved to sequences folder!')
        sequences = os.listdir('../sequences')
        seqs = {}
        for s in sequences:
            for record in SeqIO.parse(f'sequences/{s}', 'fasta'):
                seqs[s.split('.')[0]] = str(record.seq)
        seqs_one_hot = {}
        for s in seqs:
            seqs_one_hot[s] = str_to_one_hot(seqs[s]).type(torch.float32)
        print("Sequences one-hot encoded!")
        return seqs_one_hot

    def fetch_gene_coordinates(self):
        gene_coordinates = {}
        print('Fetching gene coordinates for the given Ensembl ID(s)...')
        for gene in self.ensemble_ids:
            time.sleep(0.5)
            url = f"https://rest.ensembl.org/lookup/id/{gene}?expand=1"
            r = requests.get(url, headers={"Content-Type": "application/json"})
            if not r.ok:
                r.raise_for_status()
                sys.exit()
            decoded = r.json()
            start, end, chromosome, orientation = decoded['start'], decoded['end'], \
                f"chr{int(decoded['seq_region_name'])}", decoded['strand']
            assembly_name = decoded['assembly_name']
            start, end = self.extend_gene_coordinates(start, end, chromosome)
            if assembly_name != 'GRCh38':
                # TODO Low Priority: Convert start, end coordinates to GRCh38 rather than skipping.
                warnings.warn(f'Assembly in Ensembl database for {gene} is not built with GRCh38. Skipping...')
                continue
            gene_coordinates[gene] = [start, end, chromosome, orientation]
        return gene_coordinates

    def extend_gene_coordinates(self, start, end, chromosome):
        len_left = (self.SEQ_LEN - (end - start)) // 2
        len_right = self.SEQ_LEN - (end - start) - len_left
        start -= len_left
        end += len_right
        return start, end

    def get_ensembl_ids(self):
        print('Fetching Ensembl IDs for the given list of gene(s)...')
        mg = mygene.MyGeneInfo()
        gene2ensembl = {}
        ensembl2gene = {}
        ensemble_ids = []
        processed_genes = []
        for gene in tqdm(self.genes):
            result = mg.query(gene, scopes='symbol', species='human', fields=['ensembl'], verbose=False)
            for hit in result["hits"]:
                if "ensembl" in hit and "gene" in hit["ensembl"] and gene not in processed_genes:
                    gene2ensembl[gene] = hit["ensembl"]["gene"]
                    ensembl2gene[hit["ensembl"]["gene"]] = gene
                    ensemble_ids.append(hit["ensembl"]["gene"])
                    processed_genes.append(gene)
        return gene2ensembl, ensembl2gene, ensemble_ids


class EnformerDataloaderDNAse:
    """
    This dataloader takes in the experimental DNAse data containing the DNAse read-outs for four different cell types.
    The dataloader should give the experimental data table and in parallel, process chromosome 1 in order to run
    Enformer inference for comparison with the experimental data.
    """

    def __init__(self, data_path: str):
        self.SEQ_LEN = 196608

        self.data = pd.read_csv(data_path, sep='\t')
        self.make_bed_file()

    def make_bed_file(self):
        df = self.data[self.data['chr'] == 'chr1']
        with open('../data/chr1_dnase.bedGraph', 'w') as f:
            for _, row in df.iterrows():
                f.write(f'{row["chr"]}\t{row["start"]}\t{row["end"]}\t{row["total_signal"]}\n')
        f.close()
        utils.sort_bed_file('../data/chr1_dnase.bedGraph')
        self.bedGraph2bigwig('../data/chr1_dnase.bedGraph')

    @staticmethod
    def bedGraph2bigwig(bedgraph_file):
        """
        Converts a bedGraph file to bigWig format. The scripts needed for this project are: mergeBed, bedGraphToBigWig,
        and fetchChromSizes. They can be found here: http://hgdownload.soe.ucsc.edu/admin/exe/. Don't forget to make
        them executable.
        """
        with open("../data/hg38.chrom.sizes", "w") as f:
            subprocess.run(["./exec/fetchChromSizes", "hg38"], stdout=f)
        f.close()
        subprocess.run(["exec/bedGraphToBigWig", bedgraph_file, "data/hg38.chrom.sizes", "data/chr1_dnase.bw"])
        utils.plot_dnase_track('../data/chr1_dnase.bw')
