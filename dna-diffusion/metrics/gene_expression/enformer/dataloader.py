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
from enformer_lucidrains_pytorch.enformer_pytorch.data import str_to_one_hot


def get_abc_data(_data) -> pd.DataFrame:
    df = _data.sort_values(by='ABC_Score', ascending=False)
    df = df.drop_duplicates(subset=['TargetGene']).head(5)
    df = df[['chr', 'start', 'end', 'TargetGene']]
    return df


class EnformerDataLoader:
    """
    The dataloader should take in a gene name (or list of gene names) as Ensembl IDs and return a one-hot encoded
    196,608 length sequence centred around the TSS of the gene(s).
    """

    # TODO Low Priority: Add a type checker that checks if the input genes are valid Ensembl IDs. For now we assume
    # TODO Low Priority: that we get a DataFrame as input (ABC data) and that we need to convert the gene names to
    # TODO Low Priority: Ensembl IDs.

    def __init__(self, data: pd.DataFrame):
        self.SEQ_LEN = 196608

        self.data = get_abc_data(data)  # NOTE: This is temporary while working with abc_data. Needs changing once
        # we switch to full data.
        self.genes = self.data['TargetGene'].values
        self.gene2ensembl, self.ensembl2gene, self.ensemble_ids = self.get_ensembl_ids()
        self.gene_coordinates = self.fetch_gene_coordinates()

        # TODO 1: Implement Enformer model and test it on the data.

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
        sequences = os.listdir('sequences')
        seqs = {}
        for s in sequences:
            for record in SeqIO.parse(f'sequences/{s}', 'fasta'):
                seqs[s.split('.')[0]] = str(record.seq)
        seqs_one_hot = {}
        for s in seqs:
            seqs_one_hot[s] = str_to_one_hot(seqs[s])
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
        # TODO Low Priority: Save ensembl ids to file so we don't have to fetch them every time. Might not be necessary
        # TODO Low Priority: as we might only use this function during Enformer testing.
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
