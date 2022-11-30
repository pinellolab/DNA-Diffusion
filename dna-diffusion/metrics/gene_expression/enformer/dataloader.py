import pandas as pd
import time
import mygene
from tqdm import tqdm
import requests
import sys
import warnings
from pybedtools import BedTool


def get_abc_data(_data) -> pd.DataFrame:
    df = _data.sort_values(by='ABC_Score', ascending=False)
    df = df.drop_duplicates(subset=['TargetGene']).head(1)
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
        self.data = get_abc_data(data)  # NOTE: This is temporary while working with abc_data. Needs changing once
        # we switch to full data.
        self.genes = None
        self.gene_coordinates = self.fetch_gene_coordinates()
        self.sequence = self.fetch_sequence()

        # TODO 1: Get 200 kb sequence centred around TSS from the given genomic coordinates for GRCh38 for all the genes.

    def fetch_sequence(self):
        gene_coordinates = self.gene_coordinates
        gene_ids = list(gene_coordinates.keys())
        for gene in gene_ids:
            start, end, chromosome, orientation = gene_coordinates[gene]
            if orientation == 1:
                orientation = '+'
            elif orientation == -1:
                orientation = '-'
            else:
                orientation = '.'
            with open('temp.bed', 'w') as f:
                f.write(f'{chromosome}\t{start}\t{end}\t{orientation}')
            bed = BedTool('temp.bed')
            fasta = bed.sequence(fi='GRCh38.fa')
            print(fasta)
            # TODO 1: Download the GRCh38 genome sequence as fasta file to use pybedtools sequence fetcher

        return 0

    def fetch_gene_coordinates(self):
        self.genes = self.data['TargetGene'].values
        gene2ensembl, ensembl2gene, ensemble_ids = self.get_ensembl_ids()
        gene_coordinates = {}
        print('Fetching gene coordinates for the given Ensembl ID(s)...')
        print(ensemble_ids)
        for gene in ensemble_ids:
            time.sleep(0.5)
            url = f"https://rest.ensembl.org/lookup/id/{gene}?expand=1"
            r = requests.get(url, headers={"Content-Type": "application/json"})
            if not r.ok:
                r.raise_for_status()
                sys.exit()
            decoded = r.json()
            start, end, chromosome, orientation = decoded['start'], decoded['end'], int(decoded['seq_region_name']), \
                                                  decoded['strand']
            assembly_name = decoded['assembly_name']

            if assembly_name != 'GRCh38':
                # TODO Low Priority: Convert start, end coordinates to GRCh38 rather than skipping.
                warnings.warn(f'Assembly in Ensembl database for {gene} is not built with GRCh38. Skipping...')
                continue
            gene_coordinates[gene] = [start, end, chromosome, orientation]
        return gene_coordinates

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


# DEBUGGING PURPOSES ONLY #
if __name__ == "__main__":
    loader = EnformerDataLoader(pd.read_csv("abc_data/K562.PositivePredictions.txt", sep="\t"))
