import pandas as pd
import time
import mygene
from tqdm import tqdm
import requests
import sys


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

    def fetch_gene_coordinates(self):
        # TODO 1: Extract gene coordinates from the decoded data.
        self.genes = self.data['TargetGene'].values
        gene2ensembl, ensembl2gene, ensemble_ids = self.get_ensembl_ids()
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
            print(repr(decoded))
        return 0

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
