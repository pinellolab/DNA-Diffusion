import os
import pandas as pd
import subprocess
from tqdm import tqdm

tqdm.pandas()


def read_csv(filename):
    return pd.read_csv(filename, sep='\t')

    
def write_fasta(df, savepath):
    fasta = '\n'.join(df[['Unnamed: 0', 'raw_sequence', 'component']].apply(lambda x : f'>{x[0]}_component_{x[2]}\n{x[1]}', axis=1).values.tolist())
    with open(savepath, 'w') as f:
        f.write(fasta)

        
def generate_motifs(fasta_file, savefile="train_motifs.bed"):
    subprocess.run(f"gimme scan {fasta_file} -p JASPAR2018_vertebrates -g hg38 > {savefile}", shell=True, check=True)

    
def load_motifs(file="train_motifs.bed"):
    df = pd.read_csv(file, sep='\t', header=None, skiprows=5)
    df["motifs"] = df[8].apply(lambda x: x.split( 'motif_name "'    )[1].split('"')[0]   )
    return df

    
def merge_example(motifs, x):
    return motifs[motifs[0] == f"{x['Unnamed: 0']}_component_0"]["motifs"].tolist()

    
def merge_motifs(df, motifs):
    df["motifs"] = df.progress_apply(lambda x: merge_example(motifs, x), axis=1)
    return df
                       
if __name__ == "__main__":
    df = read_csv("train_all_classifier_WM20220916.csv")

    df_comp0 = df[df['component'] == 0]
    write_fasta(df_comp0, "train_comp0.fasta")

    if not os.path.exists("train_motifs.bed"):
        generate_motifs("train_comp0.fasta")

    df_motifs = load_motifs("train_motifs.bed")

    df_merged = merge_motifs(df_comp0, df_motifs)
    df_merged.to_csv("train_all_classifier_WM20220916_motifs.csv", index=False)
    unique_motifs = sorted(df_motifs["motifs"].unique())
    with open("unique_motifs.txt", 'w') as f:
        f.write('\n'.join(unique_motifs))