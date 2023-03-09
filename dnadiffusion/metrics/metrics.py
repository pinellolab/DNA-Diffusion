import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import rel_entr
from tqdm import tqdm
from Bio import SeqIO
from dnadiffusion.sample import sampling_to_metric
from dnadiffusion.utils.utils import one_hot_encode
from collections import defaultdict
from sourmash import MinHash
import pandas as pd
import numpy as np

def _create_pandas_series_from_a_fasta_file(fasta_file:str) -> pd.Series:
    """Create a pandas series from a fasta file. Input is a fasta file"""
    data = defaultdict(list)
    for record in SeqIO.parse(fasta_file,"fasta"):
        seq = record.seq
        data["sequence"].append(seq)
    df = pd.DataFrame.from_dict(data)


def _create_mini_hash_of_a_sequence(seq:str,minihash:MinHash) -> MinHash: 
  """Create a minihash of a sequence. Input is a sequence and a minihash object"""
  for k in seq:
    minihash.add_sequence(k)
  return minihash


def _compare_two_sequences_and_return_similarity(seq:str, seq2:str, k:int, n:int) -> float:
  """Calculate similarity of two sequences. Input is 2 sequences, k size of kmer and n number of hashes"""
  mh1 = MinHash(n=n, ksize=k)
  mh2 = MinHash(n=n, ksize=k)
  mh1 = _create_mini_hash_of_a_sequence(seq,mh1)
  mh2 = _create_mini_hash_of_a_sequence(seq2,mh2)
  similarity = round(mh1.similarity(mh2), 5)
  return similarity


def average_jaccard_similarity(seq:any, seq2:any, number_of_hashes:int = 20000, k_sizes:list = [3,7,20],is_fasta = False) -> float:
  """Calculate average similarity of two sequences. Input is 2 sequences, k sizes of kmer, n number of hashes and a boolean to indicate 
  if the input is a fasta file or not"""
  if is_fasta:
    seq = _create_pandas_series_from_a_fasta_file(seq)
    seq2 = _create_pandas_series_from_a_fasta_file(seq2)
  average_similarities = []
  sequence_1 = seq.tolist()
  sequence_2 = seq2.tolist()
  for k in k_sizes:
    similarity = _compare_two_sequences_and_return_similarity(sequence_1, sequence_2, k, number_of_hashes)
    average_similarities.append(similarity)
  average_similarities = np.array(average_similarities)
  average_similarity = round(average_similarities.mean(), 3)
  return average_similarity


def compare_motif_list(df_motifs_a: pd.DataFrame, df_motifs_b: pd.DataFrame):
    # Using KL divergence to compare motifs lists distribution
    set_all_mot = set(
        df_motifs_a.index.values.tolist() + df_motifs_b.index.values.tolist()
    )
    create_new_matrix = []
    for x in set_all_mot:
        list_in = []
        list_in.append(x)  # adding the name
        if x in df_motifs_a.index:
            list_in.append(df_motifs_a.loc[x][0])
        else:
            list_in.append(1)

        if x in df_motifs_b.index:
            list_in.append(df_motifs_b.loc[x][0])
        else:
            list_in.append(1)

        create_new_matrix.append(list_in)

    df_motifs = pd.DataFrame(create_new_matrix, columns=["motif", "motif_a", "motif_b"])

    df_motifs["Diffusion_seqs"] = df_motifs["motif_a"] / df_motifs["motif_a"].sum()
    df_motifs["Training_seqs"] = df_motifs["motif_b"] / df_motifs["motif_b"].sum()
    """
    plt.rcParams["figure.figsize"] = (3,3)
    sns.regplot(x='Diffusion_seqs',  y='Training_seqs',data=df_motifs)
    plt.xlabel('Diffusion Seqs')
    plt.ylabel('Training Seqs')
    plt.title('Motifs Probs')
    plt.show()
    """
    kl_pq = rel_entr(
        df_motifs["Diffusion_seqs"].values, df_motifs["Training_seqs"].values
    )
    return np.sum(kl_pq)


def kl_comparison_between_dataset(first_dict: dict, second_dict: dict):
    final_comp_kl = []
    for _, v in first_dict.items():
        comp_array = []
        for k_second in second_dict.keys():
            kl_out = compare_motif_list(v, second_dict[k_second])
            comp_array.append(kl_out)
        final_comp_kl.append(comp_array)
    return final_comp_kl


def kl_comparison_generated_sequences(
    cell_list: list,
    dict_target_cells: dict,
    additional_variables: dict,
    conditional_numeric_to_tag: dict,
    number_of_sequences_sample_per_cell: int = 1000,
):
    final_comp_kl = []
    use_cell_list = cell_list
    for r in use_cell_list:
        # print(r)
        print(conditional_numeric_to_tag[r])
        comp_array = []
        group_compare = r
        synt_df_cond = sampling_to_metric(
            [r],
            conditional_numeric_to_tag,
            additional_variables,
            int(number_of_sequences_sample_per_cell / 10),
            specific_group=True,
            group_number=group_compare,
            cond_weight_to_metric=1,
        )
        for k in use_cell_list:
            v = dict_target_cells[conditional_numeric_to_tag[k]]
            kl_out = compare_motif_list(synt_df_cond, v)
            comp_array.append(kl_out)
        final_comp_kl.append(comp_array)
    return final_comp_kl


def generate_heatmap(
    df_heat: pd.DataFrame, x_label: str, y_label: str, cell_components: str
):
    plt.clf()
    plt.rcdefaults()
    plt.rcParams["figure.figsize"] = (10, 10)
    df_plot = pd.DataFrame(df_heat)
    df_plot.columns = [x.split("_")[0] for x in cell_components]
    df_plot.index = df_plot.columns
    sns.heatmap(df_plot, cmap="Blues_r", annot=True, lw=0.1, vmax=1, vmin=0)
    plt.title(
        f"Kl divergence \n {x_label} sequences x  {y_label} sequences \n MOTIFS probabilities"
    )
    plt.xlabel(f"{x_label} Sequences  \n(motifs dist)")
    plt.ylabel(f"{y_label} \n (motifs dist)")
    plt.grid(False)
    plt.savefig(f"./graphs/{x_label}_{y_label}_kl_heatmap.png")
    # wandb.log({f"Kl divergence \n {x_label} sequences x  {y_label} sequences \n MOTIFS probabilities": plt})


def generate_similarity_metric():
    """Capture the syn_motifs.fasta and compare with the  dataset motifs"""
    nucleotides = ["A", "C", "G", "T"]
    seqs_file = open("synthetic_motifs.fasta").readlines()
    seqs_to_hotencoder = [
        one_hot_encode(s.replace("\n", ""), nucleotides, 200).T
        for s in seqs_file
        if ">" not in s
    ]

    return seqs_to_hotencoder


def get_best_match(db, x_seq):  # transforming in a function
    return (db * x_seq).sum(1).sum(1).max()


def calculate_mean_similarity(database, input_query_seqs, seq_len=200):
    final_base_max_match = np.mean(
        [get_best_match(database, x) for x in tqdm(input_query_seqs)]
    )
    return final_base_max_match / seq_len


def generate_similarity_using_train(X_train_in):
    convert_X_train = X_train_in.copy()
    convert_X_train[convert_X_train == -1] = 0
    generated_seqs_to_similarity = generate_similarity_metric()
    return calculate_mean_similarity(convert_X_train, generated_seqs_to_similarity)
