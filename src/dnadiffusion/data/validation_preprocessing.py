import os

import genomepy
import pandas as pd
import pybedtools
from pybedtools import BedTool

from dnadiffusion.utils.data_util import GTFProcessing


def combine_all_seqs(cell_list: list, training_data_path: str, save_output: bool = False) -> pd.DataFrame:
    """A function to take the generated sequences from sample loop and combine them with the training dataset

    Args:
        cell_list (list): A list of cell types used to generate synthetic sequences
        ["GM12878_ENCLB441ZZZ",
        "HepG2_ENCLB029COU",
        "hESCT0_ENCLB449ZZZ",
        "K562_ENCLB843GMH"]

        training_dataset_path (str): Path to the training dataset
        K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt

    Returns:
        pd.DataFrame: A dataframe of the combined synthetic sequences from all four cell types
    """
    # Reading the training dataset used to train the model
    df_train = pd.read_csv(training_data_path, sep="\t")

    dfs_to_save = {}
    generated_files = []

    for cell in cell_list:
        cell_file_name = next(f for f in os.listdir(os.getcwd()) if cell in f)
        # Creating variable to save the generated files
        file_name = cell_file_name.split("_")[1]
        # Reading current generated file
        df = pd.read_csv(cell_file_name, header=None, names=["SEQUENCE"])
        # Adding some columns
        df["CELL_TYPE"] = file_name.upper()
        df["index_number"] = [str(i) for i in df.index]
        df["TAG"] = "GENERATED"
        df["ID"] = df.apply(lambda x: "_".join([x["index_number"], x["TAG"], x["CELL_TYPE"]]), axis=1)
        # Saving the modified dataframe
        dfs_to_save["DF_" + file_name] = df
        # Remove index column
        del df["index_number"]
        generated_files.append(df)

    for k, df in df_train.groupby(["data_label", "TAG"]):
        data_in, tag_in = k

        # Removing replicate ID from tag
        tag_in = (tag_in.split("_")[0]).upper()
        data_in = data_in.upper()
        print(f"Processing {data_in} {tag_in}")
        # Reformatting the dataframe
        df_slice = df[["sequence", "TAG", "data_label"]].copy()
        df_slice.columns = ["SEQUENCE", "CELL_TYPE", "TAG"]
        df_slice["TAG"] = df_slice["TAG"].apply(lambda x: x.upper())
        df_slice["CELL_TYPE"] = df_slice["CELL_TYPE"].apply(lambda x: x.upper())
        df_slice["CELL_TYPE"] = df_slice["CELL_TYPE"].apply(lambda x: x.split("_")[0])
        df_slice["index_number"] = [str(i) for i in df_slice.index]
        df_slice["ID"] = df_slice.apply(lambda x: "_".join([x["index_number"], x["TAG"], x["CELL_TYPE"]]), axis=1)
        dfs_to_save[data_in.upper() + "_" + tag_in.upper()] = df_slice
        # Remove index column
        del df_slice["index_number"]
        generated_files.append(df_slice)

    # Combine the generated sequences with the training dataset
    df_combined = pd.concat(generated_files)
    # Rename columns
    df_combined.columns = ["SEQUENCE", "CELL_TYPE", "TAG", "ID"]
    if save_output:
        df_combined.to_csv("DNA_DIFFUSION_ALL_SEQS.txt", sep="\t")
    return df_combined


def validation_table(
    training_data_path: str,
    generated_data_path: str,
    promoter_path: str = "gencode.v43.annotation.gtf",
    add_bp_interval: int = 1000,
    num_sequences: int = 10000,
    num_filter_sequences: int = 5000,
    download_genome: bool = False,
    download_gtf_gene_annotation: bool = False,
    genome_path: str = "hg38",
    save_output: bool = False,
) -> pd.DataFrame:
    """Consolidates the training dataset, generated sequences, randomized sequences and promoter sequences into a single dataframe

    Args:
        training_data_path (str): Path to the training dataset used during model training
        generated_data_path (str): Path to the generated sequences from sample loop
        promoter_path (str, optional): Path to the promoter sequences
        add_bp_interval (int, optional): Number of base pairs to add to the promoter sequences
        num_sequences (int, optional): Number of random sequences to generate
        num_filter_sequences (int, optional): Number of random sequences to filter down to
        download_genome (bool, optional): Download the hg38 genome if not already installed
        download_gtf_gene_annotation (bool, optional): Download the gtf gene annotation if not already installed
        genome_path (str, optional): Path to the genome
        save_output (bool, optional): Save the output dataframe to a file

    Returns:
        pd.DataFrame: A dataframe of the combined sequences
    """

    if download_genome:
        genomepy.install_genome("hg38", "UCSC", genome_path)
    if download_gtf_gene_annotation:
        os.system(
            "wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.annotation.gtf.gz"
        )
        os.system("gunzip gencode.v43.annotation.gtf.gz")

    # Random sequences
    print("Generating random sequences dataframe")
    bed_tool = BedTool()
    random_seqs = bed_tool.random(l=200, n=num_sequences, genome="hg38", seed=1)
    random_seqs = random_seqs.to_dataframe()
    # Filtering chromosomes
    chromosomes = ["chr" + str(i) for i in range(1, 22)] + ["X", "Y"]
    random_seqs = random_seqs[random_seqs["chrom"].isin(chromosomes)]
    filtered_random_seqs = BedTool.from_dataframe(random_seqs).sequence(f"{genome_path}/hg38/hg38.fa")
    # Cleaning up the dataframe and renaming columns
    random_seqs["SEQUENCE"] = [x.upper() for x in open(filtered_random_seqs.seqfn).read().split("\n") if ">" not in x][
        :-1
    ]
    random_seqs = random_seqs[random_seqs["SEQUENCE"].apply(lambda x: "N" not in x)]
    random_seqs = random_seqs.head(num_filter_sequences)
    random_seqs["ID"] = random_seqs.apply(lambda x: f"{x['chrom']}_{x['start']!s}_{x['end']!s}_random", axis=1)
    random_seqs["CELL_TYPE"] = "NO"
    random_seqs["TAG"] = "RANDOM_GENOME_REGIONS"
    random_seqs = random_seqs[["chrom", "start", "end", "ID", "CELL_TYPE", "SEQUENCE", "TAG"]]

    # Promoter sequences
    print("Generating promoter sequences dataframe")
    gtf = GTFProcessing(promoter_path)
    df_gtf = gtf.get_gtf_df()
    df_gtf_filtered = df_gtf.query("feature == 'transcript'  and gene_type == 'protein_coding'  ").drop_duplicates(
        "gene_name"
    )
    df_gtf_filtered["tss_position"] = df_gtf_filtered.apply(
        lambda x: x["start"] if x["strand"] == "+" else x["end"], axis=1
    )
    df_gtf_filtered["start"] = df_gtf_filtered["tss_position"] - add_bp_interval
    df_gtf_filtered["end"] = df_gtf_filtered["tss_position"] + add_bp_interval
    df_gtf = gtf.df_to_df_bed(df_gtf_filtered)
    # Rename columns
    df_gtf_filtered["chrom"] = df_gtf_filtered["chr"]
    df_gtf_filtered["ID"] = df_gtf_filtered.apply(
        lambda x: f"{x['chrom']}_{x['start']!s}_{x['end']!s}_promoter", axis=1
    )
    df_gtf_filtered["CELL_TYPE"] = "NO"
    df_gtf_filtered["TAG"] = "PROMOTERS"
    p_seqs = BedTool.from_dataframe(df_gtf_filtered).sequence(f"{genome_path}/hg38/hg38.fa")
    df_gtf_filtered["SEQUENCE"] = [x.upper() for x in open(p_seqs.seqfn).read().split("\n") if ">" not in x][:-1]
    df_gtf_filtered = df_gtf_filtered[["chrom", "start", "end", "ID", "CELL_TYPE", "SEQUENCE", "TAG"]]

    # Reading the training dataset used to train the model
    print("Generating training dataset dataframe")
    df_train = pd.read_csv(training_data_path, sep="\t")
    df_train_balanced = pd.concat([v for k, v in df_train.groupby("TAG")])
    # Adding some metadata columns
    df_train_balanced["coord_center"] = df_train_balanced["start"] + (
        df_train_balanced["end"] - df_train_balanced["start"]
    )
    df_train_balanced["start"] = df_train_balanced["coord_center"] - 100
    df_train_balanced["end"] = df_train_balanced["coord_center"] + 100
    # Selecting only the columns we need
    df_train_balanced = df_train_balanced[["chr", "start", "end", "dhs_id", "TAG", "sequence", "data_label"]]
    df_train_balanced.columns = ["chrom", "start", "end", "ID", "CELL_TYPE", "SEQUENCE", "TAG"]

    # Reading the generated sequences
    print("Generating synthetic sequences dataframe")
    df_generated = pd.read_csv(generated_data_path, sep="\t").query("TAG == 'GENERATED'")
    df_generated_balanced = pd.concat([v for k, v in df_generated.groupby("CELL_TYPE")])
    # Adding some metadata columns
    df_generated_balanced["chrom"] = "NO"
    df_generated_balanced["start"] = "NO"
    df_generated_balanced["end"] = "NO"
    df_generated_balanced = df_generated_balanced[["chrom", "start", "end", "ID", "CELL_TYPE", "SEQUENCE", "TAG"]]
    df_generated_balanced.columns = ["chrom", "start", "end", "ID", "CELL_TYPE", "SEQUENCE", "TAG"]
    df_generated_balanced["CELL_TYPE"] = df_generated_balanced["CELL_TYPE"].apply(lambda x: x.split("_")[0])

    # Combining all the dataframes
    print("Combining all the dataframes")
    df_final = pd.concat([df_train_balanced, df_generated_balanced, df_gtf_filtered, random_seqs], ignore_index=True)
    if save_output:
        df_final.to_csv("DNA_DIFFUSION_VALIDATION_TABLE.txt", sep="\t", index=None)
    return df_final
