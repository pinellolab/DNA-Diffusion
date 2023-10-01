import gtfparse
import matplotlib.pyplot as plt
import pandas as pd
from Bio import SeqIO
from IPython.display import HTML, display
from tqdm import tqdm


class DataSource:
    # Sourced from https://github.com/meuleman/SynthSeqs/blob/main/make_data/source.py

    def __init__(self, data, filepath):
        self.raw_data = data
        self.filepath = filepath

    @property
    def data(self):
        return self.raw_data


class ReferenceGenome(DataSource):
    """Object for quickly loading and querying the reference genome."""

    @classmethod
    def from_path(cls, path):
        genome_dict = {record.id: str(record.seq).upper() for record in SeqIO.parse(path, "fasta")}
        return cls(genome_dict, path)

    @classmethod
    def from_dict(cls, data_dict):
        return cls(data_dict, filepath=None)

    @property
    def genome(self):
        return self.data

    def sequence(self, chrom, start, end):
        chrom_sequence = self.genome[chrom]

        assert end < len(chrom_sequence), (
            f"Sequence position bound out of range for chromosome {chrom}. "
            f"{chrom} length {len(chrom_sequence)}, requested position {end}."
        )
        return chrom_sequence[start:end]


# Functions used to create sequence column
def sequence_bounds(summit: int, start: int, end: int, length: int):
    """Calculate the sequence coordinates (bounds) for a given DHS.
    https://github.com/meuleman/SynthSeqs/blob/main/make_data/process.py
    """
    half = length // 2

    if (summit - start) < half:
        return start, start + length
    elif (end - summit) < half:
        return end - length, end

    return summit - half, summit + half


def add_sequence_column(df: pd.DataFrame, genome, length: int):
    """
    Query the reference genome for each DHS and add the raw sequences
    to the dataframe.
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe of DHS annotations and NMF loadings.
    genome : ReferenceGenome(DataSource)
        A reference genome object to query for sequences.
    length : int
        Length of a DHS.

    https://github.com/meuleman/SynthSeqs/blob/main/make_data/process.py
    """
    seqs = []
    for rowi, row in df.iterrows():
        l, r = sequence_bounds(row["summit"], row["start"], row["end"], length)
        seq = genome.sequence(row["seqname"], l, r)

        seqs.append(seq)

    df["sequence"] = seqs
    return df


class SEQ_EXTRACT:
    def __init__(self, data):
        self.data = pd.read_csv(data, sep="\t")

    def extract_seq(self, tag, cell_type):
        return self.data.query(f'TAG == "{tag}" and CELL_TYPE == "{cell_type}" ').copy()


class GTFProcessing:
    # https://github.com/LucasSilvaFerreira/GTFProcessing
    def __init__(self, gtf_file_name):
        self.gtf_file_name = gtf_file_name
        self.df_gtf = self.__load_gtf__()
        self.change_columns_name()
        self.__add_interval_lenght()

    @staticmethod
    def remove_dup_columns(frame):
        keep_names = set()
        keep_icols = []
        for icol, name in enumerate(frame.columns):
            if name not in keep_names:
                keep_names.add(name)
                keep_icols.append(icol)
        return frame.iloc[:, keep_icols]

    def __load_gtf__(self):
        print("loading gtf_file...")
        gtf = gtfparse.parse_gtf_and_expand_attributes(self.gtf_file_name)
        return gtf

    def change_columns_name(self):
        print(self.df_gtf.columns)
        self.df_gtf.columns = ["chr"] + self.df_gtf.columns.values.tolist()[1:]

    def get_gtf_df(self):
        """
        return: pandas dataframe
        """

        return self.df_gtf

    def geneid2genename(self, gene_list):
        '''given a list of geneid convert it to list names
        ex:
            usage
            geneid2genename(["ENSG00000000003", "ENSG00000000004" ])
        Parameters:
            gene_list : list[str]

        return:
            conversion results gene_id to gene_name
            list[str]
        '''
        gtf_df = self.get_gtf_df()
        dict_conversion = dict(zip(gtf_df["gene_id"].values, gtf_df["gene_name"].values))
        return [dict_conversion[g] for g in gene_list]

    def __add_interval_lenght(self):
        self.df_gtf["interval_lenght"] = self.df_gtf.end - self.df_gtf.start

    @staticmethod
    def get_first_exon_df(gtf_df):
        """Group genes by transcript id and returns a df with first  exont relative to strand"""
        out_new_df = []
        for k, v in gtf_df.query("feature == 'exon' and exon_number == '1' ").groupby("transcript_id"):
            out_new_df.append(v)

        return pd.concat(out_new_df)

    @staticmethod
    def get_last_exon_df(gtf_df):
        """Group genes by transcript id and returns a df with last  exont relative to strand"""

        out_new_df = []
        for k, v in tqdm(gtf_df.query("feature == 'exon' ").groupby("transcript_id")):
            if v.iloc[0].strand == "+":
                out_new_df.append(v.sort_values("end", ascending=True).iloc[-1].values)

                # print v.sort_values('exon_number').iloc[0]
            if v.iloc[0].strand == "-":
                out_new_df.append(v.sort_values("start", ascending=True).iloc[0].values)

        return pd.DataFrame(out_new_df, columns=gtf_df.columns)

    @staticmethod
    def df_to_bed(gtf_df, bed_file_name, fourth_position_feature="gene_name", fifth_position_feature="transcript_id"):
        """Save a bed_file using a gtf as reference and returns the bed_file_name string"""

        print(gtf_df[["chr", "start", "end", fourth_position_feature, fifth_position_feature, "strand"]].head())

        gtf_df[["chr", "start", "end", fourth_position_feature, fifth_position_feature, "strand"]].to_csv(
            bed_file_name, sep="\t", header=None, index=None
        )
        return bed_file_name

    @staticmethod
    def df_to_df_bed(gtf_df, fourth_position_feature="gene_name", fifth_position_feature="transcript_id"):
        """Save a bed_file using a gtf as reference and returns df with a bed6 format"""

        print(gtf_df[["chr", "start", "end", fourth_position_feature, fifth_position_feature, "strand"]].head())

        return gtf_df[["chr", "start", "end", fourth_position_feature, fifth_position_feature, "strand"]]

    @staticmethod
    def hist_generate(gtf_df, feature="transcript_biotype"):
        """
        ex: GTFProcessing.hist_generate(gtf_to_test.head(1600), 'transcript_biotype')
        """
        x_axis_feature = GTFProcessing.get_first_exon_df(gtf_df).groupby(feature).count()["start"]
        plt.bar(range(0, x_axis_feature.values.shape[0]), x_axis_feature.values)
        print(x_axis_feature.keys())
        print(x_axis_feature.values)
        plt.xticks(range(0, x_axis_feature.values.shape[0]), (x_axis_feature.keys().values), rotation="vertical")
        plt.title(feature)
        plt.show()

    @staticmethod
    def generate_hist_by_transcript_biotypes(gtf_df):
        GTFProcessing.hist_generate(gtf_df, feature="transcript_biotype")

    @staticmethod
    def capture_distal_unique_tes(gtf_df):
        return_distal_exon = []
        last_exon_df = GTFProcessing.get_last_exon_df(gtf_df)
        for k, v in tqdm(last_exon_df.groupby("gene_id")):
            if v.iloc[0]["strand"] == "+":
                return_distal_exon.append(v.sort_values("end", ascending=False).iloc[0].values.tolist())
            if v.iloc[0]["strand"] == "-":
                return_distal_exon.append(v.sort_values("start", ascending=True).iloc[0].values.tolist())

        df_distal_exon_by_gene_id = pd.DataFrame(return_distal_exon, columns=last_exon_df.columns.values.tolist())
        return df_distal_exon_by_gene_id

    @staticmethod
    def capture_distal_unique_tss(gtf_df):
        return_distal_tss = []
        first_exon_df = GTFProcessing.get_first_exon_df(gtf_df)
        for k, v in tqdm(first_exon_df.groupby("gene_id")):
            if v.iloc[0]["strand"] == "+":
                return_distal_tss.append(v.sort_values("start", ascending=True).iloc[0].values.tolist())
            if v.iloc[0]["strand"] == "-":
                return_distal_tss.append(v.sort_values("end", ascending=False).iloc[0].values.tolist())

        df_distal_exon_by_gene_id = pd.DataFrame(return_distal_tss, columns=first_exon_df.columns.values.tolist())
        return df_distal_exon_by_gene_id
