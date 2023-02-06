import torch
import numpy as np 
from Bio import SeqIO

arr = np.asarray


nuc2ind = {
    'A' : 0,
    'C' : 1,
    'G' : 2,
    'T' : 3,
    'N' : 4,
}

def seq_to_index(seq):
    """
    From seq to index
    """
    seq = arr([nuc2ind[x] for x in seq])
    return torch.Tensor(seq.astype(int)).to(torch.int64)



def get_coords_from_bedfile(bedfile):
    """
    Returns chrom, s, e
    """
    bed_df = bedfile.to_dataframe()
    chrom, s, e = bed_df['chrom'], bed_df['start'], bed_df['end']
    chrom = chrom.astype(str)
    s, e = s.astype(int), e.astype(int)
    return chrom.values, s.values, e.values


def add_chrom_to_chromlist(chromlist):
    """
    Add 'chr' from all in chromlist where it exists
    """
    return ['chr' + x if 'chr' not in x else x for x in chromlist]

def del_chrom_from_chromlist(chromlist):
    """
    Remove 'chr' from all in chromlist where it exists
    """
    return [x if 'chr' not in x else x[3:] for x in chromlist]


def get_fa_seq(records, row):
    """
    Get sequence from SeqIO file
    """
    records_have_chr = all(['chr' == x[:3] for x in records.keys()])
    sequencelist = []
    chrom, s, e = row[:3]
    s, e = map(int, [s, e])
    if records_have_chr:
        chrom = add_chrom_to_chromlist(chrom)[0]
    else:
        chrom = del_chrom_from_chromlist(chrom)[0]
    seq_as_str = str(records[chrom][s : e].seq).upper()
    return seq_as_str, f'{chrom}:{s}-{e}'