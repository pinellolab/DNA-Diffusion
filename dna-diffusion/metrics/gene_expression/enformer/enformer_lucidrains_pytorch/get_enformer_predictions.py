import torch
from Bio import SeqIO
import pybedtools as pbt
from aux_functions import *
from enformer_pytorch import seq_indices_to_one_hot

def get_enformer_pred(enformer, bedfile, expand_by = 0, gpath=None, bin_size=128, enformer_seqlength = 196_608, 
                                fapath=None, enformer_keys = {}, records=None):
    """ 
    Takes a bed file and returns enformer predictions
    """
    assert type(bedfile) == pbt.bedtool.BedTool
    # If we want to add flanking regions
    if expand_by > 0:
        bedfile = bedfile.slop(b = expand_by, g = gpath)
    # Get coordinates
    bedfile_chrom, s, e = get_coords_from_bedfile(bedfile)
    assert ((e - s) == enformer_seqlength).all() # Make sure seqlength equals enformer prediction length
    
    if records is None:
        records = SeqIO.to_dict(SeqIO.parse(open(fapath), 'fasta'))    
    
    resdict = {}
    for row in bedfile:
        seq, genomic_region = get_fa_seq(records, row)
        indices = seq_to_index(seq)
        one_hot = seq_indices_to_one_hot(indices).cuda()
        with torch.no_grad():
            output = enformer(one_hot)
            
        full_output = []
        output_descriptions = []
        for organism in ['mouse', 'human']:
            organism_inds = enformer_keys.get(organism)
            if organism_inds is None:
                continue
            else:
                organism_output = output[organism][:, organism_inds].cpu()
                full_output.append(organism_output)
                output_descriptions += [f'{organism}_{i}' for i in organism_inds]
        assert len(full_output) > 0
        full_torch_output = torch.cat(full_output, axis=1)
        resdict[genomic_region] = full_torch_output
    return resdict, output_descriptions



### Minimally reproducing example:

import torch
from enformer_pytorch import Enformer
from Bio import SeqIO
records = SeqIO.to_dict(SeqIO.parse(open('mm10/mm10.fa'), 'fasta'))
bedfile = pbt.BedTool([['1', 10_000_000, 10_000_000+196608]])
enformer_keys = {
    'human' : [0, 1, 2],
    'mouse' : [100]
}

enformer = Enformer.from_hparams(
    dim = 1536,
    depth = 1,
    heads = 8,
)
results_dict, output_descriptions = get_enformer_pred(enformer.cuda(), bedfile, enformer_keys = enformer_keys, records = records)