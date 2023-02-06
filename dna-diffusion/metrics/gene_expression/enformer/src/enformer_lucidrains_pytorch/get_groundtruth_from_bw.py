## Function to get groundtruth
import bbi
import pybedtools as pbt
import numpy as np
def get_ground_truth(bedfile, bigwig_dict, expand_by = 0, gpath=None, bin_size=128):
	""" 
	Takes a bed file and a dictionary of bigwigs to make a dataframe with a column for each entry in  
	"""
	assert type(bedfile) == pbt.bedtool.BedTool
	# If we want to add flanking regions
	if expand_by > 0:
		bedfile = bedfile.slop(b = expand_by, g = gpath)
	# Get coordinates
	bedfile_chrom, s, e = get_coords_from_bedfile(bedfile)
	assert ((e - s)%bin_size == 0).all() # Make sure seqlength divisible by 128
	assert len(np.unique(e-s)) == 1 # Make sure just one seqlength

	n_bins = ((e-s)//bin_size)[0]
	bed_df = bedfile.to_dataframe()
	for condition, bigwig in bigwig_dict.items():
		modified_bedfile_chrom = add_chr_if_necessary(bedfile_chrom, bigwig)
		bigwig_vals = bigwig.stackup(modified_bedfile_chrom, s, e, bins = n_bins)
		print(bigwig_vals.shape, bed_df.shape)
		bed_df[condition] = [list(x) for x in bigwig_vals]
	return bed_df
	

def get_coords_from_bedfile(bedfile):
	"""
	Returns chrom, s, e
	"""
	bed_df = bedfile.to_dataframe()
	chrom, s, e = bed_df['chrom'], bed_df['start'], bed_df['end']
	chrom = chrom.astype(str)
	s, e = s.astype(int), e.astype(int)
	return chrom.values, s.values, e.values

def add_chr_if_necessary(bedfile_chroms, bigwig):
	"""
	Add/remove 'chr' depending on whether bigig uses 'chr' notation
	"""
	bigwig_chroms = list(bigwig.chromsizes)
	assert ('chr' in x for x in bigwig_chroms) or ('chr' not in x for x in bigwig_chroms)
	if ('chr' in x for x in bigwig_chroms):
		new_bedfile_chroms = add_chrom_to_chromlist(bedfile_chroms)
	else:
		new_bedfile_chroms = del_chrom_from_chromlist(bedfile_chroms)
	return new_bedfile_chroms
	

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
