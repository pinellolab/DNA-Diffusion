'''
This script is used for generating data for Deopen training.
Usage:
    python Gen_data.py  -pos <positive_bed_file> -neg <negative_bed_file> -out <outputfile>
    python Gen_data.py -l 1000 -s 100000 -in <inputfile> -out <outputfile>
'''
import numpy as np
from pyfasta import Fasta
import hickle as hkl
import argparse
import gzip

#transfrom a sequence to one-hot encoding matrix
def seq_to_mat(seq):
    encoding_matrix = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'n':4, 'N':4}
    mat = np.zeros((len(seq),5))  
    for i in range(len(seq)):
        mat[i,encoding_matrix[seq[i]]] = 1
    mat = mat[:,:4]
    return mat

#transform a sequence to K-mer vector (default: K=6)
def seq_to_kspec(seq, K=6):
    encoding_matrix = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'n':0, 'N':0}
    kspec_vec = np.zeros((4**K,1))
    for i in range(len(seq)-K+1):
        sub_seq = seq[i:(i+K)]
        index = 0
        for j in range(K):
            index += encoding_matrix[sub_seq[j]]*(4**(K-j-1))
        kspec_vec[index] += 1
    return kspec_vec


#assemble all the features into a dictionary
def get_all_feats(spot,genome,label):
    ret = {}
    ret['spot'] = spot
    ret['seq'] = genome[spot[0]][spot[1]:spot[2]]
    ret['mat'] = seq_to_mat(ret['seq'])
    ret['kmer'] = seq_to_kspec(ret['seq'])
    ret['y'] = label
    return ret


#save the preprocessed data in hkl format 
def  save_dataset(origin_dataset,save_dir):
    dataset = {}
    for key in origin_dataset[0].keys():
        dataset[key] = [item[key] for item in origin_dataset]
    dataset['seq'] = [item.encode('ascii','ignore') for item in dataset['seq']]
    for key in origin_dataset[0].keys():
        dataset[key] = np.array(dataset[key])
    hkl.dump(dataset,save_dir, mode='w', compression='gzip')    
    print 'Training data generation is finished!'    


#generate dataset
def  generate_dataset(positive_file,negative_file,genome_file,sample_length = 1000):
    dataset=[]
    genome = Fasta(genome_file)
    with open(positive_file,'r') as f_pos:
        for line in f_pos:
            chrom = line.rstrip('\n').split('\t')[0]
            start = int(line.rstrip('\n').split('\t')[1])
            end = int(line.rstrip('\n').split('\t')[2])
            mid = (start+end)/2
            dataset.append(get_all_feats([chrom,mid-sample_length/2,mid+sample_length/2],genome,1))
    f_pos.close()
    with open(negative_file,'r') as f_neg:
        for line in f_neg:
            chrom = line.rstrip('\n').split('\t')[0]
            start = int(line.rstrip('\n').split('\t')[1])
            end = int(line.rstrip('\n').split('\t')[2])
            mid = (start+end)/2
            dataset.append(get_all_feats([chrom,mid-sample_length/2,mid+sample_length/2],genome,0))
    f_neg.close()
    return  dataset
        
        
if  __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='Deopen data generation') 
    parser.add_argument('-pos', dest='pos', type=str, help='input positive bed file')
    parser.add_argument('-neg', dest='neg', type=str, help='input negative bed file')
    parser.add_argument('-genome', dest='genome', type=str, help='genome file in fasta format')
    parser.add_argument('-l', dest='length', type=int, default=1000, help='sequence length')
    parser.add_argument('-out', dest='output', type=str, help='output file')
    args = parser.parse_args()   
    dataset = generate_dataset(args.pos,args.neg,args.genome,args.length)
    save_dataset(dataset,args.output)
 
    
