# Data

## selected_K562_hESCT0_HepG2_GM12878_12k_sequences_per_group  


### Description  
Thi dataset is a subset of sequences that are present in four different cell types (__GM12878_ENCLB441ZZZ__  , __hESCT0_ENCLB449ZZZ__   
,__K562_ENCLB843GMH__   , __HepG2_ENCLB029COU__). Each cell type was selected using a ENCL sample that was choose as the best quality one (based in some critearias defined by Wouter). The first criteria is be present just in one of this four initial samples.
The final sequence number is balanced using the celltype with smallest number of sequences (GM12878). The  number of remaining sequences is around  ~12k  per celltype. The data is divided in training , validation (chr1), test (chr2). During the sequence selection we ed and selected the sequences having the cell specif peak in one of the four ENCL samples and they were sorted by the prencese of this peak in multiple experiments of the same celltype and reversed sorted by to decreasce the presence of this peak in other experiments of the database (total set with ~700 experiments).


4 cell types link (~48 sequences- Single file - 12k seqs per cell) 

  -   [Four cell types table](https://drive.google.com/drive/folders/1dBeZIdJZQqaZUzCBVrz_Z2fAV9ePsw7h?usp=sharing)  
  -   File: DNA-Diffusion/dna-diffusion/data/selected_K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt  

  
__dhs_id__ : dhs id  
__chr__ :	  chr   
__start__ :  start  
__end__ :  end  
__DHS_width__ :  The  peak width  
__summit__:  The estimated peak center  
__numsamples__:  Number of samples with this peak (from the ~700 seqs)  
__total_signal__:  Total dhs signal on this peak  
__component__:  dhs component (Which cells component we find this peak?)   
__proportion__:   proportion of the component  
__sequence__  : DNA sequence replresenting 200 bp from the summit of this sequence   
__GM12878_ENCLB441ZZZ__  : Will be 1 case the cell has a peak  
__hESCT0_ENCLB449ZZZ__  : Will be 1 case the cell has a peak  
__K562_ENCLB843GMH__  : Will be 1 case the cell has a peak  
__HepG2_ENCLB029COU__  : Will be 1 case the cell has a peak  
__TAG__  : The final tag used to classify the cell  
__addtional_rep_with_this_peak__ : How many replicates for the cell specific peak presents this same peak  
__other_samples_with_this_peak_not_considering_reps__   : How many of the other total cells in the database has a peak in this sequence  
__data_label__ : train, validation (chr1) or test  (chr2)


___
