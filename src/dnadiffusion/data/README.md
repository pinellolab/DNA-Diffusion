# Data

## selected_K562_hESCT0_HepG2_GM12878_12k_sequences_per_group (Sequences in the Hg38 verions)

### Description

Thi dataset is a subset of sequences that are present in four different cell types (**GM12878_ENCLB441ZZZ** , **hESCT0_ENCLB449ZZZ**
,**K562_ENCLB843GMH** , **HepG2_ENCLB029COU**). Each cell type was selected using a ENCL sample (This specific ENCL sample has the biggest quality when compared with another samples of the same celltype- We can ask the selection criteria to Wouter). The first DHS selection method was check if the DHS is present in only one of the four initial samples.
To capture sequences strongly representing a given celltype, we sorted the sequences by the presence of this DHS peak in multiple experiments of the same celltype and reversed sorted to decreasce the presence of this peak in other experiments of the database (total set with ~700 experiments).

The final sequence number is balanced using the celltype with smallest number of sequences (GM12878). The number of remaining sequences is around ~12k per celltype. The data is divided in training , validation (chr1), test (chr2). Since we need to balance the class the sorting procedure will give us DHS regions that can represent better a unique celltype.

4 cell types link (~48 sequences- Single file - 12k seqs per cell)

-   [Four cell types table](https://drive.google.com/drive/folders/1dBeZIdJZQqaZUzCBVrz_Z2fAV9ePsw7h?usp=sharing)
-   File: DNA-Diffusion/dna-diffusion/data/selected_K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt

**dhs_id** : dhs id
**chr** : chr
**start** : start
**end** : end
**DHS_width** : The peak width
**summit**: The estimated peak center
**numsamples**: Number of samples with this peak (from the ~700 seqs)
**total_signal**: Total dhs signal on this peak
**component**: dhs component (Which cells component we find this peak?)
**proportion**: proportion of the component
**sequence** : DNA sequence replresenting 200 bp from the summit of this sequence .Sequences in the Hg38 verions
**GM12878_ENCLB441ZZZ** : Will be 1 case the cell has a peak
**hESCT0_ENCLB449ZZZ** : Will be 1 case the cell has a peak
**K562_ENCLB843GMH** : Will be 1 case the cell has a peak
**HepG2_ENCLB029COU** : Will be 1 case the cell has a peak
**TAG** : The final tag used to classify the cell
**addtional_rep_with_this_peak** : How many replicates for the cell specific peak presents this same peak
**other_samples_with_this_peak_not_considering_reps** : How many of the other total cells in the database has a peak in this sequence
**data_label** : train, validation (chr1) or test (chr2)

---
