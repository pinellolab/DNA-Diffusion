import os
import pandas as pd
from dnadiffusion.utils.sample_util import extract_motifs

def test_extract_motifs():
    # Defining example motif data
    test_bed = [
    'sequence_0\tpfmscan\tmisc_feature\t123\t130\t8.931999505513964\t-\t.\tmotif_name "MA0607.1_Bhlha15" ; motif_instance "ACATATGC"',
    'sequence_0\tpfmscan\tmisc_feature\t34\t44\t9.44728482904739\t+\t.\tmotif_name "MA1683.1_FOXA3" ; motif_instance "GAGTAAACAGA"',
    'sequence_0\tpfmscan\tmisc_feature\t36\t43\t6.450593195402899\t+\t.\tmotif_name "MA1489.1_FOXN3" ; motif_instance "GTAAACAG"',
    'sequence_0\tpfmscan\tmisc_feature\t35\t45\t10.249462278895773\t-\t.\tmotif_name "MA0480.1_Foxo1" ; motif_instance "AGTAAACAGAA"',
    'sequence_0\tpfmscan\tmisc_feature\t34\t44\t10.965594517325535\t+\t.\tmotif_name "MA0593.1_FOXP2" ; motif_instance "GAGTAAACAGA"',
    'sequence_0\tpfmscan\tmisc_feature\t107\t116\t11.634852301936267\t+\t.\tmotif_name "UN0330.1_ZNF513" ; motif_instance "AGAGGAAGAG"',
    'sequence_0\tpfmscan\tmisc_feature\t84\t96\t11.572869058140274\t-\t.\tmotif_name "UN0332.1_ZNF534" ; motif_instance "AATGGGCAAGAAC"',
    'sequence_1\tpfmscan\tmisc_feature\t139\t151\t9.80846582352393\t+\t.\tmotif_name "MA0800.1_EOMES" ; motif_instance "AAGGTGTTAACAT"',
    'sequence_1\tpfmscan\tmisc_feature\t101\t111\t10.183240817371155\t+\t.\tmotif_name "MA0002.2_RUNX1" ; motif_instance "GTCTGTGGTTA"',
    'sequence_1\tpfmscan\tmisc_feature\t62\t81\t12.509519971299817\t+\t.\tmotif_name "MA0080.5_SPI1" ; motif_instance "TAAAATGAGGAACTGAAGTA"'
]

    # Writing to a bed file
    with open('synthetic_motifs.bed', 'w') as f:
        f.write('# GimmeMotifs version 0.18.0\n')
        f.write('# Input: synthetic_motifs.fasta\n')
        f.write('# Motifs: JASPAR2020_vertebrates\n')
        f.write('# FPR: 0.01 (hg38)\n')
        f.write('# Scoring: logodds score\n')
        for line in content:
            f.write(line + '\n')

    # Running extract_motifs function
    df_motifs_count_syn = extract_motifs(test_bed)

    # Asserting that the output is a dataframe
    assert isinstance(df_motifs_count_syn, pd.DataFrame)

    # Asserting that the output file is not empty
    assert df_motifs_count_syn.shape[0] > 0

    # Asserting function created the expected files
    assert os.path.exists('synthetic_motifs.bed')
    assert os.path.exists('syn_results_motifs.bed')

    # Removing files created by the function
    os.remove('synthetic_motifs.bed')
    os.remove('syn_results_motifs.bed')
