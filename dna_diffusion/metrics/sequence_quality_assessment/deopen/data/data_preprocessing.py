import pandas as pd
import argparse
import os
import math


def generate_pos_bed(data, sequence_length, output=None):
    """
    This function transforms a given DHS .csv file to the required .BED file in preparation for training Deopen.
    """
    data = data[['seqname', 'start', 'end']]
    starts = data['start'].tolist()
    ends = data['end'].tolist()
    starts, ends = trim_sequence_length(starts, ends, sequence_length)
    data['start'] = starts
    data['end'] = ends
    if output:
        data.to_csv(f'{output}/positive.bed', sep='\t', header=False, index=False)
        print(f"BED file succesfully saved to {output}/positive.bed")
    else:
        data.to_csv('positive.bed', sep='\t', header=False, index=False)
        path = os.path.dirname(os.path.abspath(__file__))
        print(f"BED file succesfully saved to {path}/positive.bed")


def trim_sequence_length(start_arr, end_arr, seq_len):
    """
    This functions centers the start and end positions of the sequences around the sequence centers and trims the
    sequences to the specified length.
    """
    starts = []
    ends = []
    for start, end in zip(start_arr, end_arr):
        act_len = int(end) - int(start)
        if act_len != seq_len:
            mid_idx = math.floor(act_len / 2)
            start_new = start + mid_idx - math.floor(seq_len / 2)
            end_new = start + mid_idx + math.ceil(seq_len / 2)
        starts.append(start_new)
        ends.append(end_new)
    return starts, ends


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None  # Supress unnecessary warnings

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', dest='data', type=str, help='The dataset (csv) to prepare for Deopen training.')
    parser.add_argument('-seq_len', dest='seq_len', type=int, default=200, help='The length of the sequence to use '
                                                                                'for training. Default is 200.')
    parser.add_argument('-out', help="Output location of the BED file.")
    args = parser.parse_args()

    # TEMPORARY FOR DEBUGGING PURPOSES #
    debug = False
    if not debug:
        df = pd.read_csv(args.data, sep="\t")
    else:
        df = pd.read_csv('train_all_classifier_light.csv', sep="\t")
    ####################################

    generate_pos_bed(df, args.seq_len, args.out)
