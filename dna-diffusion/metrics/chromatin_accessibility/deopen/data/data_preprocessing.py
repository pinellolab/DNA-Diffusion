import pandas as pd
import argparse
import os
import math


def generate_pos_bed(data, sequence_length, output=None):
    """
    This function transforms a given DHS .csv file to the required .BED file in preparation for training Deopen.
    """
    data = data[['seqname', 'start', 'end', 'summit']]
    starts, ends, summits = data['start'].tolist(), data['end'].tolist(), data['summit'].tolist()
    starts, ends = trim_sequence_length(starts, ends, summits, sequence_length)
    data['start'], data['end'] = starts, ends
    data = data.drop(columns=['summit'])
    data = delete_duplicate_entries(data)
    if output:
        data.to_csv(f'{output}/positive.bed', sep='\t', header=False, index=False)
        print(f"BED file successfully saved to {output}/positive.bed")
    else:
        data.to_csv('positive.bed', sep='\t', header=False, index=False)
        path = os.path.dirname(os.path.abspath(__file__))
        print(f"BED file successfully saved to {path}/positive.bed")

    return data


def trim_sequence_length(start_arr, end_arr, summit_arr, seq_len):
    """
    This functions centers the start and end positions of the sequences around the sequence summits and trims the
    sequences to the specified length.
    """
    starts = []
    ends = []
    for i, (start, end) in enumerate(zip(start_arr, end_arr)):
        act_len = int(end) - int(start)
        if act_len != seq_len:
            mid_idx = int(summit_arr[i])
            start_new = mid_idx - math.floor(seq_len / 2)
            end_new = mid_idx + math.ceil(seq_len / 2)
            starts.append(start_new)
            ends.append(end_new)
        else:
            starts.append(start)
            ends.append(end)
    return starts, ends


def delete_duplicate_entries(data):
    """
    This function deletes duplicate entries in the DHS .csv file.
    """
    return data.drop_duplicates(subset=['seqname', 'start', 'end'])


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None  # Supress unnecessary warnings

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', dest='data', type=str, help='The dataset (.csv) to prepare for Deopen training.')
    parser.add_argument('-seq_len', dest='seq_len', type=int, default=200, help='The length of the sequence to use '
                                                                                'for training. Default is 200.')
    parser.add_argument('-out', help="Output location of the BED file.")
    args = parser.parse_args()

    df = pd.read_csv(args.data, sep="\t")
    generate_pos_bed(df, args.seq_len, args.out)
