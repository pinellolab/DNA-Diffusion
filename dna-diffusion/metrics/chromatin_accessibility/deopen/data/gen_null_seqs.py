# ==========================================================
#  Title    : Generating negative data from input .bed file
#  Author   : Manuel Tognon
#  Date     : 2-11-2022
# ==========================================================

"""
Script to compute random background set of sequences from an input BED file.
The computed random sequences preserve the GC content, length, and repeats ratio
of the sequences stored in the input BED file.
The random sequences are stored in the FASTA file provided as argument by the
user.
"""

from typing import Any, Dict, List, Optional
from pybedtools import BedTool
from Bio.SeqUtils import GC
from tqdm import tqdm

import numpy as np

import argparse
import os


def parse_commandline() -> argparse.ArgumentParser:
    """The function parses the command line arguments.
    ...
    Parameters
    ----------
    None
    Returns
    -------
    argparse.ArgumentParser
    """

    # create parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Script to compute random background set of sequences from an input BED file",
        usage="\n\tpython3 %(prog)s --bed <BEDFILE> --genome <GENOME> --mask <MASK> --out <OUFTILE>",
    )
    # add arguments
    parser.add_argument(
        "--bed", type=str, metavar="BEDFILE", help="Foreground BED file"
    )
    parser.add_argument(
        "--genome", type=str, metavar="GENOME", help="Genome FASTA file"
    )
    parser.add_argument(
        "--mask", type=str, metavar="MASK", help="Repeat Mask file (BED format)"
    )
    parser.add_argument("--out", type=str, metavar="OUTFILE", help="Output file")
    parser.add_argument(
        "--maxiter",
        type=int,
        metavar="MAXITER",
        help="Number of maximum iterations",
        default=100,
        const=100,
        nargs="?",
    )
    parser.add_argument(
        "--fold",
        type=float,
        metavar="FOLD",
        help="Background sequence set fold size",
        default=1,
        const=1,
        nargs="?",
    )
    parser.add_argument(
        "--gc",
        type=float,
        metavar="GC THRESHOLD",
        help="GC content tolerance threshold",
        default=0.02,
        const=0.02,
        nargs="?",
    )
    parser.add_argument(
        "--length",
        type=float,
        metavar="LENGTH THRESHOLD",
        help="Sequence length tolerance threshold",
        default=0.02,
        const=0.02,
        nargs="?",
    )
    parser.add_argument(
        "--repeats",
        type=float,
        metavar="REPEATS THRESHOLD",
        help="Repeats content tolerance threshold",
        default=0.02,
        const=0.02,
        nargs="?",
    )
    parser.add_argument(
        "--outformat",
        type=str,
        metavar="OUTPUT FORMAT",
        help="Output file format (FASTA or BED)",
        default="fasta",
        const="fasta",
        nargs="?",
    )
    args = parser.parse_args()  # parse arguments
    __check_args_consistency(args)  # check args consistency
    return args


def __check_args_consistency(args: argparse.ArgumentParser) -> None:
    """(PRIVATE)
    Check the consistency of the input command line arguments.
    ...
    Parameters
    ----------
    args
        Input command line arguments
    Returns
    -------
    None
    """

    if not os.path.isfile(args.bed):
        raise FileNotFoundError(f"Unable to locate {args.bed}")
    if not os.path.isfile(args.genome):
        raise FileNotFoundError(f"Unable to locate {args.genome}")
    if not os.path.isfile(args.mask):
        raise FileNotFoundError(f"Unable to locate {args.mask}")
    if args.maxiter < 1:
        raise ValueError(f"Forbidden number of maximum iterations ({args.maxiter})")
    if args.fold < 1:
        raise ValueError(f"Forbidden fold value ({args.maxiter})")
    if args.gc < 0:
        raise ValueError(f"Forbidden GC content tolerance threshold ({args.gc})")
    if args.length < 0:
        raise ValueError(
            f"Forbidden sequence length tolerance threshold ({args.length})"
        )
    if args.repeats < 0:
        raise ValueError(
            f"Forbidden repeats content tolerance threshold ({args.repeats})"
        )
    if args.outformat.upper() != "FASTA" and args.outformat.upper() != "BED":
        raise ValueError(
            f"Forbidden output file format ({args.outformat}). Please, choose between FASTA or BED"
        )


def recover_sequences(bed: BedTool, genome: str) -> List[List[str]]:
    """Parse the input BED file and create a ```BedTool``` object and extract
    the sequences corresponding to the genomic coordinates defined in the input
    BED file.
    ...
    Parameters
    ----------
    bedfile
        Input BED file
    genome
        Reference genome
    Returns
    -------
    List[List[str]]
    """

    if not isinstance(bed, BedTool):
        raise TypeError(f"Expected {BedTool.__name__}, got {type(bed).__name__}")
    if not isinstance(genome, str):
        raise TypeError(f"Expected {str.__name__}, got {type(genome).__name__}")
    if not os.path.isfile(genome):
        raise FileNotFoundError(f"Unable to locate {genome}")
    bed = bed.sequence(fi=genome)  # recover sequences from the input genome
    # store the extracted sequences
    handle = open(bed.seqfn, mode="r")
    seqnames = [line.strip()[1:] for line in handle if line.startswith(">")]
    handle = open(bed.seqfn, mode="r")
    sequences = [
        line.strip()
        for line in handle
        if not line.startswith(">") and bool(line.strip())
    ]
    if len(seqnames) != len(bed):
        raise ValueError(
            f"The input BED file contains duplicated coordinates: remove them before running."
        )
    sequences = [seqnames, sequences]
    return sequences


def fasta_contig_length(
        fastafile: str, fasta_index: Optional[str] = None
) -> Dict[str, int]:
    """The function counts the number of base pairs in each contig of the input
    FASTA file. The contig lengths are stored in a dictionary, with the contig
    names as keys and the lengths as values.
    The function searches for the FASTA index (FAI) file to speed-up the
    computations. By default the function assumes that the FAI file is in the
    same directory of the input FASTA. If the FAI location is different the user
    can provide it as an input argument to the function.
    TODO: compute FAI if not found
    ...
    Parameters
    ----------
    fastafile
        Input FASTA file
    fasta_index
        FAI file
    Returns
    -------
    Dict[str, int]
    """

    if not isinstance(fastafile, str):
        raise TypeError(f"Expected {str.__name__}, got {type(fastafile).__name__}")
    if not os.path.isfile(fastafile):
        raise FileNotFoundError(f"Unable to locate {fastafile}")
    if fasta_index is not None:  # FAI location provided
        if not isinstance(fasta_index, str):
            raise TypeError(
                f"Expected {str.__name__}, got {type(fasta_index).__name__}"
            )
        if not os.path.isfile(fasta_index):
            raise FileNotFoundError(f"Unable to locate {fasta_index}")
    else:
        fasta_index = f"{fastafile}.fai"
    handle = open(fasta_index, mode="r")  # read FAI
    contig_lengths = {}  # store contig lengths
    for line in handle:
        # the first two columns contain the information in which we are interested
        contig, length = line.strip().split()[:2]
        contig_lengths[contig] = int(length)
    return contig_lengths


def gc_content(sequences: List[List[str]]) -> List[float]:
    """Compute the GC content of each sequence in the input dataset. The GC
    content is provided as probability (value between 0 and 1).
    ...
    Parameters
    ----------
    sequences
        Input sequences
    Returns
    -------
    List[float]
    """

    if not isinstance(sequences, list):
        raise TypeError(f"Expected {list.__name__}, got {type(sequences).__name__}")
    gc = [
        (GC(sequence) / 100) for sequence in sequences[1]
    ]  # GC() returns percentage values
    assert len(gc) == len(sequences[1])
    return gc


def compute_repeats_ratio(
        sequences: List[List[str]], repeat_mask: BedTool, bed: BedTool
) -> Dict[str, float]:
    """The function computes the percentage of each genomic coordinate in the
    input BED file overlapping the repetitive elements provided as genomic
    coordinates in a second BED file.
    ...
    Parameters
    ----------
    sequences
        Input sequences
    repeat_mask
        Repeat mask (BED file)
    bed
        Input BED file
    Returns
    -------
    Dict[str, float]
    """

    if not isinstance(sequences, list):
        raise TypeError(f"Expected {list.__name__}, got {type(sequences).__name__}")
    if not isinstance(repeat_mask, BedTool):
        raise TypeError(
            f"Expected {BedTool.__name__}, got {type(repeat_mask).__name__}"
        )
    if not isinstance(bed, BedTool):
        raise TypeError(f"Expected {BedTool.__name__}, got {type(bed).__name__}")
    seqnames = np.array(sequences[0])
    # store repeats in each genomic coordinate defined in the BED file
    repeats = np.array([0.0 for seqname in seqnames])
    # recover regions width
    widths = [len(sequence) for sequence in sequences[1]]
    # compute overlaps between query BED and repeat mask
    overlaps = bed.intersect(repeat_mask, wo=True)
    for overlap in overlaps:
        ov_width = float(overlap[-1])  # overlap width is the last field
        # recover the original genomic region
        query = ":".join([overlap[0], "-".join([overlap[1], overlap[2]])])
        query_idxs = np.where(seqnames == query)[0]
        repeats[query_idxs] += ov_width
    # compute repeats ratio for each genomic region in the original BED
    for i in range(len(repeats)):
        repeats[i] /= widths[i]
    return list(repeats)


def __sequence_position(
        positions: np.ndarray, sequences_positions: List[int]
) -> List[int]:
    """(PRIVATE)
    Pick random sequence positions, while maintaining chromosome location.
    ...
    Parameters
    ----------
    positions
        Randomly picked sequence positions
    sequences_positions
        Cumulative sum array of sequences length
    Returns
    -------
    List[int]
    """

    if not isinstance(positions, np.ndarray):
        raise TypeError(
            f"Expected {np.ndarray.__name__}, got {type(positions).__name__}"
        )
    if not isinstance(sequences_positions, list):
        raise TypeError(
            f"Expected {list.__name__}, got {type(sequences_positions).__name__}"
        )
    idxs_sorted = np.argsort(positions)  # sorted array indexes
    positions = sorted(positions)  # sort positions
    result = [None for _ in range(len(positions))]
    j = 0
    for i in range(len(positions)):
        while positions[i] > sequences_positions[j]:
            j += 1
        result[idxs_sorted[i]] = j
    return result


def generate_random_sequences(
        sequences_len: List[int], contigs_len: Dict[str, int]
) -> BedTool:
    """The function generates random genomic coordinates from the original
    coordinates stored in the input BED file.
    ...
    Parameters
    ----------
    sequences_len
        Sequences length
    contigs_len
        Original contigs length
    Returns
    -------
    BedTool
    """

    if not isinstance(sequences_len, list):
        raise TypeError(f"Expected {list.__name__}, got {type(sequences_len).__name__}")
    if not isinstance(contigs_len, dict):
        raise TypeError(f"Expected {dict.__name__}, got {type(contigs_len).__name__}")
    # compute contigs length cumulative sum
    contigs_len_cumsum = np.cumsum(list(contigs_len.values()))
    position_limit = int(max(contigs_len_cumsum))
    contigs_len_cumsum = list(contigs_len_cumsum) + [int(1e12)]
    contigs_len_cumsum0 = np.array([0] + contigs_len_cumsum)
    # random sampled positions
    positions_rnd = np.random.choice(position_limit, len(sequences_len), replace=True)
    chrom_pos1 = __sequence_position(positions_rnd, contigs_len_cumsum)
    chrom_pos2 = __sequence_position(
        positions_rnd + np.array(sequences_len), contigs_len_cumsum
    )
    diff_positions = np.where(np.array(chrom_pos1) != np.array(chrom_pos2))[0]
    while len(diff_positions) > 0:
        positions_rnd[diff_positions] = np.random.choice(
            position_limit, len(diff_positions), replace=True
        )
        chrom_pos1 = __sequence_position(positions_rnd, contigs_len_cumsum)
        chrom_pos2 = __sequence_position(
            positions_rnd + np.array(sequences_len), contigs_len_cumsum
        )
        diff_positions = np.where(np.array(chrom_pos1) != np.array(chrom_pos2))[0]
    # pick random genomic coordinates
    seqnames = np.array(list(contigs_len.keys()))
    seqnames_sampled = seqnames[chrom_pos1]
    starts = positions_rnd - contigs_len_cumsum0[chrom_pos1]
    shuffle_coordinates = [
        f"{seqnames_sampled[i]}\t{start}\t{start + sequences_len[i]}\n"
        for i, start in enumerate(starts)
    ]
    # create BedTool object to store the random genomic coordinates
    shuffle_bed = BedTool(shuffle_coordinates)
    assert len(shuffle_bed) == len(sequences_len)
    return shuffle_bed


def __extract_elements(x: List[Any], y: List[int]) -> List[Any]:
    """(PRIVATE)
    The function reproduce R vector subscripting using other vectors.
    ...
    Parameters
    ----------
    x
        Input list
    y
        Indexes list
    Returns
    -------
    List[Any]
    """

    if not isinstance(x, np.ndarray):
        raise TypeError(f"Expected {np.ndarray.__name__}, got {type(x).__name__}")
    if not isinstance(y, list):
        raise TypeError(f"Expected {list.__name__}, got {type(y).__name__}")
    if not all([isinstance(e, int) or e is None for e in y]):
        raise TypeError(
            f"All index values must be either of type {int.__name__} or NoneType"
        )
    result = [None for _ in range(len(x))]  # result list
    for i, idx in enumerate(y):
        if idx is not None:
            result[i] = x[idx]
    return result


def __switch_index(x: np.ndarray, sort_idxs: np.ndarray, y: List[Any]) -> np.ndarray:
    """(PRIVATE)
    Switch the elements of the input array with those given in the second input
    array.
    ...
    Parameters
    ----------
    x
        Input array
    sort_idxs
        Sorted x indexes
    y
        Input array
    Returns
    -------
    np.ndarray
    """

    if not isinstance(x, np.ndarray):
        raise TypeError(f"Expected {np.ndarray.__name__}, got {type(x).__name__}")
    if not isinstance(sort_idxs, np.ndarray):
        raise TypeError(
            f"Expected {np.ndarray.__name__}, got {type(sort_idxs).__name__}"
        )
    if not isinstance(y, list):
        raise TypeError(f"Expected {list.__name__}, got {type(y).__name__}")
    # substitute NoneType elements with -1 character
    x_none = np.array([e if e is not None else -1 for e in x])
    y_none = np.array([e if e is not None else -1 for e in y])
    # switching arrays elements
    x_none[sort_idxs] = y_none
    return x_none


def match_sequence_features(
        gc_ref: List[float],
        gc_shuffle: List[float],
        len_ref: List[int],
        len_shuffle: List[int],
        rr_ref: List[float],
        rr_shuffle: List[float],
        gc_threshold: Optional[float] = 0.02,
        len_threshold: Optional[float] = 0.02,
        rr_threshold: Optional[float] = 0.02,
) -> np.ndarray:
    """
    ...
    Parameters
    ----------
    gc_ref
        GC content percentage of original sequences
    gc_shuffle
        GC content of shuffled sequences
    len_ref
        Lengths of original sequences
    len_shuffle
        Lengths of shuffled sequences
    rr_ref
        Repeats ratio of original sequences
    rr_shuffle
        Repeats ratio of shuffled sequences
    gc_threshold
        GC content threshold
    len_thershold
        Sequence length threshold
    rr_threshold
        Repeats ratio threshold
    Returns
    -------
    np.ndarray
    """

    if not isinstance(gc_ref, list):
        raise TypeError(f"Expected {list.__name__}, got {type(gc_ref).__name__}")
    if not isinstance(gc_shuffle, list):
        raise TypeError(f"Expected {list.__name__}, got {type(gc_shuffle).__name__}")
    if not isinstance(len_ref, list):
        raise TypeError(f"Expected {list.__name__}, got {type(len_ref).__name__}")
    if not isinstance(len_shuffle, list):
        raise TypeError(f"Expected {list.__name__}, got {type(len_shuffle).__name__}")
    if not isinstance(rr_ref, list):
        raise TypeError(f"Expected {list.__name__}, got {type(rr_ref).__name__}")
    if not isinstance(rr_shuffle, list):
        raise TypeError(f"Expected {list.__name__}, got {type(rr_shuffle).__name__}")
    if not isinstance(gc_threshold, float):
        raise TypeError(f"Expected {float.__name__}, got {type(gc_threshold).__name__}")
    if not isinstance(len_threshold, float):
        raise TypeError(f"Expected {float.__name__}, got {type(gc_threshold).__name__}")
    if not isinstance(rr_threshold, float):
        raise TypeError(f"Expected {float.__name__}, got {type(gc_threshold).__name__}")
    len_threshold = np.array(len_ref) * len_threshold
    # sort GC content
    ref_sort_idxs = np.argsort(gc_ref)
    shuffle_sort_idxs = np.argsort(gc_shuffle)
    # sort the lists accordingly
    gc_ref = np.array(gc_ref)[ref_sort_idxs]
    gc_shuffle = np.array(gc_shuffle)[shuffle_sort_idxs]
    len_ref = np.array(len_ref)[ref_sort_idxs]
    len_shuffle = np.array(len_shuffle)[shuffle_sort_idxs]
    rr_ref = np.array(rr_ref)[ref_sort_idxs]
    rr_shuffle = np.array(rr_shuffle)[shuffle_sort_idxs]
    gc_shuffle = np.array(list(gc_shuffle) + [int(1e10)])
    # sort the sequence length thresholds accordingly
    len_threshold = len_threshold[ref_sort_idxs]
    n_ref = len(ref_sort_idxs)
    n_shuffle = len(shuffle_sort_idxs)
    match_ref = [None for _ in range(n_ref)]
    match_shuffle = [0 for _ in range(n_shuffle)]
    j = 0
    for i in tqdm(range(n_ref)):
        gc_r = gc_ref[i]
        len_r = len_ref[i]
        rr_r = rr_ref[i]
        while gc_r - gc_shuffle[j] > gc_threshold:
            j += 1
        if j <= n_shuffle:
            jj = j
            while gc_shuffle[jj] - gc_r <= gc_threshold:
                if (
                        match_shuffle[jj] == 0
                        and np.abs(len_r - len_shuffle[jj] <= len_threshold[i])
                        and np.abs(rr_r - rr_shuffle[jj] <= rr_threshold)
                ):
                    match_shuffle[jj] = i
                    match_ref[i] = jj
                    if jj == j:
                        j += 1
                    break
                jj += 1
        else:
            break
    # get matching shuffling sequences indexes
    x = __extract_elements(shuffle_sort_idxs, match_ref)
    result = __switch_index(np.array([None for _ in range(n_ref)]), ref_sort_idxs, x)
    return result


def extend(lst: List[Any], fold: Optional[float] = 1) -> List[Any]:
    """Replicate the elements of a list up to list length by ```fold``` value.
    ...
    Parameters
    ----------
    lst
        Input list
    fold
        Fold
    Returns
    -------
    List[Any]
    """

    if not isinstance(lst, list):
        raise TypeError(f"Expected {list.__name__}, got {type(lst).__name__}")
    if not isinstance(fold, float) and not isinstance(fold, int):
        raise TypeError(
            f"Expected {float.__name__} or {int.__name__}, got {type(fold).__name__}"
        )
    outlength = int(len(lst) * fold)
    counter = 0
    result = []  # result list
    while counter < outlength:
        for e in lst:
            result.append(e)
            counter += 1
            if counter == outlength:  # reached the desired length
                break
    assert len(result) == outlength
    return result


def __write_sequences(bed: BedTool, genome: str, outfile: str) -> None:
    """(PRIVATE)
    Write a FASTA file from the input BED file containing genomic coordinates.
    ...
    Parameters
    ----------
    bed
        Input BED file
    genome
        Genome
    outfile
        Output filename
    Returns
    -------
    None
    """

    if not isinstance(bed, BedTool):
        raise TypeError(f"Expected {BedTool.__name__}, got {type(bed).__name__}")
    if not isinstance(genome, str):
        raise TypeError(f"Expected {str.__name__}, got {type(genome).__name__}")
    if not os.path.isfile(genome):
        raise FileNotFoundError(f"Unable to locate {genome}")
    if not isinstance(outfile, str):
        raise TypeError(f"Expected {str.__name__}, got {type(outfile).__name__}")
    # extract sequences
    bed.getfasta(fi=genome, fo=outfile)
    # check that the output file has been correctly created
    assert os.path.isfile(outfile) and os.stat(outfile).st_size > 0


def random_sequences() -> None:
    """Compute random genomic sequences starting from an input BED file. The
    random sequences match the original genomic intervals by GC content, sequence
    length, and repeats ratio.
    The computed random sequences are stored in a FASTA file specified in input
    by the user.
    ...
    Parameters
    ----------
    None
    Returns
    -------
    None
    """

    # parse command line arguments
    args = parse_commandline()
    # parse the input BED and the repeats BED
    bed = BedTool(args.bed)  # input BED
    mask = BedTool(args.mask)  # mask BED
    # recover the input BED sequences
    sequences_bed = recover_sequences(bed, args.genome)
    # compute BED sequences length
    sequences_bed_len = [len(sequence) for sequence in sequences_bed[1]]
    # compute genome chromosome lengths (or contig lengths when --genome is not a genome)
    contigs_len = fasta_contig_length(args.genome)
    # compute GC content of input BED sequences
    gc_bed = gc_content(sequences_bed)
    # compute repeats ratio of input BED sequences
    repeats_ratio_bed = compute_repeats_ratio(sequences_bed, mask, bed)
    # start sequence shuffling
    unmatched_sequences = [i for i in range(len(sequences_bed[0]) * args.fold)]
    # extend sequence features arrays accordingly to fold size
    gc_bed_e = extend(gc_bed, fold=args.fold)
    sequences_bed_len_e = extend(sequences_bed_len, fold=args.fold)
    repeats_ratio_bed_e = extend(repeats_ratio_bed, fold=args.fold)
    outbed_rnd = []  # output BED file storing random sequences
    for _ in range(args.maxiter):
        if len(unmatched_sequences) > 0:
            bed_rnd = generate_random_sequences(
                list(np.array(sequences_bed_len_e)[unmatched_sequences]), contigs_len
            )
            bed_rnd.sequence(fi=args.genome)  # recover random sequences
            sequences_rnd = recover_sequences(bed_rnd, args.genome)
            sequences_rnd_len = [len(sequence) for sequence in sequences_rnd[1]]
            # compute random sequence features
            gc_rnd = gc_content(sequences_rnd)
            repeats_ratio_rnd = compute_repeats_ratio(sequences_rnd, mask, bed_rnd)
            match = match_sequence_features(
                gc_bed_e,
                gc_rnd,
                sequences_bed_len_e,
                sequences_rnd_len,
                repeats_ratio_bed_e,
                repeats_ratio_rnd,
                args.gc,
                args.length,
                args.repeats,
            )
            idxs = [i for i, m in enumerate(match) if m is not None]
            if len(idxs) > 0:
                for i, line in enumerate(bed_rnd):
                    if i in match:
                        outbed_rnd.append(str(line))
                unmatched_sequences = np.delete(unmatched_sequences, idxs)
    bed_rnd = BedTool("".join(outbed_rnd), from_string=True)
    if args.outformat.upper() == "FASTA":  # write the sequences to a FASTA file
        __write_sequences(bed_rnd, args.genome, args.out)
    else:  # write sequences to a BED file
        bed_rnd.saveas(f"{args.out}")


# -> script entry point <-
if __name__ == "__main__":
    random_sequences()
