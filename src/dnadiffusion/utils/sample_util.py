import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dnadiffusion.utils.utils import convert_to_seq


def create_sample(
    model: torch.nn.Module,
    cell_type: int,
    sample_bs: int,
    conditional_numeric_to_tag: dict,
    number_of_samples: int = 1000,
    cond_weight_to_metric: int = 0,
    save_timesteps: bool = False,
    save_dataframe: bool = False,
    generate_attention_maps: bool = False,
    sequence_length: int = 200,
) -> None:
    nucleotides = ["A", "C", "G", "T"]
    final_sequences = []
    num_batches = number_of_samples // sample_bs
    for n_a in tqdm(range(num_batches)):
        sampled = torch.from_numpy(np.array([cell_type] * sample_bs))
        classes = sampled.float().to(model.device)

        if generate_attention_maps:
            sampled_images, cross_att_values = model.sample_cross(
                classes, (sample_bs, 1, 4, sequence_length), cond_weight_to_metric
            )
            # save cross attention maps in a numpy array
            np.save(f"cross_att_values_{conditional_numeric_to_tag[cell_type]}.npy", cross_att_values)

        else:
            sampled_images = model.sample(classes, (sample_bs, 1, 4, sequence_length), cond_weight_to_metric)

        if save_timesteps:
            seqs_to_df = {}
            for en, step in enumerate(sampled_images):
                seqs_to_df[en] = [convert_to_seq(x, nucleotides) for x in step]
            final_sequences.append(pd.DataFrame(seqs_to_df))

        if save_dataframe:
            # Only using the last timestep
            for en, step in enumerate(sampled_images[-1]):
                final_sequences.append(convert_to_seq(step, nucleotides))
        else:
            for n_b, x in enumerate(sampled_images[-1]):
                seq_final = f">seq_test_{n_a}_{n_b}\n" + "".join(
                    [nucleotides[s] for s in np.argmax(x.reshape(4, sequence_length), axis=0)]
                )
                final_sequences.append(seq_final)

    if save_timesteps:
        # Saving dataframe containing sequences for each timestep
        pd.concat(final_sequences, ignore_index=True).to_csv(
            f"data/outputs/{conditional_numeric_to_tag[cell_type]}.txt",
            header=True,
            sep="\t",
            index=False,
        )
        return

    if save_dataframe:
        # Saving list of sequences to txt file
        with open(f"data/outputs/{conditional_numeric_to_tag[cell_type]}.txt", "w") as f:
            f.write("\n".join(final_sequences))
        return
