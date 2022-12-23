import time
import os
import pickle as pkl


def inference(one_hot_seqs, model):
    start_time = time.perf_counter()
    for gene, seq in one_hot_seqs.items():
        # check if gene is already in outputs folder
        if not os.path.exists(f'outputs/{gene}.pkl'):
            print(f"Running Enformer inference for {gene}")
            output = model.forward(seq)
            with open(f'outputs/{gene}.pkl', 'wb') as f:
                pkl.dump(output, f)
            f.close()
        else:
            print(f'Output for {gene} already exists. Skipping...')
    end_time = time.perf_counter()
    print(f'Inference took {round(end_time - start_time, 2)} seconds')