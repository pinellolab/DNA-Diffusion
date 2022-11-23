from sourmash import MinHash
import pandas as pd
import numpy as np


def create_mini_hash_of_a_sequence(seq,minihash): 
  for k in seq:
    minihash.add_sequence(k)
  return minihash


def compare_two_sequences_and_return_similarity(seq:str, seq2:str, k:int, n:int):
  mh1 = MinHash(n=n, ksize=k)
  mh2 = MinHash(n=n, ksize=k)
  mh1 = create_mini_hash_of_a_sequence(seq,mh1)
  mh2 = create_mini_hash_of_a_sequence(seq2,mh2)
  similarity = round(mh1.similarity(mh2), 5)
  return similarity


def average_similarity(seq:pd.Series, seq2:pd.Series, number_of_hashes:int = 20000, k_sizes:list = [3,7,20]):
  average_similarities = []
  sequence_1 = sequence_1.sample(frac=1, random_state=42).reset_index(drop=True)
  sequence_1 = seq.tolist()
  sequence_2 = sequence_2.sample(frac=1, random_state=42).reset_index(drop=True)
  sequence_2 = seq2.tolist()
  for k in k_sizes:
    similarity = compare_two_sequences_and_return_similarity(sequence_1, sequence_2, k, number_of_hashes)
    average_similarities.append(similarity)
  average_similarities = np.array(average_similarities)
  average_similarity = round(average_similarities.mean(), 3)
  return average_similarity

