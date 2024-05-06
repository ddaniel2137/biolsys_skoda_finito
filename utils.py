# utils.py
import numpy as np
import pandas as pd
from typing import List, Dict

def preprocess_data(stats_stacked: Dict[str, Dict[str, List]], roles: List[str]) -> pd.DataFrame:
    """
    Preprocess nested data into a flat structure suitable for a DataFrame.
    """
    data = []
    for role in roles:
        for gen in range(len(stats_stacked['generation'][role])):
            entry = {
                'role': role,
                'generation': gen,
                'mean_fitness': stats_stacked['mean_fitness'][role][gen],
                'size': stats_stacked['size'][role][gen],
                'optimal_genotype': stats_stacked['optimal_genotype'][role][gen],
                # Flatten genotypes and fitnesses if they are stored as lists or arrays
                'genotypes': stats_stacked['genotypes'][role][gen].flatten(),
                'fitnesses': stats_stacked['fitnesses'][role][gen].flatten()
            }
            data.append(entry)
    return pd.DataFrame(data)

# Assuming pad_sizes is another utility function without Streamlit dependencies
def pad_sizes(sizes: List[np.ndarray], target_length: int) -> List[np.ndarray]:
    """
    Pad the sizes of numpy arrays in a list to match a target length.
    """
    padded_sizes = [np.pad(size_array, (0, max(0, target_length - len(size_array))), 
                           mode='constant', constant_values=0) 
                    for size_array in sizes]
    return padded_sizes

# Add any additional utility functions below