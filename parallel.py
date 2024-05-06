# parallel.py
import concurrent.futures
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from typing import List, Dict, Any, Optional
import dill
from utils import preprocess_data
from environment import Environment, run_simulation
from icecream import ic
from functools import partial


def run_simulation_and_preprocess(params: Dict[str, Any], fixed_params: Dict[str, Any]) -> pd.DataFrame:
    env = Environment(**{**fixed_params, **params})
    _, stats_stacked = run_simulation(env, fixed_params['num_generations'])
    df = preprocess_data(stats_stacked, fixed_params['roles'])
    df = df.drop(columns=['fitnesses', 'genotypes'])
    fixed_param_columns = pd.DataFrame({param: [params[param]] * len(df) for param in params}, index=df.index)
    df = pd.concat([df, fixed_param_columns], axis=1)
    return df

def search_optimal_parameters_parallel(
        roles: List[str] = ['prey', 'predator'],
        num_populations: int = 2,
        num_generations: int = 200,
        scenario: str = 'global_warming',
        global_warming_scale: float = 1.0,
        global_warming_var: float = 0.05,
        meteor_impact_strategy: Optional[str] = None,
        meteor_impact_every: Optional[int] = None,
        meteor_impact_at: Optional[List[int]] = None,
        seed: int = 42,
        optimal_genotypes: Optional[np.ndarray] = None,
        num_genes: List[int] = [5, 5],
        init_populations: List[int] = [200, 200],
        max_num_children: List[int] = [2, 2],
        max_populations: List[int] = [10000, 10000],
        grid_params: Dict[str, Any] =  {
            'fitness_coefficients': (
                [0.1, 0.3, 0.5, 0.75, 1.0, 5.0, 10.0, 100.0, 1000.0],
                [0.1, 0.3, 0.5, 0.75, 1.0, 5.0, 10.0, 100.0, 1000.0],
            ),
            'interaction_values': (
                -1*[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            )
        }
    ):
    
    grid_params_set = {
        'fitness_coefficients': [0.9, 0.9],
        'mutation_effects': [0.1, 0.1],
        'interaction_values': [-0.4, 0.75],
        'mutation_probabilities': [0.3, 0.3]
    }
    #ic(grid_params_set)
    #ic(grid_params)
    for key in grid_params.keys():
        del grid_params_set[key]
    
    #ic(grid_params_set)
    #ic(grid_params)

    fixed_params = {
        'roles': roles,
        'num_populations': num_populations,
        'num_generations': num_generations,
        'scenario': scenario,
        'global_warming_scale': global_warming_scale,
        'global_warming_var': global_warming_var,
        'meteor_impact_strategy': meteor_impact_strategy,
        'meteor_impact_every': meteor_impact_every,
        'meteor_impact_at': meteor_impact_at,
        'seed': seed,
        'num_genes': num_genes,
        'optimal_genotypes': [np.zeros(num_gene) for num_gene in num_genes] if optimal_genotypes is None
        else optimal_genotypes,
        'init_populations': init_populations,
        'max_num_children': max_num_children,
        'max_populations': max_populations,
        **grid_params_set
    }

    # Extract tunable parameters from kwargs
    #param_mutable = {
    #    'fitness_coefficients': list(kwargs.get('fitness_coefficients', product([0.1, 0.3, 0.5, 0.75, 1.0, 5.0, 10.0,
    #                                                                             100.0], repeat=2))),
    #    'mutation_effects': list(kwargs.get('mutation_effects', product([0.1, 0.3, 0.5, 0.75, 1.0], repeat=2))),
    #    'interaction_values': list(kwargs.get('interaction_values', product([-0.7, -0.5, -0.3, -0.1,
    #                                                                         0.0], [0.0, 0.3, 0.5, 0.7, 0.9]))),
    #    'mutation_probabilities': list(
    #        kwargs.get('mutation_probabilities', product([0.1, 0.25, 0.5, 0.75, 0.9], repeat=2))),
    #}
    #ic(grid_params_set)
    param_mutable = {k: list(product(v1, v2)) for k, (v1, v2) in grid_params.items()}
    #ic(param_mutable)
    #ic(-1*[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # Create all combinations of parameters
    param_list = list(ParameterGrid(param_mutable))


        # Run simulations in parallel
    outputs= []
    num_cores = multiprocessing.cpu_count() // 2
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = executor.map(
            partial(run_simulation_and_preprocess, fixed_params=fixed_params),
            param_list,
            chunksize=1000
        )
        for i, result in enumerate(results):
            try:
                outputs.append(result)
            except Exception as e:
                ic(f"Error in simulation {i}: {e}")
                continue

    # Combine results into a single DataFrame
    return pd.concat(outputs, ignore_index=True)

# Example usage:
# optimal_params_result = search_optimal_parameters_parallel(fitness_coefficients=[0.1, 0.2, 0.3])