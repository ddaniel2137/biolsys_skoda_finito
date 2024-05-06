import numpy as np
from typing import List, Callable
from icecream import ic
import copy
from scipy.spatial.distance import cdist


class Population:
    genotypes: np.ndarray
    num_genes: int
    generation: int
    optimal_genotype: np.ndarray
    fitness_coefficient: float
    size: int
    max_population: int
    mutation_probability: float
    mutation_effect: float
    max_num_children: int
    interaction_value: float
    fitnesses: np.ndarray
    prev_mean_fitness: float
    prev_size: int
    mean_fitness: float
    interaction_value: float
    def __init__(self, init_population: int, num_genes: int, optimal_genotype: np.ndarray,
                 fitness_coefficient: float, max_population: int, mutation_probability: float,
                 mutation_effect: float, max_num_children: int, interaction_value: float, **kwargs):
        if kwargs.get("seed", None) is not None:
            np.random.seed(kwargs.get("seed"))
        self.genotypes = np.random.uniform(-1, 1, (init_population, num_genes))
        self.num_genes = num_genes
        self.generation = 1
        self.optimal_genotype = optimal_genotype
        self.fitness_coefficient = fitness_coefficient
        self.size = init_population
        self.max_population = max_population
        self.mutation_probability = mutation_probability
        self.mutation_effect = mutation_effect
        self.max_num_children = max_num_children
        self.fitnesses = self._evaluate_fitness()
        self.prev_mean_fitness = 0.0
        self.prev_size = 0
        self.mean_fitness = float(np.mean(self.fitnesses) if self.fitnesses.size > 0 else 0.0)
        self.interaction_value = interaction_value
        #ic(self.reproduce())
        #ic(self.evaluate(0.5, 10))
        #ic(self.mutate())
        #ic(self._evaluate_fitness())
        
        
    def _evaluate_fitness(self) -> np.ndarray:
        distances = cdist(self.genotypes, self.optimal_genotype.reshape(1, -1), metric='euclidean').ravel()
        #ic()
        fitnesses = np.exp(-distances / (2 * self.fitness_coefficient ** 2))
        #ic()
        return fitnesses
    def evaluate(self, mean_fitness_other: float, size_other: int):
        total_size = max(self.size + size_other, 1)
        #ic()
        genotypic_fitnesses = self._evaluate_fitness()
        #ic()
        freq_other = size_other / total_size
        freq_self = self.size / total_size
        interaction_fitnesses = mean_fitness_other * self.interaction_value * freq_other / max(freq_self, 1e-15)
        #ic()
        self.fitnesses = np.clip(genotypic_fitnesses + interaction_fitnesses, 0.0, 1.0)
        #ic()
        self.prev_mean_fitness = self.mean_fitness
        self.mean_fitness = float(np.mean(self.fitnesses) if self.fitnesses.size > 0 else 0.0)
    
    def mutate(self):
        mask = np.random.uniform(0, 1, self.size * self.num_genes) < self.mutation_probability
        #ic()
        mutations = mask * np.random.normal(0, self.mutation_effect, self.size * self.num_genes)
        #ic()
        self.genotypes += mutations.reshape(self.size, self.num_genes)
    
    def reproduce(self):
        indices = self.select()
        #ic()
        np.random.shuffle(indices)
        if indices.shape[0] % 2 != 0:
            indices = indices[:-1]
        parents1, parents2 = np.array_split(indices, 2)
        offspring_numbers = np.random.poisson(self.max_num_children, parents1.shape[0])
        parent_indices = np.repeat(np.arange(parents1.shape[0]), offspring_numbers)
        gene_indices = np.tile(np.arange(self.num_genes), (len(parent_indices), 1))
        genotypes_new_parents_choice = np.random.randint(0, 2, (len(parent_indices), self.num_genes))
        #ic()
        genotypes1 = self.genotypes[parents1[parent_indices]]
        #ic()
        genotypes2 = self.genotypes[parents2[parent_indices]]
        genotypes_new = np.where(genotypes_new_parents_choice, genotypes1, genotypes2)
        #ic()
        self.genotypes = np.concatenate([self.genotypes[indices], genotypes_new])[:self.max_population]
        self.generation += 1
        self.prev_size = self.size
        self.size = self.genotypes.shape[0]
        
    
    def select(self):
        selection_prob = np.exp(-self.size / self.max_population) * self.fitnesses
        #ic()
        return np.where(np.random.rand(self.size) < selection_prob)[0]
