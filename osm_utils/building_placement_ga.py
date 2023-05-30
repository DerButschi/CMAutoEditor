# Copyright (C) 2023  Nicolas Möser

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Tuple
import numpy as np
from numpy.random import default_rng
from collections import OrderedDict
from itertools import product

def get_score_table(board: np.ndarray, tokens: OrderedDict):
    # score_table = np.zeros((board.shape[0], board.shape[1], len(tokens.keys())))

    pattern_shapes = [tokens[key]['pattern'].shape for key in tokens]
    max_pattern_rows = max([ps[0] for ps in pattern_shapes])
    max_pattern_cols = max([ps[1] for ps in pattern_shapes])

    # compatibility_tensor = np.zeros((len(tokens.keys()), len(tokens.keys()), 2 * max_pattern_rows - 1, 2 * max_pattern_cols - 1))
    # for i_token in range(len(tokens.keys())):
    #     token_i_id = list(tokens.keys())[i_token]
    #     pattern_i = tokens[token_i_id]['pattern']
    #     pattern_check = np.zeros((max_pattern_rows + 2 * (max_pattern_rows - 1), max_pattern_cols + 2 * (max_pattern_cols - 1)))
    #     pattern_check[max_pattern_rows - 1:max_pattern_rows - 1 + pattern_i.shape[0], 
    #                     max_pattern_cols - 1:max_pattern_cols - 1 + pattern_i.shape[1]] = pattern_i
    #     for j_token in range(len(tokens.keys())):
    #         for i_row in range(2 * max_pattern_rows - 1):
    #             for i_col in range(2 * max_pattern_cols - 1):
    #                 token_j_id = list(tokens.keys())[j_token]
    #                 pattern_j = tokens[token_j_id]['pattern']

    #                 pattern_check_extract = pattern_check[i_row:i_row + pattern_j.shape[0], i_col:i_col + pattern_j.shape[1]]
    #                 if (pattern_check_extract[pattern_j > 0] == 0).all():
    #                     compatibility_tensor[i_token, j_token, i_row, i_col] = 1

    score_table = []
    for i_row, i_col, i_token in product(range(board.shape[0]), range(board.shape[1]), range(len(tokens.keys()))):
        token_id = list(tokens.keys())[i_token]
        pattern = tokens[token_id]['pattern']
        board_extract = board[i_row:i_row + pattern.shape[0], i_col:i_col + pattern.shape[1]]

        if board_extract.shape != pattern.shape:
            # score_table[i_row, i_col, i_token] = -np.inf
            continue
        elif (board_extract[pattern > 0] == -1).any():
            # score_table[i_row, i_col, i_token] = -np.inf
            continue
        else:
            score = -1
            score -= np.count_nonzero((board_extract > 0) & (pattern == 0)) * 2
            score -= np.count_nonzero((board_extract == 0) & (pattern > 0)) * 2

            _, id_counts = np.unique(board_extract[(pattern > 0) & (board_extract > 0)], return_counts=True)
            id_counts = sorted(id_counts, key=lambda x: -x)
            if len(id_counts) > 0:
                score += id_counts[0] * 2
            if len(id_counts) > 1:
                score -= np.sum(id_counts[1:]) * 2

            if score > 0:
                score_table.append([i_row, i_col, i_token, score])

    return np.array(score_table)

def get_collision_matrix(board: np.ndarray, score_table: np.ndarray, tokens: OrderedDict):
    collision_matrix = np.zeros((score_table.shape[0], score_table.shape[0]), dtype=bool)

    pattern_shapes = [tokens[key]['pattern'].shape for key in tokens]
    max_pattern_rows = max([ps[0] for ps in pattern_shapes])
    max_pattern_cols = max([ps[1] for ps in pattern_shapes])

    for i in range(score_table.shape[0]):
        i_row, i_col = score_table[i, 0:2]
        token_i_id = list(tokens.keys())[score_table[i, 2]]
        pattern_i = tokens[token_i_id]['pattern']
        pattern_check = np.zeros((max_pattern_rows + 2 * (max_pattern_rows - 1), max_pattern_cols + 2 * (max_pattern_cols - 1)))
        pattern_check[max_pattern_rows - 1:max_pattern_rows - 1 + pattern_i.shape[0], 
                        max_pattern_cols - 1:max_pattern_cols - 1 + pattern_i.shape[1]] = pattern_i
        
        collision_candidates = np.where(
            (np.abs(i_row - score_table[:,0]) <= max_pattern_rows - 1) & 
            (np.abs(i_col - score_table[:,1]) <= max_pattern_cols - 1)
      
        )[0]
        for j in collision_candidates[collision_candidates >= i]:
            j_row = score_table[j, 0] - i_row + max_pattern_rows - 1
            j_col = score_table[j, 1] - i_col + max_pattern_cols - 1

            assert j_row >= 0 and j_col >= 0

            token_j_id = list(tokens.keys())[score_table[j, 2]]
            pattern_j = tokens[token_j_id]['pattern']

            pattern_check_extract = pattern_check[j_row:j_row + pattern_j.shape[0], j_col:j_col + pattern_j.shape[1]]

            if (pattern_check_extract[pattern_j > 0] > 0).any():
                collision_matrix[i, j] = True
                collision_matrix[j, i] = True

    return collision_matrix

def is_valid(pop: Tuple[np.array, int], collision_matrix: np.ndarray):
    valid = True
    active_genes = np.where(pop[0])[0]
    for i in range(len(active_genes)):
        idx0 = active_genes[i]
        for j in range(i+1, len(active_genes)):
            idx1 = active_genes[j]
            if collision_matrix[idx0, idx1]:
                valid = False
                break
        if not valid:
            break

    return valid

def fitness(pop: Tuple[np.array, int], score_table: np.ndarray):
    score = 0
    active_genes = np.where(pop[0])[0]
    for i in range(len(active_genes)):
        idx0 = active_genes[i]
        score += score_table[idx0, 3]

    return score

def generate_initial_population(score_table: np.ndarray, collision_matrix: np.ndarray, population_size=1000):
    population = []
    while len(population) < population_size:
        print(len(population))
        pop = (np.random.choice(a=[True, False], size=score_table.shape[0], p=[0.001, 0.999]), 0)
        if is_valid(pop, collision_matrix):
            population.append(pop)

    return population

def selection_by_tournament(population: list, tournament_size: int = 3):
    indices = np.random.permutation(len(population))

    parents = []
    for i in range(0, len(indices), tournament_size):
        pool = population[i:i+tournament_size]
        pool = sorted(pool, key=lambda x: -x[1])
        parents.append(pool[0])
    
    return parents

def two_point_crossover(parents, score_table, collision_matrix):
    rng = default_rng()
    children = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i+1]

        crossover_done = False
        while not crossover_done:
            p1, p2 = rng.choice(a=list(range(len(parent1[0]))), size=(2,), replace=False)
            child1 = (parent1[0], 0)
            child2 = (parent2[0], 0)
            child1[0][p1:p2] = parent2[0][p1:p2]
            child2[0][p1:p2] = parent1[0][p1:p2]
            if is_valid(child1, collision_matrix) and is_valid(child2, collision_matrix):
                crossover_done = True
                child1 = (child1[0], fitness(child1, score_table))
                child2 = (child2[0], fitness(child2, score_table))
                children.extend([child1, child2])

    return children

def flip_mutation(population: list, n_genes: int = 1):
    pass




def ga_placement(board: np.ndarray, tokens: OrderedDict):
    score_table = get_score_table(board, tokens)
    collision_matrix = get_collision_matrix(board, score_table, tokens)
    population = generate_initial_population(score_table, collision_matrix, 1000)

    parents = selection_by_tournament(population, 3)
    children = two_point_crossover(parents, score_table, collision_matrix)
    a = 1

