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

from typing import List, Tuple
import numpy as np
from numpy.random import default_rng
from collections import OrderedDict
from itertools import product
import logging
from tqdm import tqdm

class Pop:
    def __init__(self, p_active, score_table, compatibility_tensor):
        self.compatibility_tensor = compatibility_tensor
        self.score_table = score_table
        self.valid_genes = np.zeros((score_table.shape[0],), dtype=bool)
        self.randomly_initialize_genes(p_active)
        self.update_fitness()
        

    def randomly_initialize_genes(self, p_active):
        genes = np.random.choice(a=[True, False], size=self.score_table.shape[0], p=[p_active, 1-p_active])
        for idx in np.argwhere(genes):
            self.activate_gene(idx)

    def can_activate_gene(self, idx):
        return self.valid_genes[idx]
    
    def activate_gene(self, idx):
        if self.can_activate_gene(idx):
            self.gene[idx] = True

    def update_valid_genes(self, new_gene_idx):
        valid_gene_indices = np.argwhere(self.valid_genes)
        for idx in valid_gene_indices:
            if idx == new_gene_idx:
                continue
            if abs(self.score_table[new_gene_idx,0] - self.score_table[idx, 0]) <= (self.compatibility_tensor.shape[2] - 1) / 2 and \
                abs(self.score_table[new_gene_idx,1] - self.score_table[idx, 1]) <= (self.compatibility_tensor.shape[3] - 1) / 2:
                if self.compatibility_tensor[
                    self.score_table[new_gene_idx,2],
                    self.score_table[idx,2],
                    self.score_table[idx,0] - self.score_table[new_gene_idx,0] + int((self.compatibility_tensor.shape[2] - 1) / 2),
                    self.score_table[idx,1] - self.score_table[new_gene_idx,1] + int((self.compatibility_tensor.shape[3] - 1) / 2),
                ] == 0:
                    self.valid_genes[idx] = False

    def activate_genes(self, indices):
        for idx in indices:
            self.activate_gene(idx)

    def get_active_genes(self):
        return np.argwhere(self.genes)

    def update_fitness(self):
        active_genes = self.get_active_genes

        score = np.sum(self.score_table[active_genes, 3])
        building_ids, counts = np.unique(self.score_table[active_genes, 4], return_counts=True)
        multiple_assignments = np.where(counts > 1)[0]

        if len(multiple_assignments) > 0:
            for ma_idx in multiple_assignments:
                building_id = building_ids[ma_idx]
                max_tiles = -np.sort(-self.score_table[active_genes, 5][self.score_table[active_genes, 4] == building_id])[0]
                score -= (max_tiles - 1) * 2

        self.fitness = score


        

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
    relevant_rows_cols = []
    for j_row, j_col in np.argwhere(board > 0):
        for i_row in range(max(0, j_row - max_pattern_rows - 1), j_row + 1):
            for i_col in range(max(0, j_col - max_pattern_cols - 1), j_col + 1):
                relevant_rows_cols.append([i_row, i_col])    

    relevant_rows_cols = np.unique(np.array(relevant_rows_cols), axis=0)
    for i_row, i_col in tqdm(relevant_rows_cols, 'generating score table'):
        for i_token in range(len(tokens.keys())):
            token_id = list(tokens.keys())[i_token]
            pattern = tokens[token_id]['pattern']
            has_modular = tokens[token_id]['has_modular']
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

                ids, id_counts = np.unique(board_extract[(pattern > 0) & (board_extract > 0)], return_counts=True)
                id_counts = sorted(id_counts, key=lambda x: -x)
                max_id = -1
                max_counts = 0
                if len(id_counts) > 0:
                    score += id_counts[0] * 2
                    max_id = ids[np.argmax(id_counts)]
                    max_counts = id_counts[0]
                if len(id_counts) > 1:
                    score -= np.sum(id_counts[1:]) * 2

                if score > 0:
                    score_table.append([i_row, i_col, i_token, score, max_id, max_counts, has_modular])

    return np.array(score_table)

def get_collision_matrix(board: np.ndarray, score_table: np.ndarray, tokens: OrderedDict):
    collision_matrix = np.zeros((score_table.shape[0], score_table.shape[0]), dtype=bool)

    pattern_shapes = [tokens[key]['pattern'].shape for key in tokens]
    max_pattern_rows = max([ps[0] for ps in pattern_shapes])
    max_pattern_cols = max([ps[1] for ps in pattern_shapes])

    for i in tqdm(range(score_table.shape[0]), 'calculating collision matrix'):
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
        for j in collision_candidates[collision_candidates > i]:
            j_row = score_table[j, 0] - i_row + max_pattern_rows - 1
            j_col = score_table[j, 1] - i_col + max_pattern_cols - 1

            assert j_row >= 0 and j_col >= 0

            token_j_id = list(tokens.keys())[score_table[j, 2]]
            pattern_j = tokens[token_j_id]['pattern']

            pattern_check_extract = pattern_check[j_row:j_row + pattern_j.shape[0], j_col:j_col + pattern_j.shape[1]]

            if (pattern_check_extract[pattern_j > 0.5] > 0.5).any():
                collision_matrix[i, j] = True
                collision_matrix[j, i] = True

    return collision_matrix

def get_compatibility_tensor(tokens):
    pattern_shapes = [tokens[key]['pattern'].shape for key in tokens]
    max_pattern_rows = max([ps[0] for ps in pattern_shapes])
    max_pattern_cols = max([ps[1] for ps in pattern_shapes])

    compatibility_tensor = np.zeros((len(tokens.keys()), len(tokens.keys()), 2 * max_pattern_rows - 1, 2 * max_pattern_cols - 1))
    for i_token in range(len(tokens.keys())):
        token_i_id = list(tokens.keys())[i_token]
        pattern_i = tokens[token_i_id]['pattern']
        pattern_check = np.zeros((max_pattern_rows + 2 * (max_pattern_rows - 1), max_pattern_cols + 2 * (max_pattern_cols - 1)))
        pattern_check[max_pattern_rows - 1:max_pattern_rows - 1 + pattern_i.shape[0], 
                        max_pattern_cols - 1:max_pattern_cols - 1 + pattern_i.shape[1]] = pattern_i
        for j_token in range(len(tokens.keys())):
            for i_row in range(2 * max_pattern_rows - 1):
                for i_col in range(2 * max_pattern_cols - 1):
                    token_j_id = list(tokens.keys())[j_token]
                    pattern_j = tokens[token_j_id]['pattern']

                    pattern_check_extract = pattern_check[i_row:i_row + pattern_j.shape[0], i_col:i_col + pattern_j.shape[1]]
                    if (pattern_check_extract[pattern_j > 0] == 0).all():
                        compatibility_tensor[i_token, j_token, i_row, i_col] = 1

    return compatibility_tensor

def is_valid(pop: Pop, score_table: np.ndarray, compatibility_tensor: np.ndarray):
    valid = True
    active_genes = np.where(pop.genes)[0]
    # for i in range(len(active_genes)):
    #     idx0 = active_genes[i]
    #     for j in range(i+1, len(active_genes)):
    #         idx1 = active_genes[j]
    #         if collision_matrix[idx0, idx1]:
    #             valid = False
    #             break
    #     if not valid:
    #         break
    for i in range(len(active_genes)):
        for j in range(i+1,len(active_genes)):
            idx_i = active_genes[i]
            idx_j = active_genes[j]
            if abs(score_table[idx_i,0] - score_table[idx_j, 0]) <= (compatibility_tensor.shape[2] - 1) / 2 and \
                abs(score_table[idx_i,1] - score_table[idx_j, 1]) <= (compatibility_tensor.shape[3] - 1) / 2:
                if compatibility_tensor[
                    score_table[idx_i,2],
                    score_table[idx_j,2],
                    score_table[idx_j,0] - score_table[idx_i,0] + int((compatibility_tensor.shape[2] - 1) / 2),
                    score_table[idx_j,1] - score_table[idx_i,1] + int((compatibility_tensor.shape[3] - 1) / 2),
                ] == 0:
                    return False
                
    # for i in range(len(active_genes)):
    #     if compatibility_tensor[active_genes[i], active_genes].any():
    #         return False

    # check if all building ids for which more than one building is assigned have a modular building token
    building_ids, counts = np.unique(score_table[active_genes, 4], return_counts=True)
    multiple_assignments = np.argwhere(counts > 1)

    if len(multiple_assignments) > 0:
        if not score_table[active_genes, 6][np.isin(score_table[active_genes, 4], building_ids[multiple_assignments])].all():
            return False

    return valid

def fitness(pop: Pop, score_table: np.ndarray):
    active_genes = np.where(pop.genes)[0]
    # for i in range(len(active_genes)):
    #     idx0 = active_genes[i]
    #     score += score_table[idx0, 3]

    score = np.sum(score_table[active_genes, 3])
    building_ids, counts = np.unique(score_table[active_genes, 4], return_counts=True)
    multiple_assignments = np.where(counts > 1)[0]

    if len(multiple_assignments) > 0:
        for ma_idx in multiple_assignments:
            building_id = building_ids[ma_idx]
            max_tiles = -np.sort(-score_table[active_genes, 5][score_table[active_genes, 4] == building_id])[0]
            score -= (max_tiles - 1) * 2


        a = 1


    return score

def generate_initial_population(score_table: np.ndarray, compatibility_tensor: np.ndarray, population_size=1000):
    population = []
    while len(population) < population_size:
        print(len(population))
        genes = np.random.choice(a=[True, False], size=score_table.shape[0], p=[0.001, 0.999])
        pop = Pop(genes=genes)
        if is_valid(pop, score_table, compatibility_tensor):
            population.append(pop)

    return population

def selection_by_tournament(population: List[Pop], tournament_size: int = 3):
    indices = np.random.permutation(len(population))

    parents = []
    for i in range(0, len(indices), tournament_size):
        pool = population[i:i+tournament_size]
        pool = sorted(pool, key=lambda x: -x.fitness)
        parents.append(pool[0])
    
    return parents

def two_point_crossover(parents, score_table, compatibility_tensor):
    rng = default_rng()
    children = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i+1]

        crossover_done = False
        while not crossover_done:
            p1, p2 = rng.choice(a=list(range(len(parent1.genes))), size=(2,), replace=False)
            child1 = Pop(genes=np.copy(parent1.genes))
            child2 = Pop(genes=np.copy(parent2.genes))
            child1.genes[p1:p2] = parent2.genes[p1:p2]
            child2.genes[p1:p2] = parent1.genes[p1:p2]
            if is_valid(child1, score_table, compatibility_tensor) and is_valid(child2, score_table, compatibility_tensor):
                crossover_done = True
                child1.fitness = fitness(child1, score_table)
                child2.fitness = fitness(child2, score_table)
                children.extend([child1, child2])

    return children

def two_point_crossover_valid_only(parents, score_table, compatibility_tensor):
    rng = default_rng()
    children = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i+1]

        p1, p2 = rng.choice(a=list(range(len(parent1.genes))), size=(2,), replace=False)
        child1 = Pop(genes=np.copy(parent1.genes))
        child2 = Pop(genes=np.copy(parent2.genes))
        for pi in range(p1, p2):
            child1.genes[pi] = parent2.genes[pi]
            if not is_valid(child1, score_table, compatibility_tensor):
                child1.genes[pi] = parent1.genes[pi]
            child2.genes[pi] = parent1.genes[pi]
            if not is_valid(child2, score_table, compatibility_tensor):
                child2.genes[pi] = parent2.genes[pi]

        child1.fitness = fitness(child1, score_table)
        child2.fitness = fitness(child2, score_table)
        children.extend([child1, child2])

    return children


def flip_mutation(population: list, score_table: np.ndarray, compatibility_tensor: np.ndarray, n_genes: int = 1):
    for i in range(len(population)):
        pop = population[i]
        mutation_done = False
        for j in range(n_genes):
            cnt = 0
            while (not mutation_done) and cnt < 5:
                cnt += 1
                gidx = np.random.randint(low=0, high=len(pop.genes))
                pop.genes[gidx] = not pop.genes[gidx]
                if is_valid(pop, score_table, compatibility_tensor):
                    mutation_done = True
                    pop.fitness = fitness(pop, score_table)
                else:
                    pop.genes[gidx] = not pop.genes[gidx]

    

def elitist_replacement(population, children):
    new_population = sorted(population, key=lambda x: x.fitness)
    new_population[0:len(children)] = children

    return new_population

def state(board: np.ndarray, pop: Pop, score_table: np.ndarray, tokens: OrderedDict):
    active_genes = np.where(pop.genes)
    state = np.zeros_like(board)
    
    for gidx in active_genes[0]:
        i_row, i_col, i_token = score_table[gidx, 0:3]
        token_id = list(tokens.keys())[i_token]
        pattern = tokens[token_id]['pattern']
        state[i_row:i_row + pattern.shape[0], i_col:i_col + pattern.shape[1]] = gidx

    return len(np.unique(board[(state > 0) & (board > 0)])) / len(np.unique(board[board > 0]))
    

def ga_placement(board: np.ndarray, tokens: OrderedDict):
    logger = logging.getLogger('osm2cm')
    
    score_table = get_score_table(board, tokens)
    compatibility_tensor = get_compatibility_tensor(tokens)
    logger.debug('Generating initial population.')
    population = generate_initial_population(score_table, compatibility_tensor, 1000)

    best_score = -np.inf
    best_solution = None
    terminate = False
    t_cnt = 5
    while not terminate:
        parents = selection_by_tournament(population, 2)
        children = two_point_crossover(parents, score_table, compatibility_tensor)
        flip_mutation(children, score_table, compatibility_tensor, 1)
        population = elitist_replacement(population, children)

        best_pop = sorted(population, key=lambda x: -x.fitness)[0]
        if best_pop.fitness > best_score:
            t_cnt = 20
            best_score = best_pop.fitness
            best_solution = best_pop
        else:
            t_cnt -= 1

        
        best_state = state(board, best_solution, score_table, tokens)

        print(best_score, len(np.where(best_solution.genes)[0]), len(np.unique(np.nonzero(board))), 
              best_state)
        
        if best_state >= 0.98 or t_cnt == 0:
            terminate = True
        
    tokens_placed = []
    for gidx in np.where(best_solution.genes)[0]:
        token_id = list(tokens.keys())[score_table[gidx,2]]
        pattern = tokens[token_id]['pattern']

        tokens_placed.append({
            'token_id': token_id,
            'pattern': pattern,
            'row': score_table[gidx,0],
            'col': score_table[gidx,1]
        })


    #     token_dict_entry = {
    #     'token_id': token_ids[i_token],
    #     'pattern': pattern
    # }

    return tokens_placed, state(board, best_solution, score_table, tokens)


