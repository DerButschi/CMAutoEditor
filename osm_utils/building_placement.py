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

import pyomo.environ as pyo
from itertools import product
import numpy as np
from collections import OrderedDict
from queue import Queue, LifoQueue
import copy

def create_model(board, tokens):
    model = pyo.ConcreteModel()

    # Set of indices for the board and tokens
    model.I = pyo.RangeSet(0, len(board)-1)
    model.J = pyo.RangeSet(0, len(board[1])-1)
    model.K = pyo.RangeSet(0, len(tokens)-1)

    # Binary variable indicating whether token k is placed at position (i, j) on the board
    model.x = pyo.Var(model.I, model.J, model.K, within=pyo.Binary, initialize=0)

    # Binary variable indicating whether tile (i, j) belongs to area n
    model.y = pyo.Var(model.I, model.J, within=pyo.Binary, initialize=0)

    # Objective function to maximize the score
    model.score = pyo.Objective(expr=sum(2 * board[i][j] * model.y[i, j] for i, j in product(model.I, model.J) if board[i][j] > 0) -
                                2 * sum(board[i][j] * model.y[i, j] for i, j in product(model.I, model.J) if board[i][j] != board[i][j] * pyo.value(model.y[i, j])) -
                                sum(model.x[i, j, k] for i, j, k in model.x), sense=pyo.maximize)


    # Constraint: Each number must be covered exactly once or as desired
    model.number_covering = pyo.ConstraintList()
    for num in set(board[i][j] for i, j in product(model.I, model.J) if board[i][j] > 0):
        count = sum(model.x[i, j, k] for i, j, k in model.x if board[i][j] == num)
        model.number_covering.add(count == 1)

    # Constraint: No token placement on tiles marked with -1
    model.no_token_on_minus_one = pyo.ConstraintList()
    for i, j, k in model.x:
        if board[i][j] == -1:
            model.no_token_on_minus_one.add(model.x[i, j, k] == 0)

    # Constraint: Token placement rules
    model.token_placement_rules = pyo.ConstraintList()
    for i, j in product(model.I, model.J):
        model.token_placement_rules.add(sum(model.x[i, j, k] for k in model.K) <= 1)  # Maximum one token at each position

    # Constraint: Each tile belongs to only one area
    model.tile_belongs_to_one_area = pyo.ConstraintList()
    for i, j in product(model.I, model.J):
        model.tile_belongs_to_one_area.add(sum(model.y[i, j] for i, j in product(model.I, model.J)) == 1)

    return model

def branch_and_bound(board, tokens):
    best_score = float('-inf')
    best_solution = None

    search_tree = [create_model(board, tokens)]  # Initialize the search tree with the root node

    while search_tree:
        node = search_tree.pop()  # Select a node from the search tree
        lower_bound = node.score.expr()

        if lower_bound > best_score:
            # Branch the node and generate child nodes
            child_nodes = []
            for i, j, k in node.x:
                if node.x[i, j, k].value == 0:
                    child = node.clone()
                    child.x[i, j, k].fix(1)
                    child_nodes.append(child)

            # Solve the Pyomo model for each child node
            for child_node in child_nodes:
                solver = pyo.SolverFactory('glpk')
                solver_result = solver.solve(child_node)

                if str(solver_result.solver.status) == 'ok' and solver_result.solver.termination_condition == 'optimal':
                    score = pyo.value(child_node.score)

                    if score > best_score:
                        best_score = score
                        best_solution = [[0] * len(board[1]) for _ in range(len(board))]
                        for i, j, k in child_node.x:
                            if child_node.x[i, j, k].value == 1:
                                best_solution[i-1][j-1] = tokens[k-1]

                    if score > best_score:
                        search_tree.append(child_node)  # Add the child node to the search tree

    return best_solution, best_score


class Node:
    def __init__(self, state, score, tokens_placed: OrderedDict):
        self.state = state
        self.tokens_placed = tokens_placed
        self.score = score

    def clone(self):
        cloned_node = Node(
            state=copy.deepcopy(self.state),
            tokens_placed=copy.deepcopy(self.tokens_placed),
            score=self.score
        )

        return cloned_node


def get_score(board: np.ndarray, node: Node):
    score = 0
    score -= np.count_nonzero((board > 0) & (node.state == 0)) * 2
    score1 = 0
    score1 -= len(node.tokens_placed.keys())
    score1 -= np.count_nonzero((board == 0) & (node.state > 0)) * 2
    
    # score += np.count_nonzero((board > 0) & (node.state > 0))
    
    for token_id in np.unique(node.state[node.state > 0]):
        building_ids, id_counts = np.unique(board[(node.state == token_id) & (board > 0)], return_counts=True)
        id_counts = sorted(id_counts, key=lambda x: -x)
        if len(id_counts) > 1:
            score1 -= np.sum(id_counts[1:]) * 2

    # score2 = 0        

    # for token_id in node.tokens_placed.keys():
    #     score2 -= 1
    #     pattern = node.tokens_placed[token_id]['pattern']
    #     coord = node.tokens_placed[token_id]['coord']
    #     building_id = node.tokens_placed[token_id]['building_id']

    #     # state_extract = node.state[coord[0]:coord[0] + pattern.shape[0], coord[1]:coord[1] + pattern.shape[1]]

    #     board_extract = board[coord[0]:coord[0] + pattern.shape[0], coord[1]:coord[1] + pattern.shape[1]]

    #     token_index = np.where(pattern == 1)
    #     for idx0, idx1 in [(token_index[0][i], token_index[1][i]) for i in range(len(token_index[0]))]:
    #         if board_extract[idx0, idx1] == 0:
    #             score2 -= 2
    #         elif board_extract[idx0, idx1] != building_id:
    #             score2 -= 2
    #         elif board_extract[idx0, idx1] == building_id:
    #             score2 += 2

    # assert score1 == score2
    print(score + score1)
    return score + score1
    # return score


def custom_branch_and_bound(board: np.ndarray, tokens: OrderedDict):
    pattern_shapes = [tokens[key]['pattern'].shape for key in tokens]
    max_pattern_shape = (
        max([ps[0] for ps in pattern_shapes]),
        max([ps[1] for ps in pattern_shapes])
    )
    min_n_pattern_squares = (
        min([ps[0] * ps[1] for ps in pattern_shapes])
    )

    start_node = Node(
        state=np.zeros_like(board, dtype=int),
        score=-np.inf,
        tokens_placed=OrderedDict()
    )
    start_node.score = get_score(board, start_node)

    best_score = start_node.score
    best_solution = None

    queue = LifoQueue()
    queue.put(start_node)

    while not queue.empty():
        node = queue.get()
        # lower_bound = get_score(board, node)
        lower_bound = node.score

        if lower_bound >= best_score:
            not_covered_index = np.where(node.state == 0)
            for idx0, idx1 in [(not_covered_index[0][i], not_covered_index[1][i]) for i in range(len(not_covered_index[0]))]:
                max_board_extract = board[idx0:idx0 + max_pattern_shape[0], idx1:idx1 + max_pattern_shape[1]]
                if len(max_board_extract[max_board_extract > 0]) == 0:
                    continue
                for token_id, token_dict in tokens.items():
                    pattern = token_dict['pattern']
                    if idx0 + pattern.shape[0] - 1 > board.shape[0] - 1:
                        continue
                    if idx1 + pattern.shape[1] - 1 > board.shape[1] - 1:
                        continue

                    board_extract = board[idx0:idx0 + pattern.shape[0], idx1:idx1 + pattern.shape[1]]
                    state_extract = node.state[idx0:idx0 + pattern.shape[0], idx1:idx1 + pattern.shape[1]]

                    if (board_extract[pattern == 1] == -1).any():
                        continue
                    if (state_extract[pattern == 1] > 0).any():
                        continue
                    if (board_extract[pattern == 1] == 0).all():
                        continue

                    child_node = node.clone()
                    if len(child_node.tokens_placed.keys()) == 0:
                        token_placement_idx = 0
                    else:
                        token_placement_idx = max(child_node.tokens_placed.keys()) + 1

                    child_node.state[idx0:idx0 + pattern.shape[0], idx1:idx1 + pattern.shape[1]] = ((child_node.state[idx0:idx0 + pattern.shape[0], idx1:idx1 + pattern.shape[1]]) | pattern) * (token_placement_idx + 1)

                    building_ids = np.unique(board_extract[(pattern == 1) & (board_extract != 0)], return_counts=True)
                    building_id = building_ids[0][0]


                    token_dict_entry = {
                        'building_id': building_id,
                        'pattern': pattern,
                        'coord': [idx0, idx1]
                    }
                    
                    
                    child_node.tokens_placed[token_placement_idx] = token_dict_entry

                    child_node_score = get_score(board, child_node)
                    if child_node_score > best_score:
                        best_score = child_node_score
                        best_solution = child_node
                        child_node.score = child_node_score
                        queue.put(child_node)


                    # child_node.

                    a = 1

    return best_solution, best_score




        


board = np.array([
    [ 0,  0,  0,  1,  1,  0,  0],
    [-1,  0,  0,  1,  1,  2,  2],
    [-1,  0,  0,  1,  1,  2,  2],
    [-1, -1, -1, -1, -1, -1, -1],
    [-1,  0,  3,  0, -1,  0,  4],
    [-1,  3,  3,  3,  0, -1,  4],
    [-1,  0,  3,  0,  5,  5, -1],
    [-1,  0,  0,  0,  5,  5,  0],
], dtype=int)

tokens = OrderedDict()
tokens[0] = np.array([
    [1, 1],
    [1, 1]
])
tokens[1] = np.array([
    [1, 1],
    [1, 1],
    [1, 1]
])
tokens[2] = np.array([
    [1, 1, 1],
    [1, 1, 1],
])
tokens[3] = np.array([
    [1, 1],
])
tokens[4] = np.array([
    [1],
    [1]
])
tokens[5] = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])


# board = [
#     [ 0,  0,  1,  1,  0,  0,  0,  2,  2,  0],
#     [ 0,  0,  0,  1,  0, -1,  0,  0,  2,  2],
#     [ 0,  3,  3,  1,  0, -1,  0,  0,  2,  0],
#     [ 0,  3,  0,  0,  0, -1,  4,  0,  2,  0],
#     [ 0,  0,  0,  0,  5,  5,  4,  0,  0,  0],
#     [ 0,  0,  6,  0,  5, -1,  4,  0,  0,  0],
#     [ 0,  6,  6,  0,  0, -1,  4,  0,  0,  0],
#     [ 0,  0,  0,  7,  7, -1,  0,  0,  8,  8],
#     [ 0,  9,  0,  7,  0, -1,  0,  0,  8,  0],
#     [ 0,  9,  0,  0,  0, -1,  0,  0,  0,  0]
# ]

# tokens = [
#     [[1]],                       # Token 1: 1x1 size (vertical or horizontal placement)
#     [[2, 2]],                    # Token 2: 1x2 size (vertical or horizontal placement)
#     [[3],                        # Token 3: 3x1 size (vertical or horizontal placement)
#      [3],
#      [3]],
#     [[4, 4],                     # Token 4: 2x2 size (diagonal placement)
#      [4, 4]],
#     [[5, 5, 5],                  # Token 5: 3x3 size (diagonal placement)
#      [5, 5, 5],
#      [5, 5, 5]]
# ]

# best_solution, best_score = branch_and_bound(board, tokens)
# best_solution, best_score = custom_branch_and_bound(board, tokens)
a = 1