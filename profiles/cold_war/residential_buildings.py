# Copyright (C) 2022  Nicolas Möser

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

import pandas

# direction, row, column, is diagonal?, lower right point, upper right point, stories
# lower left is always (0, 0)
buildings = [  
    ["Independent Buildings",   "House",    0, 0, 0,    False, 1,     0,      1,      1,      2,    1.0,    2,  2],
    ["Independent Buildings",   "House",    0, 0, 1,    False, 1.5,   0,      1.5,    1,      2,    1.0,    3,    2],
    ["Independent Buildings",   "House",    0, 0, 2,    False, 1.5,   0,      1.5,    1,      2,    1.0,    3,    2],
    ["Independent Buildings",   "House",    0, 0, 3,    False, 1,     0,      1,      1,      1,    0.25,   2,    2],
    ["Independent Buildings",   "House",    0, 1, 0,    False, 1.5,   0,      1.5,    1,      1,    0.25,   3,    2],
    ["Independent Buildings",   "House",    0, 1, 1,    False, 1.5,   0,      1.5,    1,      3,    0.5,    3,    2],
    ["Independent Buildings",   "House",    0, 1, 2,    True,  1,     -1,     2,      0,      2,    1.0,    2,    2],
    ["Independent Buildings",   "House",    0, 1, 3,    False, 1.5,   0,      1.5,    1,      3,    0.5,    3,    2],
    ["Independent Buildings",   "House",    0, 2, 0,    False, 1.5,   0,      1.5,    1,      2,    1.0,    3,  2],
    ["Independent Buildings",   "House",    0, 2, 1,    True,  1,     -1,     2.5,    0.5,    2,    1.0,    2,  3],
    ["Independent Buildings",   "House",    0, 2, 2,    True,  1,     -1,     2.5,    0.5,    2,    1.0,    2,  3],
    ["Independent Buildings",   "House",    0, 2, 3,    True,  1,     -1,     2,      0,      1,    0.25,   2,  2],
    ["Independent Buildings",   "House",    0, 3, 0,    True,  1,     -1,     2.5,    0.5,    1,    0.25,    2,  3],
    ["Independent Buildings",   "House",    0, 3, 1,    True,  1,     -1,     2.5,    0.5,    3,    0.5,    2,  3],
    ["Independent Buildings",   "House",    0, 3, 2,    True,  1,     -1,     2.5,    0.5,    3,    0.5,    2,  3],
    ["Independent Buildings",   "House",    0, 3, 3,    True,  1,     -1,     2.5,    0.5,    2,    1.0,    2,  3],

    ["Independent Buildings",   "House",    1, 0, 0,    False, 1,     0,      1,      1,        2,    1.0,  2,  2],
    ["Independent Buildings",   "House",    1, 0, 1,    False, 1,     0,      1,      1.5,      2,    1.0,  2,  3],
    ["Independent Buildings",   "House",    1, 0, 2,    False, 1,     0,      1,      1.5,      2,    1.0,  2,  3],
    ["Independent Buildings",   "House",    1, 0, 3,    False, 1,     0,      1,      1,        1,    0.25, 2,  2],
    ["Independent Buildings",   "House",    1, 1, 0,    False, 1,     0,      1,      1.5,      1,    0.25, 2,  3],
    ["Independent Buildings",   "House",    1, 1, 1,    False, 1,     0,      1,      1.5,      3,    0.5,  2,  3],
    ["Independent Buildings",   "House",    1, 1, 2,    True,  1,     -1,     2,      0,        2,    1.0,  2,  2],
    ["Independent Buildings",   "House",    1, 1, 3,    False, 1,     0,      1,      1.5,      3,    0.5,  2,  3],
    ["Independent Buildings",   "House",    1, 2, 0,    False, 1,     0,      1,      1.5,      2,    1.0,  2,  3],
    ["Independent Buildings",   "House",    1, 2, 1,    True,  1.5,   -1.5,   2.5,    -0.5,     2,    1.0,  3,  2],
    ["Independent Buildings",   "House",    1, 2, 2,    True,  1.5,   -1.5,   2.5,    -0.5,     2,    1.0,  3,  2],
    ["Independent Buildings",   "House",    1, 2, 3,    True,  1,     -1,     2,      0,        1,    0.25, 2,  2],
    ["Independent Buildings",   "House",    1, 3, 0,    True,  1.5,   -1.5,   2.5,    -0.5,     1,    0.25, 3,  2],
    ["Independent Buildings",   "House",    1, 3, 1,    True,  1.5,   -1.5,   2.5,    -0.5,     3,    0.5,  3,  2],
    ["Independent Buildings",   "House",    1, 3, 2,    True,  1.5,   -1.5,   2.5,    -0.5,     3,    0.5,  3,  2],
    ["Independent Buildings",   "House",    1, 3, 3,    True,  1.5,   -1.5,   2.5,    -0.5,     2,    1.0,  3,  2],

    ["Independent Buildings",   "Barn",    0, 0, 2,    False,  1,     0,      1,      0.5,      1,    1.0,  2,  1],
    ["Independent Buildings",   "Barn",    0, 1, 0,    False,  1,     0,      1,      0.5,      1,    1.0,  2,  1],

    ["Independent Buildings",   "Barn",    1, 0, 2,    False,  0.5,   0,      0.5,    1,        1,    1.0,  1,  2],
    ["Independent Buildings",   "Barn",    1, 1, 0,    False,  0.5,   0,      0.5,    1,        1,    1.0,  1,  2],


    ["Modular Buildings",       "1 Story",    0, 0, 0,    False,  1,    0,      1,      1,      1,    0.25, 2,  2],
    ["Modular Buildings",       "1 Story",    0, 0, 1,    False,  1.5,  0,      1.5,    1,      1,    0.25, 3,  2],
    ["Modular Buildings",       "1 Story",    0, 0, 2,    False,  1,    0,      1,      1.5,    1,    0.25, 2,  3],
    ["Modular Buildings",       "1 Story",    0, 0, 3,    False,  2,    0,      2,      1,      1,    0.25, 4,  2],
    ["Modular Buildings",       "1 Story",    0, 1, 0,    False,  1,    0,      1,      2,      1,    0.25, 2,  4],
    ["Modular Buildings",       "1 Story",    0, 1, 1,    False,  1.5,  0,      1.5,    1.5,    1,    0.25, 3,  3],
    ["Modular Buildings",       "1 Story",    0, 1, 2,    False,  2,    0,      2,      1.5,    1,    0.25, 4,  3],
    ["Modular Buildings",       "1 Story",    0, 1, 3,    False,  1.5,  0,      1.5,    2,      1,    0.25, 3,  4],
    ["Modular Buildings",       "1 Story",    0, 2, 0,    False,  2,    0,      2,      2,      1,    0.25, 4,  4],
    ["Modular Buildings",       "1 Story",    0, 2, 1,    True,  1,    -1,     2,      0,      1,    0.25,  2,  2],
    ["Modular Buildings",       "1 Story",    0, 2, 2,    True,  1,    -1,     2.5,    0.5,    1,    0.25,  2,  3],
    ["Modular Buildings",       "1 Story",    0, 2, 3,    True,  1.5,  -1.5,   2.5,    -0.5,   1,    0.25,  3,  2],
    ["Modular Buildings",       "1 Story",    0, 3, 0,    True,  1.5,  -1.5,   3,      0,      1,    0.25,  3,  3],

    ["Modular Buildings",       "1 Story",    1, 0, 0,    False,  1,    0,      1,      1,      1,    0.25, 2,  2],
    ["Modular Buildings",       "1 Story",    1, 0, 1,    False,  1,    0,      1,      1.5,    1,    0.25, 2,  3],
    ["Modular Buildings",       "1 Story",    1, 0, 2,    False,  1.5,  0,      1.5,    1,      1,    0.25, 3,  2],
    ["Modular Buildings",       "1 Story",    1, 0, 3,    False,  1,    0,      1,      1.5,    1,    0.25, 2,  4],
    ["Modular Buildings",       "1 Story",    1, 1, 0,    False,  2,    0,      2,      1,      1,    0.25, 4,  2],
    ["Modular Buildings",       "1 Story",    1, 1, 1,    False,  1.5,  0,      1.5,    1.5,    1,    0.25, 3,  3],
    ["Modular Buildings",       "1 Story",    1, 1, 2,    False,  1.5,  0,      1.5,    2,      1,    0.25, 3,  4],
    ["Modular Buildings",       "1 Story",    1, 1, 3,    False,  2,    0,      2,      1.5,    1,    0.25, 4,  3],
    ["Modular Buildings",       "1 Story",    1, 2, 0,    False,  2,    0,      2,      2,      1,    0.25, 4,  4],
    ["Modular Buildings",       "1 Story",    1, 2, 1,    True,  1,    -1,      2,      0,      1,    0.25, 2,  2],
    ["Modular Buildings",       "1 Story",    1, 2, 2,    True,  1.5,  -1.5,    2.5,    -0.5,   1,    0.25, 3,  2],
    ["Modular Buildings",       "1 Story",    1, 2, 3,    True,  1,    -1,      2.5,    0.5,    1,    0.25, 2,  3],
    ["Modular Buildings",       "1 Story",    1, 3, 0,    True,  1.5,  -1.5,    3,      0,      1,    0.25, 3,  3],

    ["Modular Buildings",       "2 Story",    0, 0, 0,    False,  1,    0,      1,      1,      2,    0.25, 2,  2],
    ["Modular Buildings",       "2 Story",    0, 0, 1,    False,  1.5,  0,      1.5,    1,      2,    0.25, 3,  2],
    ["Modular Buildings",       "2 Story",    0, 0, 2,    False,  1,    0,      1,      1.5,    2,    0.25, 2,  3],
    ["Modular Buildings",       "2 Story",    0, 0, 3,    False,  2,    0,      2,      1,      2,    0.25, 4,  2],
    ["Modular Buildings",       "2 Story",    0, 1, 0,    False,  1,    0,      1,      2,      2,    0.25, 2,  4],
    ["Modular Buildings",       "2 Story",    0, 1, 1,    False,  1.5,  0,      1.5,    1.5,    2,    0.25, 3,  3],
    ["Modular Buildings",       "2 Story",    0, 1, 2,    False,  2,    0,      2,      1.5,    2,    0.25, 4,  3],
    ["Modular Buildings",       "2 Story",    0, 1, 3,    False,  1.5,  0,      1.5,    2,      2,    0.25, 3,  4],
    ["Modular Buildings",       "2 Story",    0, 2, 0,    False,  2,    0,      2,      2,      2,    0.25, 4,  4],
    ["Modular Buildings",       "2 Story",    0, 2, 1,    True,  1,    -1,     2,      0,      2,    0.25,  2,  2],
    ["Modular Buildings",       "2 Story",    0, 2, 2,    True,  1,    -1,     2.5,    0.5,    2,    0.25,  2,  3],
    ["Modular Buildings",       "2 Story",    0, 2, 3,    True,  1.5,  -1.5,   2.5,    -0.5,   2,    0.25,  3,  2],
    ["Modular Buildings",       "2 Story",    0, 3, 0,    True,  1.5,  -1.5,   3,      0,      2,    0.25,  3,  3],

    ["Modular Buildings",       "2 Story",    1, 0, 0,    False,  1,    0,      1,      1,      2,    0.25, 2,  2],
    ["Modular Buildings",       "2 Story",    1, 0, 1,    False,  1,    0,      1,      1.5,    2,    0.25, 2,  3],
    ["Modular Buildings",       "2 Story",    1, 0, 2,    False,  1.5,  0,      1.5,    1,      2,    0.25, 3,  2],
    ["Modular Buildings",       "2 Story",    1, 0, 3,    False,  1,    0,      1,      1.5,    2,    0.25, 2,  4],
    ["Modular Buildings",       "2 Story",    1, 1, 0,    False,  2,    0,      2,      1,      2,    0.25, 4,  2],
    ["Modular Buildings",       "2 Story",    1, 1, 1,    False,  1.5,  0,      1.5,    1.5,    2,    0.25, 3,  3],
    ["Modular Buildings",       "2 Story",    1, 1, 2,    False,  1.5,  0,      1.5,    2,      2,    0.25, 3,  4],
    ["Modular Buildings",       "2 Story",    1, 1, 3,    False,  2,    0,      2,      1.5,    2,    0.25, 4,  3],
    ["Modular Buildings",       "2 Story",    1, 2, 0,    False,  2,    0,      2,      2,      2,    0.25, 4,  4],
    ["Modular Buildings",       "2 Story",    1, 2, 1,    True,  1,    -1,      2,      0,      2,    0.25, 2,  2],
    ["Modular Buildings",       "2 Story",    1, 2, 2,    True,  1.5,  -1.5,    2.5,    -0.5,   2,    0.25, 3,  2],
    ["Modular Buildings",       "2 Story",    1, 2, 3,    True,  1,    -1,      2.5,    0.5,    2,    0.25, 2,  3],
    ["Modular Buildings",       "2 Story",    1, 3, 0,    True,  1.5,  -1.5,    3,      0,      2,    0.25, 3,  3],

    ["Modular Buildings",       "3 Story",    0, 0, 0,    False,  1,    0,      1,      1,      3,    0.25, 2,  2],
    ["Modular Buildings",       "3 Story",    0, 0, 1,    False,  1.5,  0,      1.5,    1,      3,    0.25, 3,  2],
    ["Modular Buildings",       "3 Story",    0, 0, 2,    False,  1,    0,      1,      1.5,    3,    0.25, 2,  3],
    ["Modular Buildings",       "3 Story",    0, 0, 3,    False,  2,    0,      2,      1,      3,    0.25, 4,  2],
    ["Modular Buildings",       "3 Story",    0, 1, 0,    False,  1,    0,      1,      2,      3,    0.25, 2,  4],
    ["Modular Buildings",       "3 Story",    0, 1, 1,    False,  1.5,  0,      1.5,    1.5,    3,    0.25, 3,  3],
    ["Modular Buildings",       "3 Story",    0, 1, 2,    False,  2,    0,      2,      1.5,    3,    0.25, 4,  3],
    ["Modular Buildings",       "3 Story",    0, 1, 3,    False,  1.5,  0,      1.5,    2,      3,    0.25, 3,  4],
    ["Modular Buildings",       "3 Story",    0, 2, 0,    False,  2,    0,      2,      2,      3,    0.25, 4,  4],
    ["Modular Buildings",       "3 Story",    0, 2, 1,    True,  1,    -1,     2,      0,      3,    0.25,  2,  2],
    ["Modular Buildings",       "3 Story",    0, 2, 2,    True,  1,    -1,     2.5,    0.5,    3,    0.25,  2,  3],
    ["Modular Buildings",       "3 Story",    0, 2, 3,    True,  1.5,  -1.5,   2.5,    -0.5,   3,    0.25,  3,  2],
    ["Modular Buildings",       "3 Story",    0, 3, 0,    True,  1.5,  -1.5,   3,      0,      3,    0.25,  3,  3],

    ["Modular Buildings",       "3 Story",    1, 0, 0,    False,  1,    0,      1,      1,      3,    0.25, 2,  2],
    ["Modular Buildings",       "3 Story",    1, 0, 1,    False,  1,    0,      1,      1.5,    3,    0.25, 2,  3],
    ["Modular Buildings",       "3 Story",    1, 0, 2,    False,  1.5,  0,      1.5,    1,      3,    0.25, 3,  2],
    ["Modular Buildings",       "3 Story",    1, 0, 3,    False,  1,    0,      1,      1.5,    3,    0.25, 2,  4],
    ["Modular Buildings",       "3 Story",    1, 1, 0,    False,  2,    0,      2,      1,      3,    0.25, 4,  2],
    ["Modular Buildings",       "3 Story",    1, 1, 1,    False,  1.5,  0,      1.5,    1.5,    3,    0.25, 3,  3],
    ["Modular Buildings",       "3 Story",    1, 1, 2,    False,  1.5,  0,      1.5,    2,      3,    0.25, 3,  4],
    ["Modular Buildings",       "3 Story",    1, 1, 3,    False,  2,    0,      2,      1.5,    3,    0.25, 4,  3],
    ["Modular Buildings",       "3 Story",    1, 2, 0,    False,  2,    0,      2,      2,      3,    0.25, 4,  4],
    ["Modular Buildings",       "3 Story",    1, 2, 1,    True,  1,    -1,      2,      0,      3,    0.25, 2,  2],
    ["Modular Buildings",       "3 Story",    1, 2, 2,    True,  1.5,  -1.5,    2.5,    -0.5,   3,    0.25, 3,  2],
    ["Modular Buildings",       "3 Story",    1, 2, 3,    True,  1,    -1,      2.5,    0.5,    3,    0.25, 2,  3],
    ["Modular Buildings",       "3 Story",    1, 3, 0,    True,  1.5,  -1.5,    3,      0,      3,    0.25, 3,  3],

]

def build_building_df():
    columns = ['menu', 'cat1', 'direction', 'row', 'col', 'is_diagonal', 'x1', 'y1', 'x2', 'y2', 'stories', 'weight', 'width', 'height']
    building_dict = {col: [] for col in columns}

    for entry in buildings:
        for idx, el in enumerate(entry):
            building_dict[columns[idx]].append(el)

    building_df = pandas.DataFrame(building_dict)

    return building_df

