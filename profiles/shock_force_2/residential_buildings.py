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

# direction, row, column, is diagonal?, lower right point, upper right point, stories
# lower left is always (0, 0)
residential_buildings = [  
    ["Buildings",       "1 Story",    0, 0, 0,    False,  1,    0.25, 2,  2, True],
    ["Buildings",       "1 Story",    0, 0, 1,    False,  1,    0.25, 3,  2, True],
    ["Buildings",       "1 Story",    0, 0, 2,    False,  1,    0.25, 2,  3, True],
    ["Buildings",       "1 Story",    0, 0, 3,    False,  1,    0.25, 4,  2, True],
    ["Buildings",       "1 Story",    0, 1, 0,    False,  1,    0.25, 2,  4, True],
    ["Buildings",       "1 Story",    0, 1, 1,    False,  1,    0.25, 3,  3, True],
    ["Buildings",       "1 Story",    0, 1, 2,    False,  1,    0.25, 4,  3, True],
    ["Buildings",       "1 Story",    0, 1, 3,    False,  1,    0.25, 3,  4, True],
    ["Buildings",       "1 Story",    0, 2, 0,    False,  1,    0.25, 4,  4, True],
    ["Buildings",       "1 Story",    0, 2, 1,    True,  1,    0.25,  2,  2, True],
    ["Buildings",       "1 Story",    0, 2, 2,    True,  1,    0.25,  2,  3, True],
    ["Buildings",       "1 Story",    0, 2, 3,    True,  1,    0.25,  3,  2, True],
    ["Buildings",       "1 Story",    0, 3, 0,    True,  1,    0.25,  3,  3, True],

    ["Buildings",       "1 Story",    1, 0, 0,    False,  1,    0.25, 2,  2, True],
    ["Buildings",       "1 Story",    1, 0, 1,    False,  1,    0.25, 2,  3, True],
    ["Buildings",       "1 Story",    1, 0, 2,    False,  1,    0.25, 3,  2, True],
    ["Buildings",       "1 Story",    1, 0, 3,    False,  1,    0.25, 2,  4, True],
    ["Buildings",       "1 Story",    1, 1, 0,    False,  1,    0.25, 4,  2, True],
    ["Buildings",       "1 Story",    1, 1, 1,    False,  1,    0.25, 3,  3, True],
    ["Buildings",       "1 Story",    1, 1, 2,    False,  1,    0.25, 3,  4, True],
    ["Buildings",       "1 Story",    1, 1, 3,    False,  1,    0.25, 4,  3, True],
    ["Buildings",       "1 Story",    1, 2, 0,    False,  1,    0.25, 4,  4, True],
    ["Buildings",       "1 Story",    1, 2, 1,    True,   1,    0.25, 2,  2, True],
    ["Buildings",       "1 Story",    1, 2, 2,    True,   1,    0.25, 3,  2, True],
    ["Buildings",       "1 Story",    1, 2, 3,    True,   1,    0.25, 2,  3, True],
    ["Buildings",       "1 Story",    1, 3, 0,    True,   1,    0.25, 3,  3, True],

    ["Buildings",       "2 Story",    0, 0, 0,    False,  2,    0.25, 2,  2, True],
    ["Buildings",       "2 Story",    0, 0, 1,    False,  2,    0.25, 3,  2, True],
    ["Buildings",       "2 Story",    0, 0, 2,    False,  2,    0.25, 2,  3, True],
    ["Buildings",       "2 Story",    0, 0, 3,    False,  2,    0.25, 4,  2, True],
    ["Buildings",       "2 Story",    0, 1, 0,    False,  2,    0.25, 2,  4, True],
    ["Buildings",       "2 Story",    0, 1, 1,    False,  2,    0.25, 3,  3, True],
    ["Buildings",       "2 Story",    0, 1, 2,    False,  2,    0.25, 4,  3, True],
    ["Buildings",       "2 Story",    0, 1, 3,    False,  2,    0.25, 3,  4, True],
    ["Buildings",       "2 Story",    0, 2, 0,    False,  2,    0.25, 4,  4, True],
    ["Buildings",       "2 Story",    0, 2, 1,    True,  2,    0.25,  2,  2, True],
    ["Buildings",       "2 Story",    0, 2, 2,    True,  2,    0.25,  2,  3, True],
    ["Buildings",       "2 Story",    0, 2, 3,    True,  2,    0.25,  3,  2, True],
    ["Buildings",       "2 Story",    0, 3, 0,    True,  2,    0.25,  3,  3, True],

    ["Buildings",       "2 Story",    1, 0, 0,    False,  2,    0.25, 2,  2, True],
    ["Buildings",       "2 Story",    1, 0, 1,    False,  2,    0.25, 2,  3, True],
    ["Buildings",       "2 Story",    1, 0, 2,    False,  2,    0.25, 3,  2, True],
    ["Buildings",       "2 Story",    1, 0, 3,    False,  2,    0.25, 2,  4, True],
    ["Buildings",       "2 Story",    1, 1, 0,    False,  2,    0.25, 4,  2, True],
    ["Buildings",       "2 Story",    1, 1, 1,    False,  2,    0.25, 3,  3, True],
    ["Buildings",       "2 Story",    1, 1, 2,    False,  2,    0.25, 3,  4, True],
    ["Buildings",       "2 Story",    1, 1, 3,    False,  2,    0.25, 4,  3, True],
    ["Buildings",       "2 Story",    1, 2, 0,    False,  2,    0.25, 4,  4, True],
    ["Buildings",       "2 Story",    1, 2, 1,    True,   2,    0.25, 2,  2, True],
    ["Buildings",       "2 Story",    1, 2, 2,    True,   2,    0.25, 3,  2, True],
    ["Buildings",       "2 Story",    1, 2, 3,    True,   2,    0.25, 2,  3, True],
    ["Buildings",       "2 Story",    1, 3, 0,    True,   2,    0.25, 3,  3, True],

    ["Buildings",       "3 Story",    0, 0, 0,    False,  3,    0.25, 2,  2, True],
    ["Buildings",       "3 Story",    0, 0, 1,    False,  3,    0.25, 3,  2, True],
    ["Buildings",       "3 Story",    0, 0, 2,    False,  3,    0.25, 2,  3, True],
    ["Buildings",       "3 Story",    0, 0, 3,    False,  3,    0.25, 4,  2, True],
    ["Buildings",       "3 Story",    0, 1, 0,    False,  3,    0.25, 2,  4, True],
    ["Buildings",       "3 Story",    0, 1, 1,    False,  3,    0.25, 3,  3, True],
    ["Buildings",       "3 Story",    0, 1, 2,    False,  3,    0.25, 4,  3, True],
    ["Buildings",       "3 Story",    0, 1, 3,    False,  3,    0.25, 3,  4, True],
    ["Buildings",       "3 Story",    0, 2, 0,    False,  3,    0.25, 4,  4, True],
    ["Buildings",       "3 Story",    0, 2, 1,    True,  3,    0.25,  2,  2, True],
    ["Buildings",       "3 Story",    0, 2, 2,    True,  3,    0.25,  2,  3, True],
    ["Buildings",       "3 Story",    0, 2, 3,    True,  3,    0.25,  3,  2, True],
    ["Buildings",       "3 Story",    0, 3, 0,    True,  3,    0.25,  3,  3, True],

    ["Buildings",       "3 Story",    1, 0, 0,    False,  3,    0.25, 2,  2, True],
    ["Buildings",       "3 Story",    1, 0, 1,    False,  3,    0.25, 2,  3, True],
    ["Buildings",       "3 Story",    1, 0, 2,    False,  3,    0.25, 3,  2, True],
    ["Buildings",       "3 Story",    1, 0, 3,    False,  3,    0.25, 2,  4, True],
    ["Buildings",       "3 Story",    1, 1, 0,    False,  3,    0.25, 4,  2, True],
    ["Buildings",       "3 Story",    1, 1, 1,    False,  3,    0.25, 3,  3, True],
    ["Buildings",       "3 Story",    1, 1, 2,    False,  3,    0.25, 3,  4, True],
    ["Buildings",       "3 Story",    1, 1, 3,    False,  3,    0.25, 4,  3, True],
    ["Buildings",       "3 Story",    1, 2, 0,    False,  3,    0.25, 4,  4, True],
    ["Buildings",       "3 Story",    1, 2, 1,    True,   3,    0.25, 2,  2, True],
    ["Buildings",       "3 Story",    1, 2, 2,    True,   3,    0.25, 3,  2, True],
    ["Buildings",       "3 Story",    1, 2, 3,    True,   3,    0.25, 2,  3, True],
    ["Buildings",       "3 Story",    1, 3, 0,    True,   3,    0.25, 3,  3, True],

    ["Buildings",       "4 Story",    1, 0, 0,    False,  4,    0.25, 2,  2, True],
    ["Buildings",       "4 Story",    1, 0, 1,    False,  4,    0.25, 2,  3, True],
    ["Buildings",       "4 Story",    1, 0, 2,    False,  4,    0.25, 3,  2, True],
    ["Buildings",       "4 Story",    1, 0, 3,    False,  4,    0.25, 2,  4, True],
    ["Buildings",       "4 Story",    1, 1, 0,    False,  4,    0.25, 4,  2, True],
    ["Buildings",       "4 Story",    1, 1, 1,    False,  4,    0.25, 3,  3, True],
    ["Buildings",       "4 Story",    1, 1, 2,    False,  4,    0.25, 3,  4, True],
    ["Buildings",       "4 Story",    1, 1, 3,    False,  4,    0.25, 4,  3, True],
    ["Buildings",       "4 Story",    1, 2, 0,    False,  4,    0.25, 4,  4, True],
    ["Buildings",       "4 Story",    1, 2, 1,    True,   4,    0.25, 2,  2, True],
    ["Buildings",       "4 Story",    1, 2, 2,    True,   4,    0.25, 3,  2, True],
    ["Buildings",       "4 Story",    1, 2, 3,    True,   4,    0.25, 2,  3, True],
    ["Buildings",       "4 Story",    1, 3, 0,    True,   4,    0.25, 3,  3, True],

    ["Buildings",       "4 Story",    1, 0, 0,    False,  4,    0.25, 2,  2, True],
    ["Buildings",       "4 Story",    1, 0, 1,    False,  4,    0.25, 2,  3, True],
    ["Buildings",       "4 Story",    1, 0, 2,    False,  4,    0.25, 3,  2, True],
    ["Buildings",       "4 Story",    1, 0, 3,    False,  4,    0.25, 2,  4, True],
    ["Buildings",       "4 Story",    1, 1, 0,    False,  4,    0.25, 4,  2, True],
    ["Buildings",       "4 Story",    1, 1, 1,    False,  4,    0.25, 3,  3, True],
    ["Buildings",       "4 Story",    1, 1, 2,    False,  4,    0.25, 3,  4, True],
    ["Buildings",       "4 Story",    1, 1, 3,    False,  4,    0.25, 4,  3, True],
    ["Buildings",       "4 Story",    1, 2, 0,    False,  4,    0.25, 4,  4, True],
    ["Buildings",       "4 Story",    1, 2, 1,    True,   4,    0.25, 2,  2, True],
    ["Buildings",       "4 Story",    1, 2, 2,    True,   4,    0.25, 3,  2, True],
    ["Buildings",       "4 Story",    1, 2, 3,    True,   4,    0.25, 2,  3, True],
    ["Buildings",       "4 Story",    1, 3, 0,    True,   4,    0.25, 3,  3, True],

]
