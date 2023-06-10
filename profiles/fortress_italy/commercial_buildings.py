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

# direction, row, column, is diagonal?, lower right point x2, upper right point x2, stories, len x, len y, is_modular

commercial_buildings = [  
    ["Independent Buildings",   "Commercial",    0, 0, 0,    False, 1,    1.0,    2,  2, False],
    ["Independent Buildings",   "Commercial",    0, 0, 1,    False, 1,    2.0,    2,  2, False],
    ["Independent Buildings",   "Commercial",    0, 0, 2,    False, 1,    1.0,    2,  2, False],
    ["Independent Buildings",   "Commercial",    0, 1, 0,    False, 1,    1.0,    2,  3, False],
    ["Independent Buildings",   "Commercial",    0, 1, 1,    True,  1,    1.0,    2,  2, False],
    ["Independent Buildings",   "Commercial",    0, 1, 2,    True,  1,    2.0,    2,  2, False],
    ["Independent Buildings",   "Commercial",    0, 2, 0,    True, 1,    1.0,    2,  2, False],
    ["Independent Buildings",   "Commercial",    0, 2, 1,    True,  1,    1.0,    3,  2, False],
    ["Independent Buildings",   "Commercial",    0, 2, 2,    False,  1,    2.0,    2,  3, False],
    ["Independent Buildings",   "Commercial",    0, 3, 0,    True,  1,    2.0,    3,  2, False],

    ["Independent Buildings",   "Commercial",    1, 0, 0,    False, 1,    1.0,    2,  2, False],
    ["Independent Buildings",   "Commercial",    1, 0, 1,    False, 1,    2.0,    2,  2, False],
    ["Independent Buildings",   "Commercial",    1, 0, 2,    False, 1,    1.0,    2,  2, False],
    ["Independent Buildings",   "Commercial",    1, 1, 0,    False, 1,    1.0,    3,  2, False],
    ["Independent Buildings",   "Commercial",    1, 1, 1,    True,  1,    1.0,    2,  2, False],
    ["Independent Buildings",   "Commercial",    1, 1, 2,    True,  1,    2.0,    2,  2, False],
    ["Independent Buildings",   "Commercial",    1, 2, 0,    True, 1,    1.0,    2,  2, False],
    ["Independent Buildings",   "Commercial",    1, 2, 1,    True,  1,    1.0,    2,  3, False],
    ["Independent Buildings",   "Commercial",    1, 2, 2,    False,  1,    2.0,    3,  2, False],
    ["Independent Buildings",   "Commercial",    1, 3, 0,    True,  1,    2.0,    2,  3, False],
]