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

import pyautogui

UPPER_LEFT_SQUARE = pyautogui.Point(233,50)
LOWER_RIGHT_SQUARE = pyautogui.Point(633,451)

SQUARE_SIZE_X = 16
SQUARE_SIZE_Y = 16

START_N_SQUARES_X = 40
START_N_SQUARES_Y = 40

PAGE_N_SQUARES_X = 26
PAGE_N_SQUARES_Y = 26

PAGE_TOP_MARGIN = 2
PAGE_BOTTOM_MARGIN = 2
PAGE_RIGHT_MARGIN = 2

START_HEIGHT = 20

MAX_N_SQUARES_X_LEFT = 520 - PAGE_RIGHT_MARGIN
MAX_N_SQUARES_Y_UP = 520 - PAGE_TOP_MARGIN - PAGE_BOTTOM_MARGIN
