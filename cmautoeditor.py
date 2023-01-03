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

from time import sleep
import pyautogui
import numpy as np
import pandas
import argparse
import keyboard
import itertools
import collections

# constants:
UPPER_LEFT_SQUARE = pyautogui.Point(234,52)
UPPER_RIGHT_SQUARE = pyautogui.Point(1882,52)
LOWER_LEFT_SQUARE = pyautogui.Point(234,996)
LOWER_RIGHT_SQUARE = pyautogui.Point(1882,996)

SQUARE_SIZE_X = 16
SQUARE_SIZE_Y = 16

START_N_SQUARES_X = 40
START_N_SQUARES_Y = 40

PAGE_N_SQUARES_X = 104
PAGE_N_SQUARES_Y = 60

START_HEIGHT = 20

POS_HORIZONTAL_PLUS = pyautogui.Point(764, 10)
POS_HORIZONTAL_MINUS = pyautogui.Point(764, 26)

POS_VERTICAL_PLUS = pyautogui.Point(1014, 10)
POS_VERTICAL_MINUS = pyautogui.Point(903, 10)

pyautogui.PAUSE = 0.01  # 0.12 almost!! works

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-i', '--input', required=True, help='File containing input data in csv-Format. Data is coded in x, y and z columns.')
arg_parser.add_argument('-c', '--countdown', required=False, type=int, help='Countdown until CMAutoEditor starts clicking in CM.', default=5)

def _consume(iterator, n=None):
    "Advance the iterator n-steps ahead. If n is None, consume entirely."
    # from https://docs.python.org/3/library/itertools.html#recipes
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(itertools.islice(iterator, n, n), None)


def set_height(current_height, target_height):
    if current_height == target_height:
        return
    elif current_height < target_height:
        n_diff = target_height - current_height
        # pyautogui.press('+', presses=n_diff, interval=0.5)
        for i in range(n_diff):
            keyboard.send('+')
            sleep(0.1)
    else:
        n_diff = current_height - target_height
        # pyautogui.press('-', presses=n_diff, interval=0.5)
        for i in range(n_diff):
            keyboard.send('-')
            sleep(0.1)


    sleep(1)

def process_segment(grid, start_height):
    print(grid.shape)
    if len(grid[grid >= 0]) == 0:
        return start_height
        
    values = np.unique(grid[grid >= 0])
    min_height = np.min(grid[grid >= 0])
    
    set_height(start_height, min_height)
    height = min_height

    for val in values:
        indices0, indices1 = np.where(grid == val)

        set_height(height, val)

        for i in range(len(indices1)):
            idx0 = indices0[i]
            idx1 = indices1[i]
            x_pos = int(idx0 * SQUARE_SIZE_X + UPPER_LEFT_SQUARE.x)
            y_pos = int(LOWER_RIGHT_SQUARE.y - (PAGE_N_SQUARES_Y - grid.shape[1] + idx1) * SQUARE_SIZE_Y)
            height = val

            pyautogui.click(x=x_pos, y=y_pos)
            pyautogui.sleep(0.05)

    return height

def set_n_squares(start_n_x, start_n_y, n_x, n_y):
    n_clicks_x = abs(int((start_n_x - n_x) / 2))
    n_clicks_y = abs(int((start_n_y - n_y) / 2))


    x_click_iterator = range(n_clicks_x).__iter__()
    for i in x_click_iterator:
        shift_pressed = False
        if n_clicks_x - i >= 5:
            keyboard.press('shift')
            shift_pressed = True
            _consume(x_click_iterator, 4)
        if n_x <= start_n_x:
            pyautogui.click(POS_HORIZONTAL_MINUS)
        else:
            pyautogui.click(POS_HORIZONTAL_PLUS)
        pyautogui.sleep(0.05)

        if shift_pressed:
            keyboard.release('shift')
        pyautogui.sleep(0.05)

    
    y_click_iterator = range(n_clicks_y).__iter__()
    for i in y_click_iterator:
        shift_pressed = False
        if n_clicks_y - i >= 5:
            keyboard.press('shift')
            shift_pressed = True
            _consume(y_click_iterator, 4)
        if n_y <= start_n_y:
            pyautogui.click(POS_VERTICAL_MINUS)
        else:
            pyautogui.click(POS_VERTICAL_PLUS)
        pyautogui.sleep(0.05)

        if shift_pressed:
            keyboard.release('shift')
        pyautogui.sleep(0.05)

    sleep(0.5)


if __name__ == '__main__':
    args = arg_parser.parse_args()

    return_val = pyautogui.confirm(text='CMAutoEditor is about to run on {}.'
        '\nIf you haven\'t done so yet, open up the CM Scenario Editor, go to map->Elevation and click \'Direct\'. Make sure the size is 320m x 320m.'
        '\n\nOnce you are ready to start click \'Ok\'. You will then have {}s to switch back to the CM Scenario Editor.'
        '\n\nIn case something goes wrong, move the mouse cursor to one of the screen corners. This will stop CMAutoEditor.'.format(args.input, args.countdown), title='CMAutoEditor')

    if return_val == 'Cancel':
        exit()

    # load height map
    height_map_df = pandas.read_csv(args.input)

    x = np.array(height_map_df.x.values, dtype=int)
    y = np.array(height_map_df.y.values, dtype=int)
    z = height_map_df.z.values

    grid = np.full((x.max() + 1, y.max() + 1), -1)
    grid[x, y] = z

    n_pages_x, n_x_remain = np.divmod(grid.shape[0], PAGE_N_SQUARES_X)
    n_pages_y, n_y_remain = np.divmod(grid.shape[1], PAGE_N_SQUARES_Y)
    n_x_remain = (np.floor(n_x_remain / 2) * 2).astype(int)
    n_y_remain = (np.floor(n_y_remain / 2) * 2).astype(int)

    grid = grid[0:(n_pages_x * PAGE_N_SQUARES_X + n_x_remain), 0:(n_pages_y * PAGE_N_SQUARES_Y + n_y_remain)]

    height = START_HEIGHT
    prev_n_x = START_N_SQUARES_X
    prev_n_y = START_N_SQUARES_Y

    pyautogui.countdown(args.countdown)

    for i_page_y in range(n_pages_y + 1):
        for i_page_x in range(n_pages_x + 1):
            # if i_page_x > 3 or i_page_y > 3:
            #     continue

            xmax = grid.shape[0] - i_page_x * PAGE_N_SQUARES_X
            ymax = (i_page_y + 1) * PAGE_N_SQUARES_Y
            if i_page_x < n_pages_x:
                n_squares_x = (i_page_x + 1) * PAGE_N_SQUARES_X
                xmin = xmax - PAGE_N_SQUARES_X
            else:
                n_squares_x = i_page_x * PAGE_N_SQUARES_X + n_x_remain 
                xmin = xmax - n_x_remain
            if i_page_y < n_pages_y:
                n_squares_y = (i_page_y + 1) * PAGE_N_SQUARES_Y
                ymin = ymax - PAGE_N_SQUARES_Y
            else:
                n_squares_y = i_page_y * PAGE_N_SQUARES_Y + n_y_remain
                ymin = ymax - PAGE_N_SQUARES_Y

            set_n_squares(prev_n_x, prev_n_y, n_squares_x, n_squares_y)
            prev_n_x = n_squares_x
            prev_n_y = n_squares_y

            sub_grid = grid[xmin:xmax, ymin:ymax]

            height = process_segment(sub_grid, height)

    pyautogui.alert(text='CMAutoEditor has finished processing the input data.', title='CMAutoEditor')




