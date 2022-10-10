#    Copyright 2022 Nicolas Möser

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License. 

from time import sleep
import pyautogui
import numpy as np
import pandas
import argparse
import keyboard

# constants:
UPPER_LEFT_SQUARE = pyautogui.Point(234,52)
UPPER_RIGHT_SQUARE = pyautogui.Point(1882,52)
LOWER_LEFT_SQUARE = pyautogui.Point(234,996)
LOWER_RIGHT_SQUARE = pyautogui.Point(1882,996)

SQUARE_SIZE_X = 16
SQUARE_SIZE_Y = 16

START_N_SQUARES_X = 40
START_N_SQUARES_Y = 40
# START_N_SQUARES_X = 104
# START_N_SQUARES_Y = 60

PAGE_N_SQUARES_X = 104
PAGE_N_SQUARES_Y = 60

START_HEIGHT = 20

POS_HORIZONTAL_PLUS = pyautogui.Point(764, 10)
POS_HORIZONTAL_MINUS = pyautogui.Point(764, 26)

POS_VERTICAL_PLUS = pyautogui.Point(1014, 10)
POS_VERTICAL_MINUS = pyautogui.Point(903, 10)

TILE_PAGE_TO_ARROW_DICT = {
    0: pyautogui.Point(248,17),
    1: pyautogui.Point(278,17),
    2: pyautogui.Point(308,17),
    3: pyautogui.Point(338,17),
}

ROAD_TYPE_DICT = {
    'Paved 1': pyautogui.Point(36,440),
    'Paved 2': pyautogui.Point(108,440),
    'Foot Path': pyautogui.Point(183,440),
}

TILE_ROW_DICT = {
    0: 615,
    1: 659,
    2: 696,
    3: 740
}

TILE_COL_DICT = {
    0: 36,
    1: 108,
    2: 182,
}

pyautogui.PAUSE = 0.2  # 0.12 almost!! works


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-i', '--input', required=True, help='File containing input data in csv-Format. Data is coded in x, y and z columns.')

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
    # values = np.unique(grid[grid >= 0])
    # min_height = np.min(grid[grid >= 0])
    values = grid.z.sort_values().unique()
    min_height = grid[grid.z >= 0].z.min()
    
    set_height(start_height, min_height)
    height = min_height

    for val in values:
        grid_extract = grid[grid.z == val]
        # indices0, indices1 = np.where(grid == val)

        set_height(height, val)

        for ridx, row in grid_extract.iterrows():
            # idx0 = indices0[i]
            # idx1 = indices1[i]
            idx0 = row.x
            idx1 = row.y
            x_pos = int(idx0 * SQUARE_SIZE_X + UPPER_LEFT_SQUARE.x)
            y_pos = int(LOWER_RIGHT_SQUARE.y - idx1 * SQUARE_SIZE_Y)
            height = val

            pyautogui.click(x=x_pos, y=y_pos)

    return height

def set_n_squares(start_n_x, start_n_y, n_x, n_y):
    n_clicks_x = abs(int((start_n_x - n_x) / 2))
    n_clicks_y = abs(int((start_n_y - n_y) / 2))

    for i in range(n_clicks_x):
        if n_x <= start_n_x:
            pyautogui.click(POS_HORIZONTAL_MINUS, interval=0.2)
        else:
            pyautogui.click(POS_HORIZONTAL_PLUS, interval=0.2)
    
    for i in range(n_clicks_y):
        if n_y <= start_n_y:
            pyautogui.click(POS_VERTICAL_MINUS, interval=0.2)
        else:
            pyautogui.click(POS_VERTICAL_PLUS, interval=0.2)

def set_roads(df):
    # for name, group in df.groupby(by='id'):
    #     prev_x = None
    #     prev_y = None
    #     for ridx, row in group.iterrows():
    #         x_pos = int(row.x * SQUARE_SIZE_X + UPPER_LEFT_SQUARE.x)
    #         y_pos = int(LOWER_RIGHT_SQUARE.y - row.y * SQUARE_SIZE_Y)
    #         if prev_x is not None and prev_y is not None:
    #             pyautogui.click(x=prev_x, y=prev_y, interval=0.5)
    #             pyautogui.click(x=x_pos, y=y_pos, interval=0.5)
    #             sleep(0.5)
    #             pass

    #         prev_x = x_pos
    #         prev_y = y_pos
        
    #     a = 1
    for tile_info, group in df.groupby(by=['type', 'sub_type', 'tile_page', 'tile_row', 'tile_col']):
        if tile_info[0] == 'road' and tile_info[2] > -1:
            road_sub_type = ROAD_TYPE_DICT[tile_info[1]]
            tile_page_point = TILE_PAGE_TO_ARROW_DICT[tile_info[2]]
            tile_pos_x = TILE_COL_DICT[tile_info[4]]
            tile_pos_y = TILE_ROW_DICT[tile_info[3]]
            pyautogui.click(road_sub_type, interval=0.25)
            pyautogui.click(tile_page_point, interval=0.25)
            pyautogui.click(x=tile_pos_x, y=tile_pos_y, interval=0.25)
            for _, row in group.iterrows():
                x_pos = int(row.x * SQUARE_SIZE_X + UPPER_LEFT_SQUARE.x)
                y_pos = int(LOWER_RIGHT_SQUARE.y - row.y * SQUARE_SIZE_Y)
                pyautogui.click(x=x_pos, y=y_pos)


if __name__ == '__main__':
    args = arg_parser.parse_args()

    # load height map
    map_df = pandas.read_csv(args.input)
    map_df.z = map_df.z.round().astype(int)

    # x = np.array(map_df.x.values, dtype=int)
    # y = np.array(map_df.y.values, dtype=int)
    # z = map_df.z.values

    # grid = np.full((x.max() + 1, y.max() + 1), -1)
    # grid[x, y] = z
    total_n_squares_x = map_df.x.max() + 1
    total_n_squares_y = map_df.y.max() + 1

    n_pages_x, n_x_remain = np.divmod(total_n_squares_x, PAGE_N_SQUARES_X)
    n_pages_y, n_y_remain = np.divmod(total_n_squares_y, PAGE_N_SQUARES_Y)
    n_x_remain = (np.floor(n_x_remain / 2) * 2).astype(int)
    n_y_remain = (np.floor(n_y_remain / 2) * 2).astype(int)

    # grid = grid[0:(n_pages_x * PAGE_N_SQUARES_X + n_x_remain), 0:(n_pages_y * PAGE_N_SQUARES_Y + n_y_remain)]

    map_df = map_df[(map_df.x >= 0) & (map_df.y >= 0) & (map_df.x < (n_pages_x * PAGE_N_SQUARES_X + n_x_remain)) & (map_df.y < (n_pages_y * PAGE_N_SQUARES_Y + n_y_remain))]

    # map_df = map_df[map_df.x.between(0, (n_pages_x * PAGE_N_SQUARES_X + n_x_remain), inclusive='left')]
    # map_df = map_df[map_df.y.between(0, (n_pages_y * PAGE_N_SQUARES_Y + n_y_remain), inclusive='left')]
                    
    total_n_squares_x = n_pages_x * PAGE_N_SQUARES_X + n_x_remain
    total_n_squares_y = n_pages_y * PAGE_N_SQUARES_Y + n_y_remain

    height = START_HEIGHT
    prev_n_x = START_N_SQUARES_X
    prev_n_y = START_N_SQUARES_Y

    pyautogui.countdown(5)

    for i_page_y in range(n_pages_y + 1):
        for i_page_x in range(n_pages_x + 1):
            # if i_page_x > 3 or i_page_y > 3:
            #     continue

            # xmax = total_n_squares_x - i_page_x * PAGE_N_SQUARES_X
            # ymax = (i_page_y + 1) * PAGE_N_SQUARES_Y
            # if i_page_x < n_pages_x:
            #     n_squares_x = (i_page_x + 1) * PAGE_N_SQUARES_X
            #     xmin = xmax - PAGE_N_SQUARES_X
            # else:
            #     n_squares_x = i_page_x * PAGE_N_SQUARES_X + n_x_remain 
            #     xmin = xmax - n_x_remain
            # if i_page_y < n_pages_y:
            #     n_squares_y = (i_page_y + 1) * PAGE_N_SQUARES_Y
            #     ymin = ymax - PAGE_N_SQUARES_Y
            # else:
            #     n_squares_y = i_page_y * PAGE_N_SQUARES_Y + n_y_remain
            #     ymin = ymax - PAGE_N_SQUARES_Y


            if i_page_x < n_pages_x:
                n_squares_x = (i_page_x + 1) * PAGE_N_SQUARES_X
                xmax = total_n_squares_x - i_page_x * PAGE_N_SQUARES_X
                xmin = xmax - PAGE_N_SQUARES_X
                origin_x = total_n_squares_x - (i_page_x + 1) * PAGE_N_SQUARES_X
            else:
                n_squares_x = i_page_x * PAGE_N_SQUARES_X + n_x_remain 
                xmax = n_x_remain
                xmin = 0
                origin_x = 0
            if i_page_y < n_pages_y:
                n_squares_y = (i_page_y + 1) * PAGE_N_SQUARES_Y
                ymax = (i_page_y + 1) * PAGE_N_SQUARES_Y
                ymin = ymax - PAGE_N_SQUARES_Y
                origin_y = i_page_y * PAGE_N_SQUARES_Y
            else:
                n_squares_y = i_page_y * PAGE_N_SQUARES_Y + n_y_remain
                ymax = total_n_squares_y
                ymin = total_n_squares_y - n_y_remain
                origin_y = total_n_squares_y - PAGE_N_SQUARES_Y

            if xmax == xmin or ymax == ymin:
                continue
            # if i_page_x == 0 and i_page_y == 0:
            #     continue

            set_n_squares(prev_n_x, prev_n_y, n_squares_x, n_squares_y)
            prev_n_x = n_squares_x
            prev_n_y = n_squares_y

            # sub_grid = grid[xmin:xmax, ymin:ymax]

            # height = process_segment(sub_grid, height)
            sub_df = map_df[map_df.x.between(xmin, xmax, inclusive='left') & map_df.y.between(ymin, ymax, inclusive='left')].copy(deep=True)
            sub_df.x = sub_df.x - origin_x
            sub_df.y = sub_df.y - origin_y
            # height = process_segment(sub_df, height)
            set_roads(sub_df)




